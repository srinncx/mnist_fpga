"""
verify_weights.py
-----------------
Reads weights.h (Q6.10 int16 arrays) and verifies fixed-point inference
accuracy against the EMNIST test set.

Does NOT import export_weights.py.
Does NOT write or modify weights.h.
Does NOT load model.pth.

Fixed-point arithmetic mirrors nn.cpp exactly:

  nn.cpp per neuron:
      acc_t sum = to_fixed(b[j]);       // sum STARTS as bias
      for i: sum += w[j*N+i] * x[i];   // accumulate products
      h[j] = (sum > 0) ? sum : 0;       // ONE ReLU on (bias + W@x) together

  Python equivalent:
      h_raw = (W @ x) >> FRAC_BITS      // Q12.20 -> Q6.10
      h_raw = h_raw + bias              // add Q6.10 bias BEFORE clamping
      h     = max(0, h_raw)             // ONE ReLU

  The critical rule: bias is added BEFORE ReLU, not after.
  Adding bias after ReLU (common mistake) makes silent neurons fire,
  corrupts activations, and causes false negatives.

Run:
    python verify_weights.py
"""

import re
import numpy as np
import torch
from torchvision import datasets, transforms

# ── Q6.10 spec ────────────────────────────────────────────────────────────────
FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS   # 1024
D_LABEL   = 4
N_EACH    = 200               # collect 200 D + 200 not-D samples

# ── Layer shapes ──────────────────────────────────────────────────────────────
SHAPES = {
    "W1": (64, 784),
    "b1": (64,),
    "W2": (32, 64),
    "b2": (32,),
    "W3": (1,  32),
    "b3": (1,),
}


# ── Parse weights.h ───────────────────────────────────────────────────────────
def parse_weights_h(path: str = "weights.h") -> dict:
    """Read int16_t arrays from the C header. Returns dict -> np.ndarray (int64)."""
    with open(path, "r") as f:
        src = f.read()

    weights = {}
    pattern = re.compile(
        r"const\s+int16_t\s+(\w+)\s*\[\d+\]\s*=\s*\{([^}]+)\}",
        re.DOTALL
    )
    for m in pattern.finditer(src):
        name   = m.group(1)
        values = np.array([int(v) for v in m.group(2).split(",") if v.strip()],
                          dtype=np.int64)
        if name in SHAPES:
            weights[name] = values.reshape(SHAPES[name])

    missing = [k for k in SHAPES if k not in weights]
    if missing:
        raise ValueError(f"weights.h is missing arrays: {missing}")
    return weights


# ── Q6.10 quantise ────────────────────────────────────────────────────────────
def to_q6_10(arr: np.ndarray) -> np.ndarray:
    raw = np.round(arr.astype(np.float64) * SCALE)
    return np.clip(raw, -(1 << 15), (1 << 15) - 1).astype(np.int64)


# ── Fixed-point inference ─────────────────────────────────────────────────────
def infer_fixed(pixel_bytes: np.ndarray, W: dict) -> int:
    """
    pixel_bytes : uint8 array (784,).
    Returns 1 = D, 0 = Not-D.

    Key: bias added BEFORE ReLU — exactly as nn.cpp does it.
    """
    # Normalise: /256 matches HLS `ap_ufixed tmp = pixel; x[i] = tmp >> 8`
    x = to_q6_10(pixel_bytes.astype(np.float64) / 256.0)

    # Layer 1 : Linear(784 -> 64) + ReLU
    # (W1 @ x) is Q12.20 — shift right 10 to get Q6.10
    # Add Q6.10 bias BEFORE clamping (mirrors: sum = bias; sum += w*x; relu(sum))
    h1 = (W["W1"] @ x) >> FRAC_BITS    # Q12.20 -> Q6.10
    h1 = np.maximum(0, h1 + W["b1"])   # bias first, then ONE relu

    # Layer 2 : Linear(64 -> 32) + ReLU
    h2 = (W["W2"] @ h1) >> FRAC_BITS
    h2 = np.maximum(0, h2 + W["b2"])

    # Output  : Linear(32 -> 1), threshold at 0
    out = int((W["W3"] @ h2)[0] >> FRAC_BITS) + int(W["b3"][0])
    return 1 if out >= 0 else 0


# ── Load weights ──────────────────────────────────────────────────────────────
print("Reading weights.h ...")
W = parse_weights_h("weights.h")

print("Arrays loaded:")
for name, arr in W.items():
    q_min, q_max = int(arr.min()), int(arr.max())
    print(f"  {name:2s}  shape={str(arr.shape):12s}  "
          f"int16 [{q_min:7d}, {q_max:6d}]  "
          f"float [{q_min/SCALE:+.4f}, {q_max/SCALE:+.4f}]")


# ── Load EMNIST ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1, 2]))
])

test_ds = datasets.EMNIST(root="./data", split="letters",
                           train=False, download=False,
                           transform=transform)

# Scan full dataset — EMNIST is sorted by label so [:2000] contains zero D's
print(f"Scanning full test set for up to {N_EACH} D + {N_EACH} not-D samples ...")
d_pixels, notd_pixels = [], []

for img, label in test_ds:
    if len(d_pixels) >= N_EACH and len(notd_pixels) >= N_EACH:
        break
    pixels = (img.numpy().flatten() * 255).astype(np.uint8)
    if label == D_LABEL and len(d_pixels) < N_EACH:
        d_pixels.append(pixels)
    elif label != D_LABEL and len(notd_pixels) < N_EACH:
        notd_pixels.append(pixels)

print(f"Collected: {len(d_pixels)} D samples, {len(notd_pixels)} not-D samples.")

# ── Run inference ─────────────────────────────────────────────────────────────
tp = fp = tn = fn = 0

for px in d_pixels:
    (tp if infer_fixed(px, W) == 1 else fn).__class__  # just to keep it concise
    if infer_fixed(px, W) == 1: tp += 1
    else:                        fn += 1

for px in notd_pixels:
    if infer_fixed(px, W) == 0: tn += 1
    else:                        fp += 1

total     = tp + fp + tn + fn
correct   = tp + tn
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) else 0.0)

print(f"\nFixed-point accuracy : {100 * correct / total:.2f}%  ({correct}/{total})")
print(f"  True  positives (D     -> D    ) : {tp}")
print(f"  False positives (not-D -> D    ) : {fp}")
print(f"  True  negatives (not-D -> not-D) : {tn}")
print(f"  False negatives (D     -> not-D) : {fn}")
print(f"  Precision : {100 * precision:.1f}%")
print(f"  Recall    : {100 * recall:.1f}%")
print(f"  F1 score  : {100 * f1:.1f}%")