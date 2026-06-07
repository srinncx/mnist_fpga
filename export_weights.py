import torch
import numpy as np

# ── Fixed-point spec ──────────────────────────────────────────────────────────
#  Format  : Q6.10  (matches ap_fixed<16,6> in nn.h)
#  Meaning : 1 sign bit + 5 integer bits + 10 fractional bits
#  Scale   : 2^10 = 1024
#  Range   : -32.0  to  +31.999023...
#  Storage : int16_t raw bit patterns in weights.h
#            Use to_fixed() helper in nn.cpp to load without value conversion.
# ─────────────────────────────────────────────────────────────────────────────
FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS        # 1024
MAX_RAW   =  (1 << 15) - 1        #  32767
MIN_RAW   = -(1 << 15)            # -32768


def to_q6_10(arr: np.ndarray) -> np.ndarray:
    """Quantise a float array to Q6.10 int16 (saturating round-to-nearest)."""
    raw = np.round(arr.astype(np.float64) * SCALE)
    return np.clip(raw, MIN_RAW, MAX_RAW).astype(np.int16)


def fmt_array(name: str, arr: np.ndarray, cols: int = 8) -> str:
    """Emit a C int16_t array initialiser."""
    flat  = arr.flatten()
    lines = [f"const int16_t {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, cols):
        row = ", ".join(str(int(v)) for v in flat[i:i + cols])
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


# ── Load PyTorch weights ──────────────────────────────────────────────────────
state = torch.load("model.pth", map_location="cpu")

layers = [
    ("W1", state["net.0.weight"].numpy()),   # (64, 784)
    ("b1", state["net.0.bias"].numpy()),     # (64,)
    ("W2", state["net.2.weight"].numpy()),   # (32, 64)
    ("b2", state["net.2.bias"].numpy()),     # (32,)
    ("W3", state["net.4.weight"].numpy()),   # (1,  32)
    ("b3", state["net.4.bias"].numpy()),     # (1,)
]

# ── Diagnostic printout ───────────────────────────────────────────────────────
print(f"Q6.10 format: scale={SCALE}, raw range=[{MIN_RAW}, {MAX_RAW}]")
print()
any_sat = False
for name, arr in layers:
    raw = to_q6_10(arr)
    sat = int(np.sum(np.abs(arr) * SCALE > MAX_RAW))
    any_sat = any_sat or (sat > 0)
    print(f"  {name:2s}: float [{arr.min():+.4f}, {arr.max():+.4f}]"
          f"  int16 [{raw.min():6d}, {raw.max():6d}]"
          f"  saturated={sat}")

if any_sat:
    print("\nWARNING: some values saturated — consider widening INT_BITS.")
else:
    print("\nAll values fit in Q6.10 without saturation.")

# ── Write weights.h ───────────────────────────────────────────────────────────
lines = [
    "#ifndef WEIGHTS_H",
    "#define WEIGHTS_H",
    "",
    "#include <stdint.h>",
    "",
    "/*",
    " * Network weights in Q6.10 fixed-point (ap_fixed<16,6>, SCALE=1024).",
    " * Arrays hold raw int16_t bit patterns — NOT real-valued integers.",
    " *",
    " * Load in nn.cpp via the to_fixed() helper:",
    " *",
    " *   static inline fixed_t to_fixed(int16_t raw) {",
    " *       ap_int<16> bits = raw;",
    " *       fixed_t f;",
    " *       f.range(15,0) = bits.range(15,0);",
    " *       return f;",
    " *   }",
    " *",
    " * This performs a bit-copy, so value = raw / 1024 as intended.",
    " * Direct fixed_t(raw) would treat raw as a real value and overflow.",
    " */",
    "",
]

for name, arr in layers:
    lines.append(fmt_array(name, to_q6_10(arr)))
    lines.append("")

lines += ["#endif /* WEIGHTS_H */", ""]

with open("weights.h", "w") as f:
    f.write("\n".join(lines))

print("\nweights.h written successfully.")