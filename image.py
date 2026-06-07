import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np

D_LABEL = 4
N_EACH  = 50   # 50 'd' + 50 non-d = 100 total

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1,2]))
])

ds = datasets.EMNIST(root="./data", split="letters",
                     train=False, download=False,
                     transform=transform)

class LetterDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

state = torch.load("model.pth", map_location="cpu")
model = LetterDNet()
model.load_state_dict(state)
model.eval()

d_count    = 0
notd_count = 0

for img, label in ds:
    if d_count >= N_EACH and notd_count >= N_EACH:
        break

    with torch.no_grad():
        prob = model(img.reshape(1, -1)).item()

    pixels = (img.numpy().flatten() * 255).astype(int)

    if label == D_LABEL and d_count < N_EACH and prob >= 0.5:
        d_count += 1
        fname = f"test_d_{d_count}.txt"
        np.savetxt(fname, pixels, fmt="%d")
        print(f"Saved {fname}  label={label}  prob={prob:.4f}")

    if label != D_LABEL and notd_count < N_EACH and prob < 0.5:
        notd_count += 1
        fname = f"test_notd_{notd_count}.txt"
        np.savetxt(fname, pixels, fmt="%d")
        print(f"Saved {fname}  label={label}  prob={prob:.4f}")

print(f"\nDone. Saved {d_count} 'd' files and {notd_count} non-d files.")