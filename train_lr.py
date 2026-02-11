# ============================================================
# Logistic Regression MNIST Training for FPGA (SIGNED INT8)
# Produces weights.mem (7840 lines) and bias.mem (10 lines)
# ============================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ------------------------------------------------------------
# 1. MODEL DEFINITION (MATCHES FPGA EXACTLY)
# ------------------------------------------------------------
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)  # 784 → 10

    def forward(self, x):
        return self.fc(x)  # No activation (FPGA uses argmax)


# ------------------------------------------------------------
# 2. LOAD MNIST DATASET (ZERO-CENTERED INPUT)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),              # 0 → 1
    transforms.Normalize((0.5,), (0.5,))  # → -1 → +1
])

trainset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# ------------------------------------------------------------
# 3. INITIALIZE MODEL
# ------------------------------------------------------------
model = LogisticRegression()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0.0

    for images, labels in trainloader:
        images = images.view(-1, 784)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss = {total_loss:.4f}")

# ------------------------------------------------------------
# 5. TEST ACCURACY
# ------------------------------------------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.view(-1, 784)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ------------------------------------------------------------
# 6. EXTRACT WEIGHTS & BIAS
# ------------------------------------------------------------
W = model.fc.weight.detach().numpy()  # (10, 784)
B = model.fc.bias.detach().numpy()    # (10,)

# ------------------------------------------------------------
# 7. QUANTIZATION (FLOAT → INT8 SAFE)
# ------------------------------------------------------------
SCALE = 64  # adjust if needed

W_q = np.clip(np.round(W * SCALE), -128, 127).astype(np.int8)
B_q = np.clip(np.round(B * SCALE), -128, 127).astype(np.int8)

# ------------------------------------------------------------
# 8. WRITE .MEM FILES (HEX, TWO'S COMPLEMENT)
# ------------------------------------------------------------

# weights.mem → 7840 lines
with open("weights.mem", "w") as f:
    for d in range(10):
        for i in range(784):
            val = int(W_q[d, i]) & 0xFF
            f.write(f"{val:02x}\n")

# bias.mem → 10 lines
with open("bias.mem", "w") as f:
    for d in range(10):
        val = int(B_q[d]) & 0xFF
        f.write(f"{val:02x}\n")

print("\nFPGA weight files generated successfully!")
print("  → weights.mem (7840 lines)")
print("  → bias.mem (10 lines)")