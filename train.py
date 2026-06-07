import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

DEVICE     = torch.device("cpu")
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3
D_LABEL    = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1,2])),
    transforms.Lambda(lambda x: x.reshape(-1))
])

class BinaryEMNIST(Dataset):
    def __init__(self, split):
        self.data = datasets.EMNIST(
            root="./data", split="letters",
            train=(split == "train"),
            download=True, transform=transform
        )
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, torch.tensor(1.0 if label == D_LABEL else 0.0)

train_ds = BinaryEMNIST("train")
test_ds  = BinaryEMNIST("test")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

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

model     = LetterDNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(DEVICE))
            correct += ((preds >= 0.5).float() == labels.to(DEVICE)).sum().item()
            total   += labels.size(0)

    print(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_acc={100*correct/total:.2f}%")

torch.save(model.state_dict(), "model.pth")
print("Done. Saved model.pth")