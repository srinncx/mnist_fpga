import serial
import numpy as np
import random
from torchvision import datasets, transforms
import time

# -----------------------------
# CHANGE THIS TO YOUR COM PORT
# -----------------------------
COM_PORT = "COM6"
BAUD_RATE = 115200

# Open UART
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)

# SAME NORMALIZATION USED IN TRAINING
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST test dataset
mnist = datasets.MNIST(
    root=".",
    train=False,
    download=True,
    transform=transform
)

# Pick random image
idx = random.randint(0, len(mnist) - 1)
img, label = mnist[idx]

# Convert to signed int8 (-128 → 127)
img_array = img.numpy().reshape(784)

# Already in -1 → +1 range
img_scaled = np.round(img_array * 127)

img_int8 = np.clip(img_scaled, -128, 127).astype(np.int8)

print("Sending MNIST image index:", idx)
print("True label:", label)

# Send exactly 784 signed bytes
ser.write(img_int8.tobytes())
ser.flush()
time.sleep(0.05)

# Read FPGA prediction (1 byte)
pred = ser.read(1)

if len(pred) == 1:
    print("FPGA prediction:", pred[0])
else:
    print("No response from FPGA")

ser.close()