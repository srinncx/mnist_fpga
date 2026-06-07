
# 🧠 MNIST FPGA Classifier (Basys-3)

This project implements a fully working handwritten digit classifier (MNIST) on the **Basys-3 FPGA board (Artix-7 35T)** using:

- ✅ Logistic Regression (784 → 10)
- ✅ Signed 8-bit fixed-point arithmetic
- ✅ Single DSP-based MAC reused 10 times
- ✅ BRAM-based weight storage
- ✅ UART image streaming from Python
- ✅ 7-segment display output

The system classifies a 28×28 image in ~78 µs.

---

# 🏗 System Architecture

```

Python (MNIST image)
↓
UART (115200 baud)
↓
FPGA input memory (784 bytes)
↓
Single MAC Engine (784×10 cycles)
↓
Bias addition
↓
Argmax (10-cycle FSM)
↓
7-Segment Display

```

---

# 🧮 Neural Network Model

We implement:

```

Input:  784 pixels (28×28)
Output: 10 classes (digits 0–9)

```

For each neuron:

```

Y[j] = Σ (input[i] × weight[j][i]) + bias[j]

```

Classification:

```

predicted_digit = argmax(Y[0..9])

```



---

# 🔢 Fixed-Point Design

| Item | Format |
|------|--------|
| Input pixel | signed int8 (-128 to 127) |
| Weight | signed int8 |
| Product | signed int16 |
| Accumulator | signed int32 |

Worst case accumulation ≈ 12.6M  
32-bit ensures no overflow.

---

# 🧠 DSP Usage

The Artix-7 35T FPGA has **90 DSP48E1 slices**.

This design uses:

- 1 DSP slice (for multiplication)
- Reused 7840 times

Vivado automatically maps:

```verilog
product = pixel_data * weight_data;
````

into a DSP slice.

---

# 🗄 Memory Usage

Weights:

```
7840 weights × 8 bits = 62,720 bits
```

Bias:

```
10 × 8 bits = 80 bits
```

Stored in BRAM using:

```verilog
$readmemh("weights.mem", weights_mem);
$readmemh("bias.mem", bias_mem);
```

Basys-3 has ~1.8 Mbits of BRAM — well within limits.

---

# 🚀 Performance

Clock: 100 MHz

| Stage                   | Cycles |
| ----------------------- | ------ |
| 784 pixels × 10 neurons | 7840   |
| Argmax                  | 10     |
| Total                   | ~7850  |

Latency:

```
7850 / 100 MHz ≈ 78 µs
```

---

# 🖥 Python Training

Model:

```python
nn.Linear(784, 10)
```

Training uses normalized input:

```
(x - 0.5) / 0.5 → range (-1 to +1)
```

Quantization:

```python
SCALE = 64
W_q = clip(round(W * SCALE), -128, 127)
```

Output:

* `weights.mem` (7840 lines)
* `bias.mem` (10 lines)

---

# 🔌 UART Interface

Baud rate: **115200**

Clock: **100 MHz**

```
CLKS_PER_BIT = 868
```

Frame format:

```
1 start bit
8 data bits (LSB first)
1 stop bit
```

784 bytes streamed per image.

---

# 🖲 Display

Basys-3 7-segment display:

* Active LOW segments
* Active LOW anodes

Only first digit is enabled.

Predicted value (0–9) shown directly.

---



