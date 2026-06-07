"""
show_d.py
---------
Shows what a confirmed D image looks like on the canvas
so you know exactly how to draw it.

Also opens the canvas with the D pre-loaded so you can
click Send and confirm it works, then clear and redraw.

Put this in the same folder as test_d_1.txt
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import threading
import os

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
COM_PORT  = "COM5"
BAUD      = 115200
CANVAS_PX = 308
GRID      = 28
CELL      = CANVAS_PX // GRID
BRUSH     = 2

BG    = "#1a1a2e"
PANEL = "#16213e"
ACCENT= "#0f3460"
EMPTY = "#0d1b2a"
OK_C  = "#00d4aa"
ERR_C = "#e94560"
MUTED = "#666688"

# ── Load confirmed D image ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
d_path = os.path.join(script_dir, "test_d_1.txt")

if not os.path.exists(d_path):
    print(f"ERROR: test_d_1.txt not found in {script_dir}")
    exit(1)

pixels = np.loadtxt(d_path, dtype=np.float32)

# Flip horizontally so the b-shape becomes a D on screen
# The raw data looks like b — fliplr mirrors it to look like D
raw_grid = (pixels.reshape(28, 28) / 255.0).astype(np.float32)
grid = np.fliplr(raw_grid).copy()

print("Loaded test_d_1.txt and flipped horizontally")
print("Now it looks like a D on screen")
print(f"Min pixel: {pixels.min():.0f}  Max pixel: {pixels.max():.0f}")

# ── Serial ────────────────────────────────────────────────────────────────────
def send_pixels(pixels_bytes: bytes) -> str:
    if not SERIAL_AVAILABLE:
        return "ERROR: pyserial not installed"
    try:
        with serial.Serial(COM_PORT, BAUD, timeout=10) as ser:
            ser.reset_input_buffer()
            ser.write(pixels_bytes)
            ser.flush()
            result = ser.read(1)
            if len(result) == 0:
                return "ERROR: timeout"
            return "D DETECTED" if result[0] == 1 else "NOT D"
    except serial.SerialException as e:
        return f"ERROR: {e}"

def grid_to_bytes() -> bytes:
    # flip back before sending so the network receives the original orientation
    original = np.fliplr(grid)
    return bytes(np.clip(original * 255.0, 0, 255).astype(np.uint8).flatten())

# ── UI ────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("D Shape Viewer — draw like this!")
root.configure(bg=BG)
root.resizable(False, False)

hf  = tkfont.Font(family="Helvetica", size=11, weight="bold")
sf  = tkfont.Font(family="Helvetica", size=9)
ssf = tkfont.Font(family="Helvetica", size=8)
rf  = tkfont.Font(family="Helvetica", size=16, weight="bold")

tk.Label(root, text="This is what a D looks like to the network",
         font=hf, fg="#ffffff", bg=BG).pack(pady=(10,2), padx=16)
tk.Label(root, text="Step 1: Click Send — it should say D DETECTED",
         font=ssf, fg=OK_C, bg=BG).pack(padx=16)
tk.Label(root, text="Step 2: Click Clear, then draw this same D shape",
         font=ssf, fg="#aaaaaa", bg=BG).pack(padx=16, pady=(0,6))

# Canvas
cv_frame = tk.Frame(root, bg=BG, padx=16)
cv_frame.pack()

cv = tk.Canvas(cv_frame, width=CANVAS_PX, height=CANVAS_PX,
               bg=EMPTY, cursor="crosshair",
               highlightthickness=1, highlightbackground=ACCENT)
cv.pack()

for i in range(1, GRID):
    cv.create_line(i*CELL, 0, i*CELL, CANVAS_PX, fill="#111827", width=1)
    cv.create_line(0, i*CELL, CANVAS_PX, i*CELL, fill="#111827", width=1)

rects = [[None]*GRID for _ in range(GRID)]
for r in range(GRID):
    for c in range(GRID):
        x0, y0 = c*CELL, r*CELL
        rects[r][c] = cv.create_rectangle(x0, y0, x0+CELL, y0+CELL,
                                           fill=EMPTY, outline="")

def refresh_canvas():
    for r in range(GRID):
        for c in range(GRID):
            v = int(grid[r, c] * 255)
            cv.itemconfig(rects[r][c], fill=f"#{v:02x}{v:02x}{v:02x}")

def paint(event, erase=False):
    gc = int(event.x / CELL)
    gr = int(event.y / CELL)
    for dr in range(-BRUSH, BRUSH + 1):
        for dc in range(-BRUSH, BRUSH + 1):
            nr, nc = gr+dr, gc+dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                dist = (dr**2 + dc**2) ** 0.5
                strength = max(0.0, 1.0 - dist / (BRUSH + 0.5))
                if erase:
                    grid[nr, nc] = max(0.0, grid[nr, nc] - strength * 0.6)
                else:
                    grid[nr, nc] = min(1.0, grid[nr, nc] + strength * 0.6)
    refresh_canvas()

cv.bind("<B1-Motion>",  lambda e: paint(e, erase=False))
cv.bind("<Button-1>",   lambda e: paint(e, erase=False))
cv.bind("<B3-Motion>",  lambda e: paint(e, erase=True))
cv.bind("<Button-3>",   lambda e: paint(e, erase=True))

# Load the flipped D image onto canvas
refresh_canvas()

# Result label
result_var = tk.StringVar(value="Click Send to test")
result_lbl = tk.Label(root, textvariable=result_var,
                      font=rf, fg=MUTED, bg=BG)
result_lbl.pack(pady=4)

# Buttons
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=6, padx=16, fill="x")

status_var = tk.StringVar(value="Ready")

def on_send():
    status_var.set("Sending...")
    result_var.set("...")
    result_lbl.config(fg=MUTED)
    root.update_idletasks()
    def worker():
        res = send_pixels(grid_to_bytes())
        root.after(0, lambda: result_var.set(res))
        root.after(0, lambda: result_lbl.config(
            fg=OK_C if res == "D DETECTED" else ERR_C))
        root.after(0, lambda: status_var.set("Done"))
    threading.Thread(target=worker, daemon=True).start()

def on_clear():
    grid[:] = 0.0
    refresh_canvas()
    result_var.set("Canvas cleared — draw your D now")
    result_lbl.config(fg=MUTED)

def on_reload():
    pixels2 = np.loadtxt(d_path, dtype=np.float32)
    raw = (pixels2.reshape(28, 28) / 255.0).astype(np.float32)
    grid[:] = np.fliplr(raw)
    refresh_canvas()
    result_var.set("Reloaded reference D")
    result_lbl.config(fg=MUTED)

def make_btn(text, cmd, bg_col, width=12):
    return tk.Button(btn_frame, text=text, command=cmd,
                     bg=bg_col, fg="#ffffff",
                     activebackground="#334466",
                     relief="flat", font=sf, width=width,
                     padx=6, pady=6, cursor="hand2", bd=0)

make_btn("Send to FPGA", on_send, ACCENT, 14).pack(side="left", padx=(0,6))
make_btn("Clear", on_clear, "#2a2a44", 8).pack(side="left", padx=(0,6))
make_btn("Reload D", on_reload, "#1a4a2a", 8).pack(side="left")

tk.Label(root, textvariable=status_var, font=ssf,
         fg=MUTED, bg=BG).pack(pady=(2,8))

root.mainloop()