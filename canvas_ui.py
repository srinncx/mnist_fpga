"""
showd1_branded.py
-----------------
FPGA 28x28 MNIST Character Classifier — Anthropic light theme.
Fully scales to any screen. Tested for 1920x1200 fullscreen.
COM port selector with live scan.
Loads all 50 reference D images and cycles randomly.

Drop in the same folder as test_d_1.txt ... test_d_50.txt
"""

import tkinter as tk
from tkinter import font as tkfont, ttk
import numpy as np
import threading
import random
import os

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
BAUD  = 115200
GRID  = 28
BRUSH = 2

# ── Anthropic Brand Palette (light theme) ────────────────────────────────────
BG_MAIN    = "#faf9f5"
BG_PANEL   = "#f0ede4"
BG_CARD    = "#e8e6dc"
ORANGE     = "#d97757"
ORANGE_DIM = "#a85a3e"
TEXT_DARK  = "#141413"
TEXT_MUTED = "#7a7870"
CANVAS_BG  = "#0d0d0c"
GRID_LINE  = "#1a1a18"

# ── Load ALL reference D images ───────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

ref_images = []   # list of (filename, np.float32 28x28 flipped array)
for i in range(1, 51):
    p = os.path.join(script_dir, f"test_d_{i}.txt")
    if os.path.exists(p):
        raw = np.loadtxt(p, dtype=np.float32)
        arr = np.fliplr((raw.reshape(28, 28) / 255.0).astype(np.float32)).copy()
        ref_images.append((f"test_d_{i}.txt", arr))

if not ref_images:
    print("ERROR: no test_d_*.txt files found in", script_dir)
    exit(1)

print(f"Loaded {len(ref_images)} reference D image(s)")

# Start with a random one
current_ref_idx = [random.randint(0, len(ref_images) - 1)]
grid = ref_images[current_ref_idx[0]][1].copy()

# ── COM port helpers ──────────────────────────────────────────────────────────
def get_available_ports():
    if not SERIAL_AVAILABLE:
        return ["pyserial not installed"]
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return ports if ports else ["No ports found"]

def send_pixels(pixels_bytes: bytes, port: str) -> str:
    if not SERIAL_AVAILABLE:
        return "ERROR: pyserial not installed"
    if port in ("No ports found", "pyserial not installed"):
        return "ERROR: no port selected"
    try:
        with serial.Serial(port, BAUD, timeout=10) as ser:
            ser.reset_input_buffer()
            ser.write(pixels_bytes)
            ser.flush()
            result = ser.read(1)
            return "D DETECTED" if len(result) and result[0] == 1 else (
                   "ERROR: timeout" if not len(result) else "NOT D")
    except serial.SerialException as e:
        return f"ERROR: {e}"

def grid_to_bytes() -> bytes:
    return bytes(np.clip(np.fliplr(grid) * 255.0, 0, 255).astype(np.uint8).flatten())

# ─────────────────────────────────────────────────────────────────────────────
# ROOT — measure screen, derive all sizes from it
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
selected_port = tk.StringVar()
ref_label_var = tk.StringVar()

root.title("FPGA Character Classifier")
root.configure(bg=BG_MAIN)
root.attributes("-fullscreen", True)
root.update_idletasks()

SW = root.winfo_screenwidth()
SH = root.winfo_screenheight()

root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
root.bind("<F11>",    lambda e: root.attributes("-fullscreen",
                                not root.attributes("-fullscreen")))

# ── Derived layout constants ──────────────────────────────────────────────────
HDR_H     = int(SH * 0.062)
FTR_H     = int(SH * 0.038)
BODY_H    = SH - HDR_H - FTR_H - 6
H_PAD     = int(SW * 0.021)
V_PAD     = int(SH * 0.020)

CANVAS_PX = ((BODY_H - V_PAD*2 - 80) // GRID) * GRID
CELL      = CANVAS_PX // GRID
SIDEBAR_W = SW - CANVAS_PX - H_PAD*2 - 50

def fs(base): return max(8, int(base * SH / 1200))

# ── Fonts ─────────────────────────────────────────────────────────────────────
def load_font(families, size, weight="normal"):
    for fam in families:
        try:
            f = tkfont.Font(family=fam, size=size, weight=weight)
            if fam.lower() in f.actual("family").lower():
                return f
        except Exception:
            pass
    return tkfont.Font(family=families[-1], size=size, weight=weight)

font_hero    = load_font(["Poppins","Arial"],       fs(21), "bold")
font_section = load_font(["Poppins","Arial"],       fs(11), "bold")
font_body    = load_font(["Lora","Georgia"],        fs(11))
font_mono    = load_font(["Courier New","Courier"], fs(10))
font_result  = load_font(["Poppins","Arial"],       fs(32), "bold")
font_btn     = load_font(["Poppins","Arial"],       fs(12), "bold")
font_label   = load_font(["Lora","Georgia"],        fs(9))
font_hint    = load_font(["Lora","Georgia"],        fs(10))

# ── ttk combobox style ────────────────────────────────────────────────────────
style = ttk.Style()
style.theme_use("clam")
style.configure("Brand.TCombobox",
    fieldbackground=BG_MAIN, background=BG_CARD,
    foreground=TEXT_DARK, selectbackground=ORANGE,
    selectforeground=BG_MAIN, bordercolor=ORANGE,
    arrowcolor=ORANGE, padding=5,
)
style.map("Brand.TCombobox",
    fieldbackground=[("readonly", BG_MAIN)],
    foreground=[("readonly", TEXT_DARK)],
)

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
header = tk.Frame(root, bg=BG_PANEL, height=HDR_H)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(header, text="◈  FPGA Character Classifier",
    font=font_hero, fg=ORANGE, bg=BG_PANEL
).pack(side="left", padx=H_PAD)

hdr_r = tk.Frame(header, bg=BG_PANEL)
hdr_r.pack(side="right", padx=H_PAD)
tk.Label(hdr_r, text="28 × 28  ·  MNIST-style  ·  UART",
    font=font_label, fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="e")
tk.Label(hdr_r, text="ESC — exit fullscreen   F11 — toggle",
    font=font_label, fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="e")

tk.Frame(root, bg=ORANGE, height=3).pack(fill="x")

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
tk.Frame(root, bg=ORANGE, height=3).pack(fill="x", side="bottom")
footer = tk.Frame(root, bg=BG_PANEL, height=FTR_H)
footer.pack(fill="x", side="bottom")
footer.pack_propagate(False)
tk.Label(footer,
    text="copyright  ·  srivathsan  ·  FPGA EMNIST Classifier",
    font=font_label, fg=TEXT_MUTED, bg=BG_PANEL
).place(relx=0.5, rely=0.5, anchor="center")

# ═════════════════════════════════════════════════════════════════════════════
# BODY
# ═════════════════════════════════════════════════════════════════════════════
body = tk.Frame(root, bg=BG_MAIN)
body.pack(fill="both", expand=True, padx=H_PAD, pady=V_PAD)

# ── LEFT: draw canvas ─────────────────────────────────────────────────────────
left = tk.Frame(body, bg=BG_MAIN)
left.pack(side="left", anchor="n")

# Reference label above canvas
ref_label_var.set(f"Reference: {ref_images[current_ref_idx[0]][0]}  ({current_ref_idx[0]+1}/{len(ref_images)})")
tk.Label(left, textvariable=ref_label_var,
    font=font_label, fg=TEXT_MUTED, bg=BG_MAIN, anchor="w"
).pack(anchor="w", pady=(0, 4))

canvas_border = tk.Frame(left, bg=ORANGE, padx=3, pady=3)
canvas_border.pack()

cv = tk.Canvas(canvas_border,
    width=CANVAS_PX, height=CANVAS_PX,
    bg=CANVAS_BG, cursor="crosshair", highlightthickness=0)
cv.pack()

for i in range(1, GRID):
    cv.create_line(i*CELL, 0, i*CELL, CANVAS_PX, fill=GRID_LINE, width=1)
    cv.create_line(0, i*CELL, CANVAS_PX, i*CELL, fill=GRID_LINE, width=1)

rects = [[None]*GRID for _ in range(GRID)]
for r in range(GRID):
    for c in range(GRID):
        x0, y0 = c*CELL, r*CELL
        rects[r][c] = cv.create_rectangle(x0, y0, x0+CELL, y0+CELL,
                                           fill=CANVAS_BG, outline="")

hint_row = tk.Frame(left, bg=BG_MAIN)
hint_row.pack(fill="x", pady=(6, 0))
tk.Label(hint_row, text="● Left click — draw",
    font=font_hint, fg=TEXT_MUTED, bg=BG_MAIN).pack(side="left")
tk.Label(hint_row, text="○ Right click — erase",
    font=font_hint, fg=TEXT_MUTED, bg=BG_MAIN).pack(side="right")

# ── RIGHT: sidebar ────────────────────────────────────────────────────────────
CP   = int(SIDEBAR_W * 0.04)
CV2  = int(BODY_H    * 0.014)
DIV  = int(BODY_H    * 0.012)
BPAD = int(BODY_H    * 0.012)

right = tk.Frame(body, bg=BG_MAIN, width=SIDEBAR_W)
right.pack(side="left", fill="y", padx=(int(H_PAD*0.8), 0), anchor="n")
right.pack_propagate(False)

def divider():
    tk.Frame(right, bg=BG_CARD, height=1).pack(fill="x", pady=DIV)

def card(parent):
    return tk.Frame(parent, bg=BG_CARD, padx=CP, pady=CV2)

# ── RESULT ────────────────────────────────────────────────────────────────────
rc = card(right); rc.pack(fill="x")

tk.Label(rc, text="RESULT", font=font_label, fg=TEXT_MUTED, bg=BG_CARD,
    anchor="w").pack(anchor="w")

result_var = tk.StringVar(value="—")
result_lbl = tk.Label(rc, textvariable=result_var,
    font=font_result, fg=TEXT_MUTED, bg=BG_CARD, anchor="w")
result_lbl.pack(anchor="w", pady=(2, 0))

status_var = tk.StringVar(value="Select a port and click Send")
tk.Label(rc, textvariable=status_var,
    font=font_label, fg=TEXT_MUTED, bg=BG_CARD, anchor="w"
).pack(anchor="w", pady=(4, 0))

# ── CONNECTION ────────────────────────────────────────────────────────────────
divider()
cc = card(right); cc.pack(fill="x")

tk.Label(cc, text="CONNECTION", font=font_label, fg=TEXT_MUTED, bg=BG_CARD,
    anchor="w").pack(anchor="w", pady=(0, int(CV2*0.6)))

def info_row(lbl, val):
    f = tk.Frame(cc, bg=BG_CARD); f.pack(fill="x", pady=2)
    tk.Label(f, text=lbl, font=font_label, fg=TEXT_MUTED, bg=BG_CARD,
             width=10, anchor="w").pack(side="left")
    tk.Label(f, text=val, font=font_mono,  fg=ORANGE,    bg=BG_CARD,
             anchor="w").pack(side="left", padx=(8, 0))

info_row("Baud",     str(BAUD))
info_row("Grid",     f"{GRID} × {GRID}")
info_row("pyserial", "OK" if SERIAL_AVAILABLE else "NOT FOUND")
info_row("Refs",     f"{len(ref_images)} images loaded")

tk.Label(cc, text="COM Port", font=font_section, fg=TEXT_DARK, bg=BG_CARD,
    anchor="w").pack(anchor="w", pady=(int(CV2*0.8), 4))

sel_row = tk.Frame(cc, bg=BG_CARD); sel_row.pack(fill="x")

ports = get_available_ports()
selected_port.set(ports[0])

port_combo = ttk.Combobox(sel_row, textvariable=selected_port,
    values=ports, state="readonly",
    style="Brand.TCombobox", width=16,
    font=("Courier New", fs(10)))
port_combo.pack(side="left")

def refresh_ports():
    new = get_available_ports()
    port_combo["values"] = new
    if selected_port.get() not in new:
        selected_port.set(new[0])
    status_var.set(f"Scanned — {len(new)} port(s) found")

tk.Button(sel_row, text="⟳ Scan", command=refresh_ports,
    bg=BG_MAIN, fg=ORANGE, activebackground=BG_CARD,
    activeforeground=ORANGE_DIM, relief="flat", font=font_btn,
    padx=10, pady=4, cursor="hand2", bd=0
).pack(side="left", padx=(10, 0))

# ── HOW TO USE ────────────────────────────────────────────────────────────────
divider()
hc = card(right); hc.pack(fill="x")

tk.Label(hc, text="HOW TO USE", font=font_label, fg=TEXT_MUTED, bg=BG_CARD,
    anchor="w").pack(anchor="w", pady=(0, int(CV2*0.5)))

steps = [
    ("1", "Select COM port; click ⟳ Scan if not listed"),
    ("2", "A random reference D is loaded — click Send to verify"),
    ("3", "Should return D DETECTED"),
    ("4", "Click Random D to load another sample, or Clear to draw"),
]
for num, txt in steps:
    row = tk.Frame(hc, bg=BG_CARD); row.pack(fill="x", pady=2)
    tk.Label(row, text=num, font=font_btn, fg=ORANGE, bg=BG_CARD,
             width=2).pack(side="left")
    tk.Label(row, text=txt, font=font_body, fg=TEXT_DARK, bg=BG_CARD,
             anchor="w", wraplength=int(SIDEBAR_W*0.82), justify="left"
             ).pack(side="left", padx=(8, 0))

# ── BUTTONS ───────────────────────────────────────────────────────────────────
divider()

def styled_btn(text, cmd, primary=False):
    bg = ORANGE if primary else BG_CARD
    fg = BG_MAIN if primary else TEXT_DARK
    ab = ORANGE_DIM if primary else "#d4d1c7"
    tk.Button(right, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=ab, activeforeground=fg,
        relief="flat", font=font_btn,
        padx=16, pady=BPAD, cursor="hand2", bd=0
    ).pack(fill="x", pady=int(BPAD*0.4))

def on_send():
    port = selected_port.get()
    status_var.set(f"Sending via {port}…")
    result_var.set("…"); result_lbl.config(fg=TEXT_MUTED)
    root.update_idletasks()
    def worker():
        res = send_pixels(grid_to_bytes(), port)
        is_ok = res == "D DETECTED"
        root.after(0, lambda: result_var.set(res))
        root.after(0, lambda: result_lbl.config(fg=ORANGE if is_ok else "#e94560"))
        root.after(0, lambda: status_var.set("Classification complete"))
    threading.Thread(target=worker, daemon=True).start()

def on_random_d():
    """Pick a random reference D (different from current) and load it."""
    if len(ref_images) > 1:
        choices = [i for i in range(len(ref_images)) if i != current_ref_idx[0]]
        current_ref_idx[0] = random.choice(choices)
    else:
        current_ref_idx[0] = 0
    fname, arr = ref_images[current_ref_idx[0]]
    grid[:] = arr
    refresh_canvas()
    result_var.set("—"); result_lbl.config(fg=TEXT_MUTED)
    ref_label_var.set(f"Reference: {fname}  ({current_ref_idx[0]+1}/{len(ref_images)})")
    status_var.set(f"Loaded {fname}")

def on_clear():
    grid[:] = 0.0; refresh_canvas()
    result_var.set("—"); result_lbl.config(fg=TEXT_MUTED)
    status_var.set("Canvas cleared — draw your character")

styled_btn("▶   Send to FPGA",       on_send,     primary=True)
styled_btn("⚄   Random D",           on_random_d)
styled_btn("✕   Clear Canvas",       on_clear)

# ═════════════════════════════════════════════════════════════════════════════
# CANVAS DRAW LOGIC
# ═════════════════════════════════════════════════════════════════════════════
def refresh_canvas():
    for r in range(GRID):
        for c in range(GRID):
            v = grid[r, c]
            cv.itemconfig(rects[r][c],
                fill=f"#{int(217*v):02x}{int(119*v):02x}{int(87*v):02x}")

def paint(event, erase=False):
    gc = int(event.x / CELL); gr = int(event.y / CELL)
    for dr in range(-BRUSH, BRUSH+1):
        for dc in range(-BRUSH, BRUSH+1):
            nr, nc = gr+dr, gc+dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                s = max(0.0, 1.0 - (dr**2+dc**2)**0.5 / (BRUSH+0.5))
                grid[nr,nc] = max(0.0, grid[nr,nc]-s*0.6) if erase \
                              else min(1.0, grid[nr,nc]+s*0.6)
    refresh_canvas()

cv.bind("<B1-Motion>", lambda e: paint(e, False))
cv.bind("<Button-1>",  lambda e: paint(e, False))
cv.bind("<B3-Motion>", lambda e: paint(e, True))
cv.bind("<Button-3>",  lambda e: paint(e, True))

refresh_canvas()

print(f"Screen: {SW}×{SH}  |  Canvas: {CANVAS_PX}px ({CELL}px/cell)  |  Sidebar: {SIDEBAR_W}px")
root.mainloop()