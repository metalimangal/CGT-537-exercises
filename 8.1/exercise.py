"""
FlipBook Aligner — GUI Application
====================================
A desktop app to align burst photos and export animated GIFs.

Requirements:
    pip install opencv-python pillow numpy

Run:
    python flipbook_app.py
"""

import os
import sys
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageTk
import math
import tempfile


# ══════════════════════════════════════════════════════════════
#  SAMPLE FRAME GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_sample_frames(dest_dir: Path, n: int = 12):
    """
    Synthesise a burst sequence: a figure jumping across a night-sky scene
    with slight per-frame camera shake — perfect for testing alignment.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    W, H = 640, 480

    # Pre-build star field (static)
    rng = np.random.default_rng(42)
    star_x = rng.integers(0, W, 100)
    star_y = rng.integers(0, H // 2, 100)
    star_b = rng.integers(120, 255, 100)

    def make_frame(i):
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(H):
            t = y / H
            img[y, :] = (int(10 + t * 22), int(12 + t * 18), int(38 + t * 55))

        # Stars
        for sx, sy, sb in zip(star_x, star_y, star_b):
            img[sy, sx] = (int(sb), int(sb), int(sb))

        # Ground
        cv2.rectangle(img, (0, H - 80), (W, H), (20, 52, 20), -1)
        cv2.rectangle(img, (0, H - 83), (W, H - 78), (28, 68, 28), -1)

        # Moon
        cv2.circle(img, (520, 78), 38, (210, 220, 235), -1)
        cv2.circle(img, (534, 70), 32, (10, 16, 42), -1)

        # ── Jumping figure ──────────────────────────────────
        t = i / (n - 1)
        ground_y = H - 80 - 60
        arc = 4 * t * (1 - t)
        body_y = int(ground_y - arc * 170)
        body_x = int(80 + t * 400)
        lean = math.sin(t * math.pi) * 18

        fig_w, fig_h = 80, 120
        fig = np.zeros((fig_h, fig_w, 4), dtype=np.uint8)
        cx = fig_w // 2

        # Torso
        cv2.rectangle(fig, (cx - 10, 30), (cx + 10, 75), (60, 100, 185, 255), -1)
        # Head
        cv2.circle(fig, (cx, 19), 16, (220, 175, 135, 255), -1)
        # Legs (spread with arc)
        sp = int(arc * 30)
        cv2.line(fig, (cx, 75), (cx - sp, 110), (100, 65, 145, 255), 5)
        cv2.line(fig, (cx, 75), (cx + sp, 110), (100, 65, 145, 255), 5)
        cv2.ellipse(fig, (cx - sp, 113), (11, 5), 0, 0, 360, (70, 45, 110, 255), -1)
        cv2.ellipse(fig, (cx + sp, 113), (11, 5), 0, 0, 360, (70, 45, 110, 255), -1)
        # Arms raised
        ar = int(arc * 38)
        cv2.line(fig, (cx - 10, 38), (cx - 30, 38 - ar), (60, 100, 185, 255), 4)
        cv2.line(fig, (cx + 10, 38), (cx + 30, 38 - ar), (60, 100, 185, 255), 4)

        # Rotate figure
        Mrot = cv2.getRotationMatrix2D((cx, fig_h // 2), lean, 1.0)
        fig_rgb = cv2.warpAffine(fig[:, :, :3], Mrot, (fig_w, fig_h))
        fig_a   = cv2.warpAffine(fig[:, :, 3],  Mrot, (fig_w, fig_h))

        # Paste
        x1 = max(0, body_x - cx);  y1 = max(0, body_y - fig_h // 2)
        x2 = min(W, x1 + fig_w);   y2 = min(H, y1 + fig_h)
        fw, fh = x2 - x1, y2 - y1
        if fw > 0 and fh > 0:
            alpha = fig_a[:fh, :fw].astype(float) / 255.0
            for c in range(3):
                img[y1:y2, x1:x2, c] = (
                    img[y1:y2, x1:x2, c] * (1 - alpha) +
                    fig_rgb[:fh, :fw, c] * alpha
                ).astype(np.uint8)

        # Shadow
        sv = int(30 + arc * 12)
        sa = max(0.1, 0.55 - arc * 0.45)
        ov = img.copy()
        cv2.ellipse(ov, (body_x, ground_y + 62), (sv, 8), 0, 0, 360, (5, 22, 5), -1)
        cv2.addWeighted(ov, sa, img, 1 - sa, 0, img)

        # Camera shake
        rng2 = np.random.default_rng(i * 7 + 13)
        dx, dy = int(rng2.integers(-5, 6)), int(rng2.integers(-4, 5))
        Msh = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, Msh, (W, H))
        return img

    paths = []
    for i in range(n):
        frame = make_frame(i)
        p = dest_dir / f"sample_{i:04d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
    return paths


# ══════════════════════════════════════════════════════════════
#  CORE ALIGNMENT ENGINE  (same logic as flipbook_align.py)
# ══════════════════════════════════════════════════════════════

EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


def load_images_from_dir(directory):
    paths = []
    for ext in EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*.{ext.upper()}")))
    seen, unique = set(), []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p); unique.append(p)
    return unique


def extract_features(gray, method="sift"):
    det = cv2.SIFT_create() if method == "sift" else cv2.ORB_create(nfeatures=2000)
    return det.detectAndCompute(gray, None)


def match_features(d1, d2, method="sift", ratio=0.75):
    if method == "sift":
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d1, d2, k=2)
    return [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]


def compute_transform(kp1, kp2, matches, use_ransac=True):
    if len(matches) < 3:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    method = cv2.RANSAC if use_ransac else cv2.LMEDS
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=method,
                                        ransacReprojThreshold=3.0)
    return M


def invert_affine(M):
    M3 = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M3)[:2, :]


def warp_image(img, M, ref_shape, interp="bilinear"):
    h, w = ref_shape[:2]
    flags = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_CUBIC
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_common(frames):
    mask = np.ones(frames[0].shape[:2], dtype=bool)
    for f in frames:
        mask &= cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) > 0
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return frames
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return [f[r0:r1+1, c0:c1+1] for f in frames]


def cross_dissolve(a, b, steps):
    return [cv2.addWeighted(a, 1 - i/(steps+1), b, i/(steps+1), 0)
            for i in range(1, steps+1)]


def align_pipeline(paths, method, ratio, use_ransac, interp, crop,
                   log_fn, progress_fn):
    images = [cv2.imread(p) for p in paths]
    ref_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    log_fn(f"Extracting reference features ({method.upper()})…")
    kp_ref, desc_ref = extract_features(ref_gray, method)
    log_fn(f"  {len(kp_ref)} keypoints found in reference image.")

    aligned = [images[0]]
    for i, img in enumerate(images[1:], 1):
        log_fn(f"\n[{i}/{len(images)-1}] {Path(paths[i]).name}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = extract_features(gray, method)
        log_fn(f"  Detected {len(kp)} keypoints")
        matches = match_features(desc_ref, desc, method, ratio)
        log_fn(f"  {len(matches)} matches after ratio test")
        M = compute_transform(kp_ref, kp, matches, use_ransac)
        if M is None:
            log_fn("  ⚠ No transform found — using original image.")
            aligned.append(img)
        else:
            M_inv = invert_affine(M)
            warped = warp_image(img, M_inv, images[0].shape, interp)
            aligned.append(warped)
            log_fn("  ✓ Aligned.")
        progress_fn(int(i / (len(images)-1) * 80))

    if crop:
        log_fn("\nCropping to common valid area…")
        aligned = crop_common(aligned)

    return aligned


def save_gif(frames, path, delay, dissolve_steps, slomo, log_fn, progress_fn):
    if slomo and dissolve_steps > 0:
        all_frames = []
        for i in range(len(frames)-1):
            all_frames.append(frames[i])
            all_frames.extend(cross_dissolve(frames[i], frames[i+1], dissolve_steps))
        all_frames.append(frames[-1])
        frames = all_frames

    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=delay, loop=0)
    progress_fn(100)
    log_fn(f"\n✅ Saved → {path}")


# ══════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════

BG       = "#0d0f14"
PANEL    = "#151820"
CARD     = "#1c2030"
ACCENT   = "#00e5ff"
ACCENT2  = "#ff3d71"
TEXT     = "#e8eaf0"
MUTED    = "#5a6080"
SUCCESS  = "#00c896"
FONT_H   = ("Courier New", 13, "bold")
FONT_M   = ("Courier New", 11)
FONT_S   = ("Courier New", 9)


class FlipBookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FlipBook Aligner")
        self.configure(bg=BG)
        self.geometry("1080x760")
        self.minsize(900, 640)
        self.resizable(True, True)

        self._folder = tk.StringVar(value="")
        self._output = tk.StringVar(value="")
        self._method = tk.StringVar(value="sift")
        self._ratio  = tk.DoubleVar(value=0.75)
        self._ransac = tk.BooleanVar(value=True)
        self._interp = tk.StringVar(value="bilinear")
        self._delay  = tk.IntVar(value=100)
        self._slomo  = tk.BooleanVar(value=False)
        self._dissolve = tk.IntVar(value=5)
        self._crop   = tk.BooleanVar(value=True)

        self._image_paths = []
        self._aligned_frames = []
        self._preview_index = 0
        self._preview_job = None

        self._build_ui()

    # ── Layout ─────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        hdr = tk.Frame(self, bg=BG, pady=16)
        hdr.pack(fill="x", padx=30)
        tk.Label(hdr, text="◈ FLIPBOOK", font=("Courier New", 22, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text=" ALIGNER", font=("Courier New", 22),
                 bg=BG, fg=TEXT).pack(side="left")
        tk.Label(hdr, text="feature-based image alignment",
                 font=FONT_S, bg=BG, fg=MUTED).pack(side="left", padx=16, pady=4)

        # Main columns
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        self._build_controls(left)
        self._build_preview(right)
        self._build_log(right)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=PANEL, bd=0)
        outer.pack(fill="x", pady=(0, 10))
        tk.Label(outer, text=title, font=("Courier New", 9, "bold"),
                 bg=PANEL, fg=ACCENT, pady=6, padx=12,
                 anchor="w").pack(fill="x")
        sep = tk.Frame(outer, bg=ACCENT, height=1)
        sep.pack(fill="x")
        inner = tk.Frame(outer, bg=CARD, padx=12, pady=10)
        inner.pack(fill="x")
        return inner

    def _row(self, parent, label, widget_fn, **kwargs):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FONT_S, bg=CARD, fg=MUTED,
                 width=16, anchor="w").pack(side="left")
        widget_fn(row, **kwargs).pack(side="left", fill="x", expand=True)
        return row

    def _build_controls(self, parent):
        # Input
        c = self._card(parent, "▸ INPUT")
        self._folder_lbl = tk.Label(c, text="No folder selected",
                                     font=FONT_S, bg=CARD, fg=MUTED,
                                     anchor="w", wraplength=250)
        self._folder_lbl.pack(fill="x", pady=(0, 6))
        self._thumb_bar = tk.Frame(c, bg=CARD, height=48)
        self._thumb_bar.pack(fill="x", pady=(0, 6))
        btn_row = tk.Frame(c, bg=CARD)
        btn_row.pack(fill="x")
        btn_folder = tk.Button(btn_row, text="⊕  Select Folder",
                               font=FONT_M, bg=ACCENT, fg=BG,
                               relief="flat", cursor="hand2", pady=6,
                               command=self._pick_folder, activebackground="#00b8cc")
        btn_folder.pack(side="left", fill="x", expand=True, padx=(0, 4))
        btn_sample = tk.Button(btn_row, text="⚡ Demo",
                               font=FONT_M, bg=PANEL, fg=ACCENT,
                               relief="flat", cursor="hand2", pady=6,
                               command=self._load_samples)
        btn_sample.pack(side="left")

        # Settings
        c2 = self._card(parent, "▸ SETTINGS")

        def combo(p, var, vals):
            return ttk.Combobox(p, textvariable=var, values=vals,
                                state="readonly", font=FONT_S, width=14)

        self._row(c2, "Feature method", combo,
                  var=self._method, vals=["sift", "orb"])
        self._row(c2, "Interpolation", combo,
                  var=self._interp, vals=["bilinear", "bicubic"])

        def slider(p, var, from_, to, res=0.01):
            f = tk.Frame(p, bg=CARD)
            s = tk.Scale(f, variable=var, from_=from_, to=to, resolution=res,
                         orient="horizontal", bg=CARD, fg=TEXT,
                         highlightthickness=0, troughcolor=PANEL,
                         activebackground=ACCENT, sliderrelief="flat",
                         length=140, font=FONT_S)
            s.pack(side="left")
            return f

        self._row(c2, "Ratio threshold", slider, var=self._ratio,
                  from_=0.5, to=0.99, res=0.01)
        self._row(c2, "GIF delay (ms)", slider, var=self._delay,
                  from_=30, to=500, res=10)

        def checks(p):
            f = tk.Frame(p, bg=CARD)
            tk.Checkbutton(f, text="RANSAC", variable=self._ransac,
                           bg=CARD, fg=TEXT, selectcolor=BG,
                           activebackground=CARD, font=FONT_S).pack(side="left")
            tk.Checkbutton(f, text="Crop", variable=self._crop,
                           bg=CARD, fg=TEXT, selectcolor=BG,
                           activebackground=CARD, font=FONT_S).pack(side="left", padx=8)
            tk.Checkbutton(f, text="Slo-mo", variable=self._slomo,
                           bg=CARD, fg=TEXT, selectcolor=BG,
                           activebackground=CARD, font=FONT_S).pack(side="left")
            return f

        self._row(c2, "Options", lambda p, **_: checks(p))
        self._row(c2, "Dissolve steps", slider, var=self._dissolve,
                  from_=1, to=15, res=1)

        # Output
        c3 = self._card(parent, "▸ OUTPUT")
        out_row = tk.Frame(c3, bg=CARD)
        out_row.pack(fill="x", pady=(0, 6))
        self._out_lbl = tk.Label(out_row, text="output.gif", font=FONT_S,
                                  bg=CARD, fg=MUTED, anchor="w")
        self._out_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(out_row, text="…", font=FONT_S, bg=PANEL, fg=TEXT,
                  relief="flat", cursor="hand2", padx=6,
                  command=self._pick_output).pack(side="right")

        self._run_btn = tk.Button(c3, text="▶  RUN ALIGNMENT",
                                   font=("Courier New", 12, "bold"),
                                   bg=ACCENT2, fg="white", relief="flat",
                                   cursor="hand2", pady=8,
                                   command=self._run, state="disabled",
                                   activebackground="#cc2255")
        self._run_btn.pack(fill="x")

        # Progress
        c4 = self._card(parent, "▸ PROGRESS")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Cyan.Horizontal.TProgressbar",
                        troughcolor=PANEL, background=ACCENT,
                        thickness=6, borderwidth=0)
        self._progress = ttk.Progressbar(c4, style="Cyan.Horizontal.TProgressbar",
                                          length=280, mode="determinate")
        self._progress.pack(fill="x", pady=4)
        self._status = tk.Label(c4, text="Idle", font=FONT_S,
                                 bg=CARD, fg=MUTED, anchor="w")
        self._status.pack(fill="x")

    def _build_preview(self, parent):
        pf = tk.Frame(parent, bg=CARD, bd=0)
        pf.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        pf.rowconfigure(1, weight=1)
        pf.columnconfigure(0, weight=1)

        hdr = tk.Frame(pf, bg=PANEL, pady=6, padx=10)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="▸ PREVIEW", font=("Courier New", 9, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left")
        self._preview_counter = tk.Label(hdr, text="", font=FONT_S,
                                          bg=PANEL, fg=MUTED)
        self._preview_counter.pack(side="right")

        # Controls
        ctrl = tk.Frame(pf, bg=CARD, pady=4)
        ctrl.grid(row=2, column=0, sticky="ew")
        tk.Button(ctrl, text="◀", font=FONT_M, bg=PANEL, fg=TEXT,
                  relief="flat", cursor="hand2", padx=8,
                  command=self._prev_frame).pack(side="left", padx=4)
        tk.Button(ctrl, text="▶", font=FONT_M, bg=PANEL, fg=TEXT,
                  relief="flat", cursor="hand2", padx=8,
                  command=self._next_frame).pack(side="left")
        self._play_btn = tk.Button(ctrl, text="⏵  PLAY", font=FONT_S,
                                    bg=ACCENT, fg=BG, relief="flat",
                                    cursor="hand2", padx=8,
                                    command=self._toggle_play)
        self._play_btn.pack(side="left", padx=8)

        self._canvas = tk.Canvas(pf, bg="#080a10", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=1, pady=1)
        self._canvas.bind("<Configure>", lambda e: self._show_frame())

        ph_frame = tk.Frame(self._canvas, bg="#080a10")
        tk.Label(ph_frame,
                 text="no frames loaded",
                 font=("Courier New", 11), bg="#080a10", fg=MUTED).pack(pady=(0,6))
        tk.Label(ph_frame,
                 text="Select a folder  —  or hit  ⚡ Demo  to try the built-in sequence",
                 font=FONT_S, bg="#080a10", fg="#3a4060").pack()
        self._placeholder = ph_frame
        self._canvas.create_window(
            self._canvas.winfo_reqwidth()//2 or 300,
            self._canvas.winfo_reqheight()//2 or 150,
            window=self._placeholder, tags="ph")

    def _build_log(self, parent):
        lf = tk.Frame(parent, bg=CARD)
        lf.grid(row=1, column=0, sticky="nsew")
        lf.rowconfigure(1, weight=1)
        lf.columnconfigure(0, weight=1)

        hdr = tk.Frame(lf, bg=PANEL, pady=6, padx=10)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="▸ LOG", font=("Courier New", 9, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left")
        tk.Button(hdr, text="clear", font=FONT_S, bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2",
                  command=lambda: self._log_box.delete("1.0", "end")).pack(side="right")

        self._log_box = tk.Text(lf, font=("Courier New", 9), bg="#080a10",
                                 fg=TEXT, insertbackground=ACCENT,
                                 relief="flat", state="disabled",
                                 wrap="word", padx=8, pady=6)
        self._log_box.grid(row=1, column=0, sticky="nsew")

        sb = tk.Scrollbar(lf, command=self._log_box.yview, bg=PANEL,
                           troughcolor=BG, relief="flat", width=10)
        sb.grid(row=1, column=1, sticky="ns")
        self._log_box.configure(yscrollcommand=sb.set)

        # Tag colours
        self._log_box.tag_configure("ok",   foreground=SUCCESS)
        self._log_box.tag_configure("warn", foreground="#ffaa00")
        self._log_box.tag_configure("info", foreground=ACCENT)

    # ── Actions ────────────────────────────────────────────────

    def _load_samples(self):
        """Generate and load the built-in demo burst sequence."""
        self._status.config(text="Generating sample frames...", fg=MUTED)
        self.update_idletasks()
        tmp = Path(tempfile.mkdtemp(prefix="flipbook_demo_"))
        self._log("Generating demo frames (jumping figure, night sky)...\n", "info")
        paths = generate_sample_frames(tmp, n=12)
        self._image_paths = paths
        self._sample_tmp_dir = tmp          # keep ref so we can show it
        self._folder_lbl.config(
            text=f"⚡  Demo sequence  ({len(paths)} frames)", fg=ACCENT)
        out_gif = tmp / "demo_flipbook.gif"
        self._output.set(str(out_gif))
        self._out_lbl.config(text=str(out_gif), fg=TEXT)
        self._run_btn.config(state="normal")
        self._log("Demo frames ready in: " + str(tmp) + "\n", "ok")
        self._build_thumbs(paths)
        # Show the mid-jump frame immediately in preview
        mid_bgr = cv2.imread(paths[len(paths)//2])
        if mid_bgr is not None:
            self._aligned_frames = [cv2.imread(p) for p in paths]
            self._preview_index = len(paths)//2
            self.after(100, self._show_frame)

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        paths = load_images_from_dir(folder)
        if not paths:
            messagebox.showwarning("No images", "No supported images found in that folder.")
            return
        self._image_paths = paths
        name = Path(folder).name
        self._folder_lbl.config(text=f"📁  {name}  ({len(paths)} images)", fg=TEXT)
        self._output.set(str(Path(folder) / "flipbook.gif"))
        self._out_lbl.config(text=self._output.get(), fg=TEXT)
        self._run_btn.config(state="normal")
        self._log(f"Loaded {len(paths)} images from: {folder}\n", "info")
        self._build_thumbs(paths)

    def _pick_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF animation", "*.gif")],
            title="Save GIF as"
        )
        if p:
            self._output.set(p)
            self._out_lbl.config(text=p, fg=TEXT)

    def _build_thumbs(self, paths):
        for w in self._thumb_bar.winfo_children():
            w.destroy()
        max_thumbs = 8
        for p in paths[:max_thumbs]:
            try:
                img = Image.open(p)
                img.thumbnail((42, 42))
                tk_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(self._thumb_bar, image=tk_img, bg=CARD,
                               relief="flat", bd=1, cursor="hand2")
                lbl.image = tk_img
                lbl.pack(side="left", padx=2)
            except Exception:
                pass
        if len(paths) > max_thumbs:
            tk.Label(self._thumb_bar, text=f"+{len(paths)-max_thumbs}",
                     font=FONT_S, bg=CARD, fg=MUTED).pack(side="left", padx=4)

    def _run(self):
        if not self._image_paths:
            return
        out = self._output.get() or "flipbook.gif"
        self._run_btn.config(state="disabled", text="⏳  Running…")
        self._progress["value"] = 0
        self._aligned_frames = []

        def task():
            try:
                frames = align_pipeline(
                    self._image_paths,
                    method=self._method.get(),
                    ratio=self._ratio.get(),
                    use_ransac=self._ransac.get(),
                    interp=self._interp.get(),
                    crop=self._crop.get(),
                    log_fn=self._log,
                    progress_fn=self._set_progress
                )
                self._aligned_frames = frames
                self._preview_index = 0
                self.after(0, self._show_frame)

                save_gif(frames, out,
                         delay=self._delay.get(),
                         dissolve_steps=self._dissolve.get(),
                         slomo=self._slomo.get(),
                         log_fn=self._log,
                         progress_fn=self._set_progress)

                self._log(f"\nDone! {len(frames)} frames → {out}\n", "ok")
                self.after(0, lambda: self._status.config(
                    text=f"✓ Done — {len(frames)} frames", fg=SUCCESS))
            except Exception as e:
                self._log(f"\nError: {e}\n", "warn")
                self.after(0, lambda: self._status.config(text=f"Error: {e}", fg=ACCENT2))
            finally:
                self.after(0, lambda: self._run_btn.config(
                    state="normal", text="▶  RUN ALIGNMENT"))

        threading.Thread(target=task, daemon=True).start()

    # ── Preview ────────────────────────────────────────────────

    def _show_frame(self, _event=None):
        if not self._aligned_frames:
            return
        self._canvas.delete('ph')  # hide placeholder
        idx = self._preview_index % len(self._aligned_frames)
        frame_bgr = self._aligned_frames[idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        cw = self._canvas.winfo_width() or 500
        ch = self._canvas.winfo_height() or 350
        pil.thumbnail((cw, ch), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, anchor="center", image=tk_img)
        self._canvas._img = tk_img  # keep reference

        self._preview_counter.config(
            text=f"{idx+1} / {len(self._aligned_frames)}")

    def _prev_frame(self):
        self._stop_play()
        if self._aligned_frames:
            self._preview_index = (self._preview_index - 1) % len(self._aligned_frames)
            self._show_frame()

    def _next_frame(self):
        self._stop_play()
        if self._aligned_frames:
            self._preview_index = (self._preview_index + 1) % len(self._aligned_frames)
            self._show_frame()

    def _toggle_play(self):
        if self._preview_job:
            self._stop_play()
        else:
            self._play_btn.config(text="⏸  PAUSE")
            self._auto_advance()

    def _auto_advance(self):
        if not self._aligned_frames:
            return
        self._preview_index = (self._preview_index + 1) % len(self._aligned_frames)
        self._show_frame()
        delay = max(30, self._delay.get())
        self._preview_job = self.after(delay, self._auto_advance)

    def _stop_play(self):
        if self._preview_job:
            self.after_cancel(self._preview_job)
            self._preview_job = None
        self._play_btn.config(text="⏵  PLAY")

    # ── Helpers ────────────────────────────────────────────────

    def _log(self, msg, tag=None):
        def _do():
            self._log_box.config(state="normal")
            self._log_box.insert("end", msg + ("\n" if not msg.endswith("\n") else ""),
                                  tag or "")
            self._log_box.see("end")
            self._log_box.config(state="disabled")
            self._status.config(text=msg.strip()[:80], fg=MUTED)
        self.after(0, _do)

    def _set_progress(self, val):
        self.after(0, lambda: self._progress.configure(value=val))


if __name__ == "__main__":
    app = FlipBookApp()
    app.mainloop()