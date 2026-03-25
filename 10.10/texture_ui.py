"""
Texture Synthesis & Inpainting - Tkinter GUI
=============================================
Drop-in UI for the algorithms from Szeliski Ch 10.5.
Requires: numpy, Pillow, scipy  (pip install numpy pillow scipy)

Usage:
    python texture_ui.py
    python texture_ui.py --source path/to/texture.png
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import threading
import time
import os
import sys
import argparse


# =============================================
#  CORE ALGORITHMS  (self-contained)
# =============================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float64) / 255.0

def save_image(arr, path):
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)
    return path

def gaussian_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels - 1):
        h, w = pyramid[-1].shape[:2]
        pil = Image.fromarray((np.clip(pyramid[-1], 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
        pyramid.append(np.array(pil, dtype=np.float64) / 255.0)
    return pyramid

def ssd_patch(p1, p2, mask=None):
    diff = p1 - p2
    if mask is not None:
        diff = diff * mask[..., np.newaxis] if diff.ndim == 3 else diff * mask
    return np.sum(diff ** 2)

def extract_patch(img, r, c, half, fill_value=0):
    h, w = img.shape[:2]
    size = 2 * half + 1
    if img.ndim == 3:
        patch = np.full((size, size, img.shape[2]), fill_value, dtype=img.dtype)
    else:
        patch = np.full((size, size), fill_value, dtype=img.dtype)
    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    pr0, pc0 = max(0, -r0), max(0, -c0)
    pr1 = size - max(0, r1 - h)
    pc1 = size - max(0, c1 - w)
    patch[pr0:pr1, pc0:pc1] = img[max(0, r0):min(h, r1), max(0, c0):min(w, c1)]
    return patch

def extract_mask_patch(mask, r, c, half):
    h, w = mask.shape
    size = 2 * half + 1
    patch = np.zeros((size, size), dtype=mask.dtype)
    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    pr0, pc0 = max(0, -r0), max(0, -c0)
    pr1 = size - max(0, r1 - h)
    pc1 = size - max(0, c1 - w)
    patch[pr0:pr1, pc0:pc1] = mask[max(0, r0):min(h, r1), max(0, c0):min(w, c1)]
    return patch

# ---------- 1. Efros & Leung ----------
def efros_leung_synthesis(source, output_size, neighborhood=5, progress_cb=None):
    half = neighborhood // 2
    sh, sw = source.shape[:2]
    oh, ow = output_size
    output = np.zeros((oh, ow, 3), dtype=np.float64)
    filled = np.zeros((oh, ow), dtype=bool)
    seed_size = min(3, sh, sw, oh, ow)
    sr, sc = sh // 2, sw // 2
    orr, oc = oh // 2, ow // 2
    s2 = seed_size // 2
    output[orr-s2:orr-s2+seed_size, oc-s2:oc-s2+seed_size] = \
        source[sr-s2:sr-s2+seed_size, sc-s2:sc-s2+seed_size]
    filled[orr-s2:orr-s2+seed_size, oc-s2:oc-s2+seed_size] = True
    total = oh * ow
    done = int(np.sum(filled))
    for _ in range(5):
        changed = False
        for r in range(oh):
            for c in range(ow):
                if filled[r, c]:
                    continue
                m = extract_mask_patch(filled.astype(np.float64), r, c, half)
                if np.sum(m) == 0:
                    continue
                out_patch = extract_patch(output, r, c, half)
                best_err = np.inf
                candidates = []
                for sr2 in range(half, sh - half):
                    for sc2 in range(half, sw - half):
                        src_patch = source[sr2-half:sr2+half+1, sc2-half:sc2+half+1]
                        err = ssd_patch(out_patch, src_patch, m)
                        if err < best_err:
                            best_err = err
                            candidates = [(sr2, sc2)]
                        elif err < best_err * 1.1:
                            candidates.append((sr2, sc2))
                if candidates:
                    idx = np.random.randint(len(candidates))
                    br, bc = candidates[idx]
                    output[r, c] = source[br, bc]
                    filled[r, c] = True
                    changed = True
                    done += 1
                    if progress_cb and done % 100 == 0:
                        progress_cb(done / total)
        if not changed:
            break
    for r in range(oh):
        for c in range(ow):
            if not filled[r, c]:
                output[r, c] = source[np.random.randint(sh), np.random.randint(sw)]
    return output

# ---------- 2. Wei & Levoy ----------
def wei_levoy_synthesis(source, output_size, neighborhood=5, levels=3, progress_cb=None):
    half = neighborhood // 2
    src_pyramid = gaussian_pyramid(source, levels)
    scale = 2 ** (levels - 1)
    coarse_h = max(2, output_size[0] // scale)
    coarse_w = max(2, output_size[1] // scale)
    coarse_src = src_pyramid[-1]
    csh, csw = coarse_src.shape[:2]
    result = np.zeros((coarse_h, coarse_w, 3))
    for r in range(coarse_h):
        for c in range(coarse_w):
            result[r, c] = coarse_src[r % csh, c % csw]
    for level in range(levels - 2, -1, -1):
        src_level = src_pyramid[level]
        slh, slw = src_level.shape[:2]
        new_h = result.shape[0] * 2
        new_w = result.shape[1] * 2
        if level == 0:
            new_h, new_w = output_size
        upsampled = np.array(
            Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
            .resize((new_w, new_h), Image.LANCZOS), dtype=np.float64) / 255.0
        result = upsampled.copy()
        for r in range(new_h):
            for c in range(new_w):
                out_patch = extract_patch(result, r, c, half)
                m = np.ones((2 * half + 1, 2 * half + 1))
                best_err = np.inf
                best_val = None
                n_samples = min(200, max(1, (slh - 2*half) * (slw - 2*half)))
                for _ in range(n_samples):
                    sr2 = np.random.randint(half, max(half+1, slh - half))
                    sc2 = np.random.randint(half, max(half+1, slw - half))
                    src_patch = extract_patch(src_level, sr2, sc2, half)
                    err = ssd_patch(out_patch, src_patch, m)
                    if err < best_err:
                        best_err = err
                        best_val = src_level[sr2, sc2].copy()
                if best_val is not None:
                    result[r, c] = best_val
            if progress_cb and r % 5 == 0:
                prog = ((levels - 1 - level) + r / new_h) / levels
                progress_cb(prog)
    return result

# ---------- 3. Criminisi inpainting ----------
def criminisi_inpainting(image, mask, patch_size=9, progress_cb=None):
    from scipy.ndimage import binary_dilation
    half = patch_size // 2
    result = image.copy()
    fill_mask = (1 - mask).astype(np.float64)
    h, w = image.shape[:2]
    confidence = fill_mask.copy()
    total_to_fill = int(np.sum(mask))
    filled_count = 0
    def get_boundary():
        dilated = binary_dilation(fill_mask > 0.5)
        boundary = dilated & (fill_mask < 0.5)
        return list(zip(*np.where(boundary)))
    def compute_normal(r, c):
        patch = extract_mask_patch(fill_mask, r, c, 1)
        gy, gx = np.gradient(patch)
        nx, ny = np.mean(gx), np.mean(gy)
        mag = np.sqrt(nx**2 + ny**2) + 1e-8
        return nx / mag, ny / mag
    def compute_data_term(r, c):
        if r < 1 or r >= h-1 or c < 1 or c >= w-1:
            return 0.001
        gray = np.mean(result, axis=2)
        gy = gray[min(r+1, h-1), c] - gray[max(r-1, 0), c]
        gx = gray[r, min(c+1, w-1)] - gray[r, max(c-1, 0)]
        iso_x, iso_y = -gy, gx
        nx, ny = compute_normal(r, c)
        return abs(iso_x * nx + iso_y * ny) + 0.001
    iteration = 0
    max_iterations = total_to_fill + 100
    while np.any(fill_mask < 0.5) and iteration < max_iterations:
        iteration += 1
        boundary = get_boundary()
        if not boundary:
            break
        best_priority = -1
        best_pixel = None
        for (r, c) in boundary:
            m = extract_mask_patch(confidence, r, c, half)
            conf = np.mean(m)
            data = compute_data_term(r, c)
            priority = conf * data
            if priority > best_priority:
                best_priority = priority
                best_pixel = (r, c)
        if best_pixel is None:
            break
        r, c = best_pixel
        out_patch = extract_patch(result, r, c, half)
        m = extract_mask_patch(fill_mask, r, c, half)
        best_err = np.inf
        best_r, best_c = 0, 0
        step = max(1, min(h, w) // 50)
        for sr2 in range(half, h - half, step):
            for sc2 in range(half, w - half, step):
                sm = extract_mask_patch(fill_mask, sr2, sc2, half)
                if np.mean(sm) < 0.99:
                    continue
                src_patch = extract_patch(result, sr2, sc2, half)
                err = ssd_patch(out_patch, src_patch, m)
                if err < best_err:
                    best_err = err
                    best_r, best_c = sr2, sc2
        for sr2 in range(max(half, best_r - 5), min(h - half, best_r + 6)):
            for sc2 in range(max(half, best_c - 5), min(w - half, best_c + 6)):
                sm = extract_mask_patch(fill_mask, sr2, sc2, half)
                if np.mean(sm) < 0.99:
                    continue
                src_patch = extract_patch(result, sr2, sc2, half)
                err = ssd_patch(out_patch, src_patch, m)
                if err < best_err:
                    best_err = err
                    best_r, best_c = sr2, sc2
        src_patch = extract_patch(result, best_r, best_c, half)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and fill_mask[rr, cc] < 0.5:
                    result[rr, cc] = src_patch[dr + half, dc + half]
                    fill_mask[rr, cc] = 1.0
                    confidence[rr, cc] = confidence[r, c]
                    filled_count += 1
        if progress_cb:
            progress_cb(min(1.0, filled_count / max(1, total_to_fill)))
    return result

# ---------- 4. Image Quilting ----------
def min_error_boundary_cut(overlap_err):
    h, w = overlap_err.shape
    if h == 0 or w == 0:
        return np.ones_like(overlap_err)
    dp = overlap_err.copy()
    for i in range(1, h):
        for j in range(w):
            candidates = [dp[i-1, j]]
            if j > 0:
                candidates.append(dp[i-1, j-1])
            if j < w - 1:
                candidates.append(dp[i-1, j+1])
            dp[i, j] += min(candidates)
    mask = np.zeros((h, w), dtype=np.float64)
    j = np.argmin(dp[-1])
    for i in range(h-1, -1, -1):
        mask[i, j:] = 1
        if i > 0:
            cands = [(dp[i-1, j], j)]
            if j > 0:
                cands.append((dp[i-1, j-1], j-1))
            if j < w - 1:
                cands.append((dp[i-1, j+1], j+1))
            _, j = min(cands, key=lambda x: x[0])
    return mask

def image_quilting(source, output_size, block_size=32, overlap=8, progress_cb=None):
    sh, sw = source.shape[:2]
    oh, ow = output_size
    output = np.zeros((oh, ow, 3), dtype=np.float64)
    bs = min(block_size, sh, sw)
    ov = min(overlap, bs // 4)
    step = bs - ov
    n_rows = (oh - ov) // step + 1
    n_cols = (ow - ov) // step + 1
    total = n_rows * n_cols
    count = 0
    for bi in range(n_rows):
        for bj in range(n_cols):
            y, x = bi * step, bj * step
            if bi == 0 and bj == 0:
                sr2 = np.random.randint(0, max(1, sh - bs))
                sc2 = np.random.randint(0, max(1, sw - bs))
                patch = source[sr2:sr2+bs, sc2:sc2+bs].copy()
                ph, pw = patch.shape[:2]
                output[y:y+ph, x:x+pw] = patch
                count += 1
                continue
            best_err = np.inf
            best_patch = None
            n_cand = min(100, max(1, (sh - bs) * (sw - bs)))
            for _ in range(n_cand):
                sr2 = np.random.randint(0, max(1, sh - bs))
                sc2 = np.random.randint(0, max(1, sw - bs))
                candidate = source[sr2:sr2+bs, sc2:sc2+bs]
                ch, cw = candidate.shape[:2]
                if ch < bs or cw < bs:
                    continue
                err = 0
                if bj > 0 and x + ov <= ow:
                    oy_end = min(bs, oh - y)
                    err += np.sum((output[y:y+oy_end, x:x+ov] - candidate[:oy_end, :ov]) ** 2)
                if bi > 0 and y + ov <= oh:
                    ox_end = min(bs, ow - x)
                    err += np.sum((output[y:y+ov, x:x+ox_end] - candidate[:ov, :ox_end]) ** 2)
                if err < best_err:
                    best_err = err
                    best_patch = candidate.copy()
            if best_patch is None:
                sr2 = np.random.randint(0, max(1, sh - bs))
                sc2 = np.random.randint(0, max(1, sw - bs))
                best_patch = source[sr2:sr2+bs, sc2:sc2+bs].copy()
            ph, pw = min(bs, oh - y), min(bs, ow - x)
            patch_region = best_patch[:ph, :pw]
            existing = output[y:y+ph, x:x+pw]
            bmask = np.ones((ph, pw), dtype=np.float64)
            if bj > 0 and ov > 0:
                ov_w = min(ov, pw)
                left_err = np.sum((existing[:, :ov_w] - patch_region[:, :ov_w]) ** 2, axis=2)
                bmask[:, :ov_w] = min_error_boundary_cut(left_err)
            if bi > 0 and ov > 0:
                ov_h = min(ov, ph)
                top_err = np.sum((existing[:ov_h, :] - patch_region[:ov_h, :]) ** 2, axis=2)
                top_mask = min_error_boundary_cut(top_err.T).T
                bmask[:ov_h, :] = np.minimum(bmask[:ov_h, :], top_mask)
            mask_3d = bmask[..., np.newaxis]
            output[y:y+ph, x:x+pw] = existing * (1 - mask_3d) + patch_region * mask_3d
            count += 1
            if progress_cb:
                progress_cb(count / total)
    return output[:oh, :ow]

# ---------- 5. Telea inpainting ----------
def telea_inpainting(image, mask, radius=5, progress_cb=None):
    from scipy.ndimage import distance_transform_edt
    result = image.copy()
    h, w = image.shape[:2]
    known = (mask < 0.5).astype(np.float64)
    dist = distance_transform_edt(mask > 0.5)
    fill_coords = list(zip(*np.where(mask > 0.5)))
    fill_coords.sort(key=lambda p: dist[p[0], p[1]])
    total = len(fill_coords)
    for idx, (r, c) in enumerate(fill_coords):
        r_start, r_end = max(0, r - radius), min(h, r + radius + 1)
        c_start, c_end = max(0, c - radius), min(w, c + radius + 1)
        total_w = 0
        color_sum = np.zeros(3)
        for nr in range(r_start, r_end):
            for nc in range(c_start, c_end):
                if known[nr, nc] < 0.5:
                    continue
                d = np.sqrt((nr - r)**2 + (nc - c)**2)
                if d > radius or d == 0:
                    continue
                w_d = 1.0 / (d * d + 1e-6)
                w_l = 1.0 / (1.0 + abs(dist[nr, nc] - dist[r, c]))
                weight = w_d * w_l
                color_sum += weight * result[nr, nc]
                total_w += weight
        if total_w > 0:
            result[r, c] = color_sum / total_w
        else:
            for rad in range(1, max(h, w)):
                found = False
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        nr2, nc2 = r + dr, c + dc
                        if 0 <= nr2 < h and 0 <= nc2 < w and known[nr2, nc2] > 0.5:
                            result[r, c] = result[nr2, nc2]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        known[r, c] = 1.0
        if progress_cb and idx % 50 == 0:
            progress_cb(idx / total)
    return result

# ---------- Texture generators ----------
def create_sample_texture(size=64):
    h, w = size, size
    img = np.zeros((h, w, 3))
    for r in range(h):
        for c in range(w):
            check = ((r // 8) + (c // 8)) % 2
            noise = np.random.rand(3) * 0.1
            if check:
                img[r, c] = np.array([0.8, 0.6, 0.3]) + noise
            else:
                img[r, c] = np.array([0.3, 0.5, 0.7]) + noise
    return np.clip(img, 0, 1)

def create_brick_texture(size=64):
    h, w = size, size
    img = np.zeros((h, w, 3))
    brick_h, brick_w = 8, 16
    mortar = np.array([0.7, 0.7, 0.65])
    for r in range(h):
        for c in range(w):
            row = r // brick_h
            offset = (brick_w // 2) * (row % 2)
            in_mortar = (r % brick_h < 1) or ((c + offset) % brick_w < 1)
            noise = np.random.rand(3) * 0.05
            if in_mortar:
                img[r, c] = mortar + noise
            else:
                brick_color = np.array([0.7, 0.3, 0.2]) + noise
                brick_id = row * 100 + ((c + offset) // brick_w)
                np.random.seed(brick_id)
                variation = np.random.rand(3) * 0.1 - 0.05
                np.random.seed(None)
                img[r, c] = brick_color + variation
    return np.clip(img, 0, 1)

def create_circle_mask(h, w, center=None, radius=None):
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 6
    mask = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if (r - center[0])**2 + (c - center[1])**2 < radius**2:
                mask[r, c] = 1.0
    return mask


# =============================================
#  THEME
# =============================================

COLORS = {
    "bg":          "#1a1a2e",
    "bg_light":    "#0f3460",
    "accent":      "#e94560",
    "accent_hover":"#ff6b6b",
    "text":        "#eaeaea",
    "text_dim":    "#8892a4",
    "text_bright": "#ffffff",
    "border":      "#2a2a4a",
    "success":     "#2ed573",
    "warning":     "#ffa502",
    "card_bg":     "#1e1e3a",
    "input_bg":    "#12122a",
    "progress_bg": "#2a2a4a",
    "progress_fg": "#e94560",
}

FONTS = {
    "title":   ("Segoe UI", 20, "bold"),
    "heading": ("Segoe UI", 13, "bold"),
    "body":    ("Segoe UI", 10),
    "small":   ("Segoe UI", 9),
    "mono":    ("Consolas", 9),
    "button":  ("Segoe UI", 10, "bold"),
}


# =============================================
#  MAIN APPLICATION
# =============================================

class TextureSynthUI:

    ALGORITHMS = {
        "Efros & Leung (1999)":     "efros_leung",
        "Wei & Levoy (2000)":       "wei_levoy",
        "Image Quilting (2001)":    "image_quilting",
        "Criminisi Inpaint (2004)": "criminisi",
        "Telea Inpaint (2004)":     "telea",
    }

    def __init__(self, root, source_path=None):
        self.root = root
        self.root.title("Texture Synthesis & Inpainting")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(960, 640)

        self.source_array = None
        self.result_array = None
        self.mask_array = None
        self.source_path = None
        self.running = False
        self.cancel_flag = False
        self.painting_mask = False

        self._configure_styles()
        self._build_ui()

        if source_path and os.path.isfile(source_path):
            self._load_source(source_path)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["card_bg"])
        style.configure("TLabel", background=COLORS["bg"],
                        foreground=COLORS["text"], font=FONTS["body"])
        style.configure("Title.TLabel", font=FONTS["title"],
                        foreground=COLORS["text_bright"], background=COLORS["bg"])
        style.configure("Section.TLabel", font=FONTS["heading"],
                        foreground=COLORS["text_bright"], background=COLORS["card_bg"])
        style.configure("Dim.TLabel", foreground=COLORS["text_dim"],
                        background=COLORS["bg"], font=FONTS["small"])
        style.configure("CardDim.TLabel", foreground=COLORS["text_dim"],
                        background=COLORS["card_bg"], font=FONTS["small"])
        style.configure("Card.TLabel", background=COLORS["card_bg"],
                        foreground=COLORS["text"], font=FONTS["body"])
        style.configure("Status.TLabel", foreground=COLORS["success"],
                        background=COLORS["bg"], font=FONTS["mono"])
        style.configure("Accent.TButton", font=FONTS["button"],
                        foreground=COLORS["text_bright"],
                        background=COLORS["accent"], borderwidth=0,
                        padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", COLORS["accent_hover"]),
                              ("disabled", COLORS["border"])])
        style.configure("Secondary.TButton", font=FONTS["body"],
                        foreground=COLORS["text"],
                        background=COLORS["bg_light"], borderwidth=0,
                        padding=(12, 6))
        style.map("Secondary.TButton",
                  background=[("active", COLORS["border"])])
        style.configure("TCombobox", fieldbackground=COLORS["input_bg"],
                        background=COLORS["bg_light"],
                        foreground=COLORS["text"], borderwidth=1)
        style.map("TCombobox",
                  fieldbackground=[("readonly", COLORS["input_bg"])],
                  foreground=[("readonly", COLORS["text"])])
        style.configure("Custom.Horizontal.TProgressbar",
                        background=COLORS["progress_fg"],
                        troughcolor=COLORS["progress_bg"],
                        borderwidth=0, thickness=8)
        style.configure("TScale", background=COLORS["bg"],
                        troughcolor=COLORS["progress_bg"])

    def _build_ui(self):
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(14, 4))
        ttk.Label(header, text="TEXTURE SYNTH",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Szeliski Ch 10.5  |  5 algorithms",
                  style="Dim.TLabel").pack(side="left", padx=(12, 0))

        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=14, pady=6)

        # LEFT: controls
        left = ttk.Frame(content, width=300)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        # -- Source card --
        src_card = tk.Frame(left, bg=COLORS["card_bg"],
                            highlightbackground=COLORS["border"],
                            highlightthickness=1)
        src_card.pack(fill="x", pady=(0, 8))
        src_inner = ttk.Frame(src_card, style="Card.TFrame")
        src_inner.pack(fill="x", padx=10, pady=8)
        ttk.Label(src_inner, text="SOURCE IMAGE",
                  style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        btn_row = ttk.Frame(src_inner, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_row, text="Load Image...",
                   style="Accent.TButton",
                   command=self._browse_source).pack(side="left")
        ttk.Button(btn_row, text="Checker", style="Secondary.TButton",
                   command=lambda: self._gen_texture("checker")).pack(
                       side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Brick", style="Secondary.TButton",
                   command=lambda: self._gen_texture("brick")).pack(
                       side="left", padx=(6, 0))
        self.src_info_label = ttk.Label(src_inner, text="No image loaded",
                                        style="CardDim.TLabel")
        self.src_info_label.pack(anchor="w")

        # -- Algorithm card --
        alg_card = tk.Frame(left, bg=COLORS["card_bg"],
                            highlightbackground=COLORS["border"],
                            highlightthickness=1)
        alg_card.pack(fill="x", pady=(0, 8))
        alg_inner = ttk.Frame(alg_card, style="Card.TFrame")
        alg_inner.pack(fill="x", padx=10, pady=8)
        ttk.Label(alg_inner, text="ALGORITHM",
                  style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.alg_var = tk.StringVar(value=list(self.ALGORITHMS.keys())[0])
        combo = ttk.Combobox(alg_inner, textvariable=self.alg_var,
                             values=list(self.ALGORITHMS.keys()),
                             state="readonly", width=28)
        combo.pack(fill="x", pady=(0, 6))
        combo.bind("<<ComboboxSelected>>", self._on_alg_change)
        self.alg_desc = ttk.Label(alg_inner, text="",
                                  style="CardDim.TLabel", wraplength=260)
        self.alg_desc.pack(anchor="w", pady=(0, 4))

        # -- Parameters card --
        param_card = tk.Frame(left, bg=COLORS["card_bg"],
                              highlightbackground=COLORS["border"],
                              highlightthickness=1)
        param_card.pack(fill="x", pady=(0, 8))
        self.param_frame = ttk.Frame(param_card, style="Card.TFrame")
        self.param_frame.pack(fill="x", padx=10, pady=8)
        self.param_title = ttk.Label(self.param_frame, text="PARAMETERS",
                                     style="Section.TLabel")
        self.param_title.pack(anchor="w", pady=(0, 4))
        self.param_widgets = {}
        self._build_params()

        # -- Run / progress --
        run_frame = ttk.Frame(left)
        run_frame.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(run_frame, text="  RUN",
                                  style="Accent.TButton", command=self._run)
        self.run_btn.pack(fill="x")
        self.cancel_btn = ttk.Button(run_frame, text="  Cancel",
                                     style="Secondary.TButton",
                                     command=self._cancel)
        self.progress = ttk.Progressbar(run_frame, mode="determinate",
                                        style="Custom.Horizontal.TProgressbar",
                                        maximum=100)
        self.progress.pack(fill="x", pady=(6, 0))
        self.status_label = ttk.Label(run_frame, text="Ready",
                                      style="Status.TLabel")
        self.status_label.pack(anchor="w", pady=(4, 0))
        self.save_btn = ttk.Button(run_frame, text="Save Result...",
                                   style="Secondary.TButton",
                                   command=self._save_result)
        self.save_btn.pack(fill="x", pady=(6, 0))

        # RIGHT: image panels
        right = ttk.Frame(content)
        right.pack(side="left", fill="both", expand=True)

        # Source preview
        src_preview = tk.Frame(right, bg=COLORS["card_bg"],
                               highlightbackground=COLORS["border"],
                               highlightthickness=1)
        src_preview.pack(fill="both", expand=True, pady=(0, 4))
        src_top = ttk.Frame(src_preview, style="Card.TFrame")
        src_top.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(src_top, text="  SOURCE",
                  style="Section.TLabel").pack(side="left")
        # Mask controls (will show/hide based on algorithm)
        self.mask_frame = ttk.Frame(src_top, style="Card.TFrame")
        self.mask_btn = ttk.Button(self.mask_frame, text="Paint Mask",
                                   style="Secondary.TButton",
                                   command=self._toggle_mask_painting)
        self.mask_btn.pack(side="left", padx=(8, 4))
        ttk.Label(self.mask_frame, text="Brush:",
                  style="Card.TLabel").pack(side="left", padx=(4, 2))
        self.brush_var = tk.IntVar(value=10)
        ttk.Scale(self.mask_frame, from_=3, to=30,
                  variable=self.brush_var, orient="horizontal",
                  length=80).pack(side="left")
        ttk.Button(self.mask_frame, text="Clear",
                   style="Secondary.TButton",
                   command=self._clear_mask).pack(side="left", padx=(4, 0))

        self.src_canvas = tk.Canvas(src_preview, bg=COLORS["input_bg"],
                                    highlightthickness=0)
        self.src_canvas.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # Result preview
        res_preview = tk.Frame(right, bg=COLORS["card_bg"],
                               highlightbackground=COLORS["border"],
                               highlightthickness=1)
        res_preview.pack(fill="both", expand=True, pady=(4, 0))
        ttk.Label(res_preview, text="  RESULT", style="Section.TLabel",
                  background=COLORS["card_bg"]).pack(anchor="nw", padx=8,
                                                      pady=(6, 0))
        self.res_canvas = tk.Canvas(res_preview, bg=COLORS["input_bg"],
                                    highlightthickness=0)
        self.res_canvas.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # Mouse events for mask painting
        self.src_canvas.bind("<B1-Motion>", self._paint_mask_src)
        self.src_canvas.bind("<Button-1>", self._paint_mask_src)

        self._on_alg_change()

    # -- Parameters --
    def _build_params(self):
        for key in list(self.param_widgets.keys()):
            self.param_widgets[key]["frame"].destroy()
        self.param_widgets.clear()
        alg = self.ALGORITHMS.get(self.alg_var.get(), "efros_leung")
        for p in self._get_param_defs(alg):
            self._add_slider(p["key"], p["label"], p["min"], p["max"],
                             p["default"], p.get("step", 1))

    def _get_param_defs(self, alg):
        sz = {"key": "size", "label": "Output Size",
              "min": 32, "max": 256, "default": 96, "step": 8}
        defs = {
            "efros_leung": [sz, {"key": "neighborhood", "label": "Neighborhood",
                                  "min": 3, "max": 11, "default": 5, "step": 2}],
            "wei_levoy": [sz,
                          {"key": "neighborhood", "label": "Neighborhood",
                           "min": 3, "max": 11, "default": 5, "step": 2},
                          {"key": "levels", "label": "Pyramid Levels",
                           "min": 2, "max": 5, "default": 3, "step": 1}],
            "image_quilting": [
                {"key": "size", "label": "Output Size",
                 "min": 64, "max": 512, "default": 128, "step": 16},
                {"key": "block_size", "label": "Block Size",
                 "min": 8, "max": 64, "default": 24, "step": 4},
                {"key": "overlap", "label": "Overlap",
                 "min": 2, "max": 24, "default": 6, "step": 2}],
            "criminisi": [
                {"key": "patch_size", "label": "Patch Size",
                 "min": 5, "max": 15, "default": 9, "step": 2},
                {"key": "mask_radius", "label": "Default Mask Radius",
                 "min": 5, "max": 40, "default": 12, "step": 1}],
            "telea": [
                {"key": "radius", "label": "Fill Radius",
                 "min": 2, "max": 15, "default": 5, "step": 1},
                {"key": "mask_radius", "label": "Default Mask Radius",
                 "min": 5, "max": 40, "default": 12, "step": 1}],
        }
        return defs.get(alg, [])

    def _add_slider(self, key, label, vmin, vmax, default, step=1):
        frame = ttk.Frame(self.param_frame, style="Card.TFrame")
        frame.pack(fill="x", pady=2)
        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
        val_label = ttk.Label(row, text=str(default), style="Card.TLabel",
                              foreground=COLORS["accent"])
        val_label.pack(side="right")
        var = tk.IntVar(value=default)
        def on_change(v, vl=val_label, s=step):
            vl.configure(text=str(int(float(v) // s * s)))
        ttk.Scale(frame, from_=vmin, to=vmax, variable=var,
                  orient="horizontal", command=on_change).pack(fill="x")
        self.param_widgets[key] = {"var": var, "frame": frame, "step": step}

    def _get_param(self, key):
        if key not in self.param_widgets:
            return None
        w = self.param_widgets[key]
        v = w["var"].get()
        s = w.get("step", 1)
        return int(v // s * s)

    # -- Algorithm change --
    def _on_alg_change(self, event=None):
        alg = self.ALGORITHMS.get(self.alg_var.get(), "efros_leung")
        descs = {
            "efros_leung":   "Pixel-by-pixel texture synthesis. Grows texture "
                             "by matching neighborhoods. Slow but faithful.",
            "wei_levoy":     "Multi-resolution coarse-to-fine synthesis using "
                             "Gaussian pyramids. Faster than pixel-based.",
            "image_quilting": "Tiles patches with minimum error boundary cuts "
                             "for seamless seams. Best quality/speed ratio.",
            "criminisi":     "Priority-based exemplar inpainting. Fills masked "
                             "holes using surrounding texture. Paint a mask!",
            "telea":         "Fast marching isophote inpainting. Propagates "
                             "level lines inward. Good for scratches. Paint a mask!",
        }
        self.alg_desc.configure(text=descs.get(alg, ""))
        self._build_params()
        if alg in ("criminisi", "telea"):
            self.mask_frame.pack(side="left", padx=(12, 0))
        else:
            self.mask_frame.pack_forget()

    # -- Source loading --
    def _browse_source(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All", "*.*")])
        if path:
            self._load_source(path)

    def _load_source(self, path):
        try:
            self.source_array = load_image(path)
            self.source_path = path
            h, w = self.source_array.shape[:2]
            self.src_info_label.configure(
                text="{} ({}x{})".format(os.path.basename(path), w, h))
            self._clear_mask()
            self._display_source()
        except Exception as e:
            messagebox.showerror("Error", "Failed to load image:\n{}".format(e))

    def _gen_texture(self, kind):
        self.status_label.configure(text="Generating texture...")
        self.root.update()
        if kind == "checker":
            self.source_array = create_sample_texture(64)
        else:
            self.source_array = create_brick_texture(64)
        self.source_path = None
        h, w = self.source_array.shape[:2]
        self.src_info_label.configure(text="{} ({}x{})".format(kind, w, h))
        self._clear_mask()
        self._display_source()
        self.status_label.configure(text="Ready")

    def _display_source(self):
        if self.source_array is not None:
            self._show_on_canvas(self.src_canvas, self.source_array,
                                 overlay_mask=self.mask_array)

    def _display_result(self):
        if self.result_array is not None:
            self._show_on_canvas(self.res_canvas, self.result_array)

    def _show_on_canvas(self, canvas, arr, overlay_mask=None):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)
        h, w = arr.shape[:2]
        scale = min(cw / w, ch / h, 4.0)
        nw, nh = int(w * scale), int(h * scale)
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        resample = Image.NEAREST if scale > 2 else Image.LANCZOS
        pil = pil.resize((nw, nh), resample)
        # Red overlay for mask
        if overlay_mask is not None and np.any(overlay_mask > 0.5):
            mask_resized = np.array(
                Image.fromarray((overlay_mask * 255).astype(np.uint8))
                .resize((nw, nh), resample), dtype=np.uint8)
            pil = pil.convert("RGBA")
            overlay = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            mask_bool = mask_resized > 128
            for yy in range(nh):
                for xx in range(nw):
                    if mask_bool[yy, xx]:
                        draw.point((xx, yy), fill=(229, 69, 96, 120))
            pil = Image.alpha_composite(pil, overlay).convert("RGB")
        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
        canvas._photo = tk_img
        canvas._scale = scale
        canvas._img_offset = (cw // 2 - nw // 2, ch // 2 - nh // 2)

    # -- Mask painting --
    def _toggle_mask_painting(self):
        self.painting_mask = not self.painting_mask
        if self.painting_mask:
            self.mask_btn.configure(text="Stop Painting")
            if self.source_array is not None and self.mask_array is None:
                h, w = self.source_array.shape[:2]
                self.mask_array = np.zeros((h, w), dtype=np.float64)
            self.src_canvas.configure(cursor="crosshair")
        else:
            self.mask_btn.configure(text="Paint Mask")
            self.src_canvas.configure(cursor="")

    def _paint_mask_src(self, event):
        if not self.painting_mask or self.source_array is None:
            return
        if self.mask_array is None:
            h, w = self.source_array.shape[:2]
            self.mask_array = np.zeros((h, w), dtype=np.float64)
        if not hasattr(self.src_canvas, "_scale"):
            return
        scale = self.src_canvas._scale
        ox, oy = self.src_canvas._img_offset
        ix = int((event.x - ox) / scale)
        iy = int((event.y - oy) / scale)
        h, w = self.mask_array.shape
        r = self.brush_var.get()
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = iy + dr, ix + dc
                if 0 <= rr < h and 0 <= cc < w and dr*dr + dc*dc <= r*r:
                    self.mask_array[rr, cc] = 1.0
        self._show_on_canvas(self.src_canvas, self.source_array,
                             overlay_mask=self.mask_array)

    def _clear_mask(self):
        self.mask_array = None
        if self.source_array is not None:
            self._display_source()

    # -- Run --
    def _run(self):
        if self.source_array is None:
            messagebox.showwarning("No Source", "Load a source image first.")
            return
        if self.running:
            return
        alg = self.ALGORITHMS.get(self.alg_var.get(), "efros_leung")
        self.running = True
        self.cancel_flag = False
        self.run_btn.pack_forget()
        self.cancel_btn.pack(fill="x")
        self.progress["value"] = 0
        self.status_label.configure(text="Running...",
                                    foreground=COLORS["warning"])
        def progress_cb(p):
            if self.cancel_flag:
                raise InterruptedError("Cancelled")
            self.root.after(0, lambda: self._update_progress(p))
        def task():
            try:
                t0 = time.time()
                result = self._execute_algorithm(alg, progress_cb)
                elapsed = time.time() - t0
                self.result_array = result
                self.root.after(0, lambda: self._on_complete(elapsed))
            except InterruptedError:
                self.root.after(0, self._on_cancelled)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, lambda: self._on_error(str(e)))
        threading.Thread(target=task, daemon=True).start()

    def _execute_algorithm(self, alg, progress_cb):
        src = self.source_array
        if alg == "efros_leung":
            sz = self._get_param("size") or 96
            nb = self._get_param("neighborhood") or 5
            return efros_leung_synthesis(src, (sz, sz), neighborhood=nb,
                                        progress_cb=progress_cb)
        elif alg == "wei_levoy":
            sz = self._get_param("size") or 64
            nb = self._get_param("neighborhood") or 5
            lv = self._get_param("levels") or 3
            return wei_levoy_synthesis(src, (sz, sz), neighborhood=nb,
                                      levels=lv, progress_cb=progress_cb)
        elif alg == "image_quilting":
            sz = self._get_param("size") or 128
            bs = self._get_param("block_size") or 24
            ov = self._get_param("overlap") or 6
            return image_quilting(src, (sz, sz), block_size=bs, overlap=ov,
                                 progress_cb=progress_cb)
        elif alg == "criminisi":
            ps = self._get_param("patch_size") or 9
            mask = self._get_or_create_mask()
            img = src.copy()
            h, w = img.shape[:2]
            if max(h, w) > 96:
                sc = 96.0 / max(h, w)
                nh2, nw2 = int(h * sc), int(w * sc)
                img = np.array(Image.fromarray((img * 255).astype(np.uint8))
                               .resize((nw2, nh2), Image.LANCZOS),
                               dtype=np.float64) / 255.0
                mask = np.array(Image.fromarray((mask * 255).astype(np.uint8))
                                .resize((nw2, nh2), Image.NEAREST),
                                dtype=np.float64) / 255.0
                mask = (mask > 0.5).astype(np.float64)
            img[mask > 0.5] = 0
            return criminisi_inpainting(img, mask, patch_size=ps,
                                        progress_cb=progress_cb)
        elif alg == "telea":
            rad = self._get_param("radius") or 5
            mask = self._get_or_create_mask()
            img = src.copy()
            h, w = img.shape[:2]
            if max(h, w) > 96:
                sc = 96.0 / max(h, w)
                nh2, nw2 = int(h * sc), int(w * sc)
                img = np.array(Image.fromarray((img * 255).astype(np.uint8))
                               .resize((nw2, nh2), Image.LANCZOS),
                               dtype=np.float64) / 255.0
                mask = np.array(Image.fromarray((mask * 255).astype(np.uint8))
                                .resize((nw2, nh2), Image.NEAREST),
                                dtype=np.float64) / 255.0
                mask = (mask > 0.5).astype(np.float64)
            img[mask > 0.5] = 0
            return telea_inpainting(img, mask, radius=rad,
                                    progress_cb=progress_cb)
        raise ValueError("Unknown algorithm: {}".format(alg))

    def _get_or_create_mask(self):
        if self.mask_array is not None and np.any(self.mask_array > 0.5):
            return self.mask_array.copy()
        h, w = self.source_array.shape[:2]
        mr = self._get_param("mask_radius") or min(h, w) // 6
        return create_circle_mask(h, w, radius=mr)

    def _update_progress(self, p):
        self.progress["value"] = min(100, p * 100)

    def _on_complete(self, elapsed):
        self.running = False
        self.cancel_btn.pack_forget()
        self.run_btn.pack(fill="x")
        self.progress["value"] = 100
        self.status_label.configure(text="Done in {:.1f}s".format(elapsed),
                                    foreground=COLORS["success"])
        self._display_result()

    def _on_cancelled(self):
        self.running = False
        self.cancel_btn.pack_forget()
        self.run_btn.pack(fill="x")
        self.progress["value"] = 0
        self.status_label.configure(text="Cancelled",
                                    foreground=COLORS["text_dim"])

    def _on_error(self, msg):
        self.running = False
        self.cancel_btn.pack_forget()
        self.run_btn.pack(fill="x")
        self.progress["value"] = 0
        self.status_label.configure(text="Error: {}".format(msg[:60]),
                                    foreground=COLORS["accent"])
        messagebox.showerror("Error", msg)

    def _cancel(self):
        self.cancel_flag = True

    def _save_result(self):
        if self.result_array is None:
            messagebox.showinfo("Nothing to save", "Run an algorithm first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if path:
            save_image(self.result_array, path)
            self.status_label.configure(
                text="Saved: {}".format(os.path.basename(path)),
                foreground=COLORS["success"])


# =============================================
#  ENTRY POINT
# =============================================

def main():
    parser = argparse.ArgumentParser(
        description="Texture Synthesis & Inpainting GUI")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to source texture image")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = TextureSynthUI(root, source_path=args.source)
    root.mainloop()

if __name__ == "__main__":
    main()