"""
Segmentation Explorer — Ex 6.5
================================
Interactive GUI to explore Semantic, Instance, and Panoptic segmentation.
Supports custom image upload + synthetic demo mode.

Dependencies (all standard or one-line install):
    pip install numpy matplotlib scipy Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# ── Colours ────────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1c1f2e"
ACCENT   = "#5e81f4"
ACCENT2  = "#e06c75"
TEXT     = "#e8eaf6"
SUBTEXT  = "#8892b0"
BTN_ACTIVE = "#2a2d3e"

SEM_COLORS  = ["#2d2d2d", "#4e79a7", "#e15759", "#76b7b2", "#59a14f",
               "#edc948", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac"]
INST_COLORS = ["#2d2d2d", "#76b7b2", "#59a14f", "#edc948", "#e15759",
               "#4e79a7", "#af7aa1", "#ff9da7", "#f28e2b", "#b07aa1"]

# ── Synthetic scene builder ─────────────────────────────────────────────────
def make_synthetic_scene():
    H, W = 120, 240
    cls  = np.zeros((H, W), dtype=int)
    inst = np.zeros((H, W), dtype=int)

    # Sky (class 3) — top band
    cls[:30, :]  = 3
    # Ground (class 4) — bottom band
    cls[90:, :]  = 4

    # Person 1
    cls [25:90, 15:50]  = 1;  inst[25:90, 15:50]  = 1
    # Person 2
    cls [30:85, 80:115] = 1;  inst[30:85, 80:115] = 2
    # Car
    cls [60:95, 145:215] = 2; inst[60:95, 145:215] = 3
    # Bicycle (class 5)
    cls [70:95, 55:75]  = 5;  inst[70:95, 55:75]  = 4

    return cls, inst

# ── Segmentation operations ──────────────────────────────────────────────────
def get_semantic(cls, inst):
    return cls

def get_instance(cls, inst):
    return inst

def get_panoptic_display(cls, inst):
    """Encode panoptic as class*1000+inst for display (unique colour per segment)."""
    out = np.zeros_like(cls)
    uid = 1
    for c in np.unique(cls):
        if c == 0:
            continue
        for i in np.unique(inst[cls == c]):
            out[(cls == c) & (inst == i)] = uid
            uid += 1
    return out

def semantic_to_instance_approx(cls):
    """Connected-components per class — approximate."""
    from scipy.ndimage import label as nd_label
    out = np.zeros_like(cls)
    uid = 1
    for c in np.unique(cls):
        if c == 0: continue
        labeled, n = nd_label(cls == c)
        for i in range(1, n+1):
            out[labeled == i] = uid
            uid += 1
    return out

def instance_to_semantic_approx(inst, cls):
    """Majority-class vote per instance — needs original cls as reference."""
    out = np.zeros_like(inst)
    for i in np.unique(inst):
        if i == 0: continue
        mask = inst == i
        vals, counts = np.unique(cls[mask], return_counts=True)
        out[mask] = vals[np.argmax(counts)]
    return out

# ── Main app ─────────────────────────────────────────────────────────────────
class SegApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segmentation Explorer — Ex 6.5")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1200x780")

        # State
        self.mode = tk.StringVar(value="Synthetic Demo")
        self.seg_type = tk.StringVar(value="All Views")
        self.custom_img = None   # PIL Image (RGB)
        self.cls_map  = None
        self.inst_map = None

        self._build_ui()
        self._load_synthetic()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self, bg=PANEL, height=56)
        topbar.pack(fill="x", side="top")

        tk.Label(topbar, text="⬡  Segmentation Explorer",
                 font=("Courier New", 15, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=20, pady=14)

        tk.Label(topbar, text="Ex 6.5 — CGT 537",
                 font=("Courier New", 10), bg=PANEL, fg=SUBTEXT).pack(side="left", pady=14)

        # ── Left sidebar ──
        sidebar = tk.Frame(self, bg=PANEL, width=220)
        sidebar.pack(fill="y", side="left", padx=0)
        sidebar.pack_propagate(False)

        self._section(sidebar, "IMAGE SOURCE")

        for label in ["Synthetic Demo", "Custom Image"]:
            rb = tk.Radiobutton(sidebar, text=label, variable=self.mode,
                                value=label, command=self._on_mode_change,
                                bg=PANEL, fg=TEXT, selectcolor=ACCENT,
                                activebackground=PANEL, activeforeground=ACCENT,
                                font=("Courier New", 10), relief="flat",
                                cursor="hand2")
            rb.pack(anchor="w", padx=20, pady=3)

        self.upload_btn = tk.Button(sidebar, text="📂  Upload Image",
                                    command=self._upload_image,
                                    bg=BTN_ACTIVE, fg=TEXT,
                                    font=("Courier New", 10), relief="flat",
                                    cursor="hand2", pady=6,
                                    activebackground=ACCENT,
                                    activeforeground=BG)
        self.upload_btn.pack(fill="x", padx=20, pady=(6, 16))

        self.img_label = tk.Label(sidebar, text="No image loaded",
                                   bg=PANEL, fg=SUBTEXT,
                                   font=("Courier New", 8),
                                   wraplength=180)
        self.img_label.pack(padx=20)

        self._section(sidebar, "SEGMENTATION TYPE")

        views = ["All Views", "Semantic", "Instance", "Panoptic",
                 "Sem → Instance", "Inst → Semantic"]
        for v in views:
            rb = tk.Radiobutton(sidebar, text=v, variable=self.seg_type,
                                value=v, command=self._refresh_plot,
                                bg=PANEL, fg=TEXT, selectcolor=ACCENT,
                                activebackground=PANEL, activeforeground=ACCENT,
                                font=("Courier New", 10), relief="flat",
                                cursor="hand2")
            rb.pack(anchor="w", padx=20, pady=2)

        self._section(sidebar, "INFO")
        info = tk.Text(sidebar, bg=PANEL, fg=SUBTEXT, font=("Courier New", 8),
                       relief="flat", wrap="word", height=10, bd=0)
        info.pack(fill="x", padx=16, pady=4)
        info.insert("end",
            "Semantic: class per pixel\n\n"
            "Instance: unique id per object (no class stored)\n\n"
            "Panoptic: class + instance per pixel (union of both)\n\n"
            "Sem→Inst: approx via connected components\n\n"
            "Inst→Sem: needs class lookup"
        )
        info.config(state="disabled")

        # ── Main canvas area ──
        self.canvas_frame = tk.Frame(self, bg=BG)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def _section(self, parent, title):
        tk.Label(parent, text=title,
                 font=("Courier New", 8, "bold"),
                 bg=PANEL, fg=ACCENT).pack(anchor="w", padx=20, pady=(16, 4))
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=20)

    # ── Data loading ──────────────────────────────────────────────────────
    def _load_synthetic(self):
        self.cls_map, self.inst_map = make_synthetic_scene()
        self.img_label.config(text="Synthetic scene\n(2 people, 1 car, 1 bike)")
        self._refresh_plot()

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            # Resize to manageable size
            img.thumbnail((320, 240), Image.LANCZOS)
            self.custom_img = img
            arr = np.array(img)

            # Simulate segmentation via simple colour quantisation
            self.cls_map, self.inst_map = self._pseudo_segment(arr)
            name = os.path.basename(path)
            self.img_label.config(text=f"✓ {name}\n({img.width}×{img.height}px)")
            self.mode.set("Custom Image")
            self._refresh_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

    def _pseudo_segment(self, arr):
        """
        Simple pseudo-segmentation for custom images using k-means colour clustering.
        Returns a class map (0-4) and an instance map.
        """
        from scipy.ndimage import label as nd_label
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3).astype(float)

        # K-means with k=5
        np.random.seed(42)
        k = 5
        idx = np.random.choice(len(flat), k, replace=False)
        centers = flat[idx].copy()
        labels_flat = np.zeros(len(flat), dtype=int)

        for _ in range(15):
            dists = np.linalg.norm(flat[:, None] - centers[None], axis=2)
            labels_flat = np.argmin(dists, axis=1)
            for j in range(k):
                mask = labels_flat == j
                if mask.any():
                    centers[j] = flat[mask].mean(axis=0)

        cls_map = labels_flat.reshape(h, w)

        # Instance map via connected components
        inst_map = np.zeros_like(cls_map)
        uid = 1
        for c in range(1, k):
            labeled, n = nd_label(cls_map == c)
            # Only keep components > 0.5% of image
            min_size = int(0.005 * h * w)
            for i in range(1, n+1):
                component = labeled == i
                if component.sum() > min_size:
                    inst_map[component] = uid
                    uid += 1

        return cls_map, inst_map

    # ── Plotting ──────────────────────────────────────────────────────────
    def _refresh_plot(self):
        if self.cls_map is None:
            return

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        view = self.seg_type.get()

        if view == "All Views":
            self._plot_all()
        else:
            self._plot_single(view)

    def _plot_all(self):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        fig.patch.set_facecolor(BG)
        for ax in axes.flat:
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(ACCENT)
                sp.set_linewidth(0.5)

        maps = {
            "Semantic":       (get_semantic(self.cls_map, self.inst_map),  SEM_COLORS),
            "Instance":       (get_instance(self.cls_map, self.inst_map),  INST_COLORS),
            "Panoptic":       (get_panoptic_display(self.cls_map, self.inst_map), INST_COLORS),
            "Sem → Instance\n(approx)": (semantic_to_instance_approx(self.cls_map), INST_COLORS),
            "Inst → Semantic\n(approx)": (instance_to_semantic_approx(self.inst_map, self.cls_map), SEM_COLORS),
        }

        positions = [(0,0),(0,1),(0,2),(1,0),(1,1)]
        for (r,c), (title, (data, colors)) in zip(positions, maps.items()):
            ax = axes[r][c]
            n = max(data.max()+1, len(colors))
            cmap = ListedColormap(colors[:n] if n <= len(colors) else
                                  colors + ["#888888"]*(n-len(colors)))
            ax.imshow(data, cmap=cmap, interpolation="nearest",
                      vmin=0, vmax=max(data.max(), 1))
            ax.set_title(title, color=TEXT, fontsize=9,
                         fontfamily="monospace", pad=5)

        # Original if custom image
        ax6 = axes[1][2]
        if self.custom_img is not None:
            ax6.imshow(np.array(self.custom_img))
            ax6.set_title("Original Image", color=TEXT, fontsize=9,
                          fontfamily="monospace", pad=5)
        else:
            ax6.set_visible(False)

        plt.tight_layout(pad=1.5)
        self._embed(fig)

    def _plot_single(self, view):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(ACCENT); sp.set_linewidth(1)

        descs = {
            "Semantic":
                "Each pixel is assigned a CLASS LABEL only.\n"
                "Two people → same colour. Instances are merged.",
            "Instance":
                "Each pixel gets a UNIQUE OBJECT ID.\n"
                "Two people → different colours. Class not stored.",
            "Panoptic":
                "Each pixel has BOTH class + instance id.\n"
                "'Things' get unique instances; 'stuff' gets class only.",
            "Sem → Instance":
                "APPROXIMATE: connected-components per class.\n"
                "Fails when two same-class instances touch.",
            "Inst → Semantic":
                "Uses majority-class vote per instance region.\n"
                "Requires original class info — not always available.",
        }

        if view == "Semantic":
            data, colors = get_semantic(self.cls_map, self.inst_map), SEM_COLORS
        elif view == "Instance":
            data, colors = get_instance(self.cls_map, self.inst_map), INST_COLORS
        elif view == "Panoptic":
            data, colors = get_panoptic_display(self.cls_map, self.inst_map), INST_COLORS
        elif view == "Sem → Instance":
            data, colors = semantic_to_instance_approx(self.cls_map), INST_COLORS
        else:
            data, colors = instance_to_semantic_approx(self.inst_map, self.cls_map), SEM_COLORS

        n = max(data.max()+1, 2)
        cmap = ListedColormap((colors + ["#888888"]*20)[:n])
        ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(data.max(),1))
        ax.set_title(view, color=ACCENT, fontsize=14, fontfamily="monospace",
                     fontweight="bold", pad=10)

        desc = descs.get(view, "")
        fig.text(0.5, 0.02, desc, ha="center", color=SUBTEXT,
                 fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                           edgecolor=ACCENT, alpha=0.8))
        plt.tight_layout(rect=[0,0.1,1,1])
        self._embed(fig)

    def _embed(self, fig):
        for w in self.canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── Mode toggle ───────────────────────────────────────────────────────
    def _on_mode_change(self):
        if self.mode.get() == "Synthetic Demo":
            self._load_synthetic()
        else:
            if self.custom_img is None:
                self._upload_image()

if __name__ == "__main__":
    app = SegApp()
    app.mainloop()