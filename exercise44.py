import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# Object generator
# -----------------------------
def make_object(kind, size=64):
    x = np.zeros((size, size))

    if kind == "Rectangle":
        x[20:44, 28:36] = 1.0

    elif kind == "Circle":
        cx, cy, r = size // 2, size // 2, 10
        for i in range(size):
            for j in range(size):
                if (i - cx)**2 + (j - cy)**2 <= r**2:
                    x[i, j] = 1.0

    elif kind == "Two objects":
        x[15:30, 10:20] = 1.0
        x[35:50, 40:55] = 1.0

    elif kind == "Edges":
        x[16:48, 30] = 1.0
        x[16, 16:48] = 1.0
        x[48, 16:48] = 1.0

    return x


# -----------------------------
# Blur kernel (non-blind)
# -----------------------------
def make_kernel(size):
    k = np.ones((size, size))
    return k / k.sum()


# -----------------------------
# MAP inference
# -----------------------------
def deblur(y, kernel, lam, steps=40, lr=0.4):
    x = np.zeros_like(y)

    for _ in range(steps):
        blur = convolve2d(x, kernel, mode="same", boundary="symm")
        resid = blur - y

        grad_data = convolve2d(
            resid, kernel[::-1, ::-1],
            mode="same", boundary="symm"
        )

        laplacian = (
            -4 * x
            + np.roll(x, 1, 0)
            + np.roll(x, -1, 0)
            + np.roll(x, 1, 1)
            + np.roll(x, -1, 1)
        )

        x -= lr * (grad_data + lam * laplacian)

    return x


# -----------------------------
# GUI Application
# -----------------------------
class DeblurringApp:
    def __init__(self, root):
        self.root = root
        root.title("Non-blind Image Deblurring (MRF MAP Inference)")

        # Controls
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(control_frame, text="Object").pack()
        self.object_var = tk.StringVar(value="Rectangle")
        ttk.Combobox(
            control_frame,
            textvariable=self.object_var,
            values=["Rectangle", "Circle", "Two objects", "Edges"],
            state="readonly"
        ).pack()

        ttk.Label(control_frame, text="Kernel size").pack()
        self.kernel_var = tk.IntVar(value=3)
        ttk.Scale(
            control_frame, from_=3, to=11, orient=tk.HORIZONTAL,
            command=lambda _: self.kernel_var.set(int(self.kernel_var.get()))
        ).pack()

        ttk.Label(control_frame, text="Lambda (prior)").pack()
        self.lambda_var = tk.DoubleVar(value=0.1)
        ttk.Scale(
            control_frame, from_=0.001, to=1.0, orient=tk.HORIZONTAL,
            variable=self.lambda_var
        ).pack()

        ttk.Label(control_frame, text="Noise").pack()
        self.noise_var = tk.DoubleVar(value=0.0)
        ttk.Scale(
            control_frame, from_=0.0, to=0.1, orient=tk.HORIZONTAL,
            variable=self.noise_var
        ).pack()

        ttk.Button(control_frame, text="Run", command=self.update).pack(pady=10)

        # Matplotlib figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(9, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

        self.update()

    def update(self):
        obj = self.object_var.get()
        kernel_size = max(3, int(self.kernel_var.get()) | 1)  # odd
        lam = self.lambda_var.get()
        noise = self.noise_var.get()

        x_true = make_object(obj)
        kernel = make_kernel(kernel_size)

        y = convolve2d(x_true, kernel, mode="same", boundary="symm")
        y += noise * np.random.randn(*y.shape)

        x_est = deblur(y, kernel, lam)

        titles = ["Latent x", "Observed y", "MAP estimate"]
        images = [x_true, y, x_est]

        for ax, img, title in zip(self.axes, images, titles):
            ax.clear()
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

        self.canvas.draw()


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DeblurringApp(root)
    root.mainloop()
