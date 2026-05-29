import json
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

try:
    from .model import NN
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.model import NN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_SIZE = 28
CELL_SIZE = 18
CANVAS_SIZE = GRID_SIZE * CELL_SIZE


def load_weights():
    path = PROJECT_ROOT / "outputs" / "models" / "mnist_weights.npz"
    data = np.load(path)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]

    return W1, b1, W2, b2


def load_drawing_profile():
    path = PROJECT_ROOT / "experiments" / "mnist_drawing_profile.json"
    defaults = {
        "brush_radius_cells": 0.75,
        "edge_sigma_cells": 0.45,
        "draw_strength": 0.95,
        "model_box_size": 20,
    }
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as file:
        profile = json.load(file)
    return {**defaults, **profile.get("suggested_ui", {})}


class MnistDrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.profile = load_drawing_profile()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.last_cell = None

        root.title("MNIST Drawing Inference")
        root.resizable(False, False)

        app = tk.Frame(root, padx=12, pady=12)
        app.pack()

        self.canvas = tk.Canvas(
            app,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#444",
        )
        self.canvas.grid(row=0, column=0, rowspan=3)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        panel = tk.Frame(app, padx=14)
        panel.grid(row=0, column=1, sticky="n")

        self.prediction_label = tk.Label(
            panel,
            text="Draw a digit",
            font=("Segoe UI", 22, "bold"),
            width=14,
            anchor="w",
        )
        self.prediction_label.pack(anchor="w", pady=(0, 12))

        self.rank_rows = []
        for digit in range(10):
            row = tk.Frame(panel)
            row.pack(fill="x", pady=2)

            digit_label = tk.Label(row, text=str(digit), font=("Consolas", 12), width=3)
            digit_label.pack(side="left")

            bar = tk.Canvas(row, width=160, height=16, bg="#1f1f1f", highlightthickness=0)
            bar.pack(side="left", padx=6)

            prob_label = tk.Label(row, text="0.00%", font=("Consolas", 11), width=8, anchor="e")
            prob_label.pack(side="left")

            self.rank_rows.append((row, digit_label, bar, prob_label))

        controls = tk.Frame(panel)
        controls.pack(fill="x", pady=(14, 0))

        clear_button = tk.Button(controls, text="Clear", command=self.clear)
        clear_button.pack(side="left")

        self.param_label = tk.Label(
            panel,
            text=(
                f"brush {self.profile['brush_radius_cells']:.2f}, "
                f"fade {self.profile['edge_sigma_cells']:.2f}"
            ),
            font=("Segoe UI", 9),
            fg="#666",
        )
        self.param_label.pack(anchor="w", pady=(10, 0))

        self.paint_grid()
        self.update_prediction()

    def start_draw(self, event):
        self.last_cell = self.event_to_cell(event)
        self.add_stroke_at(*self.last_cell)
        self.paint_grid()
        self.update_prediction()

    def draw(self, event):
        cell = self.event_to_cell(event)
        if self.last_cell is None:
            self.last_cell = cell

        r0, c0 = self.last_cell
        r1, c1 = cell
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for step in range(steps + 1):
            t = step / steps
            row = round(r0 + (r1 - r0) * t)
            col = round(c0 + (c1 - c0) * t)
            self.add_stroke_at(row, col)

        self.last_cell = cell
        self.paint_grid()
        self.update_prediction()

    def stop_draw(self, _event):
        self.last_cell = None

    def event_to_cell(self, event):
        col = int(np.clip(event.x // CELL_SIZE, 0, GRID_SIZE - 1))
        row = int(np.clip(event.y // CELL_SIZE, 0, GRID_SIZE - 1))
        return row, col

    def add_stroke_at(self, row, col):
        radius = self.profile["brush_radius_cells"]
        sigma = self.profile["edge_sigma_cells"]
        strength = self.profile["draw_strength"]
        reach = int(np.ceil(radius + 2 * sigma))

        for r in range(max(0, row - reach), min(GRID_SIZE, row + reach + 1)):
            for c in range(max(0, col - reach), min(GRID_SIZE, col + reach + 1)):
                distance = np.sqrt((r - row) ** 2 + (c - col) ** 2)
                if distance <= reach:
                    if distance <= radius:
                        value = strength
                    else:
                        value = strength * np.exp(-((distance - radius) ** 2) / (2 * sigma**2))
                    self.grid[r, c] = max(self.grid[r, c], value)

    def resize_image(self, image, new_height, new_width):
        if new_height <= 0 or new_width <= 0:
            return np.zeros((max(new_height, 1), max(new_width, 1)), dtype=float)

        source_height, source_width = image.shape
        if source_height == new_height and source_width == new_width:
            return image.copy()

        row_positions = np.linspace(0, source_height - 1, new_height)
        col_positions = np.linspace(0, source_width - 1, new_width)
        resized = np.zeros((new_height, new_width), dtype=float)

        for out_row, source_row in enumerate(row_positions):
            row0 = int(np.floor(source_row))
            row1 = min(row0 + 1, source_height - 1)
            row_weight = source_row - row0
            for out_col, source_col in enumerate(col_positions):
                col0 = int(np.floor(source_col))
                col1 = min(col0 + 1, source_width - 1)
                col_weight = source_col - col0

                top = (1 - col_weight) * image[row0, col0] + col_weight * image[row0, col1]
                bottom = (1 - col_weight) * image[row1, col0] + col_weight * image[row1, col1]
                resized[out_row, out_col] = (1 - row_weight) * top + row_weight * bottom

        return resized

    def model_input_grid(self):
        if self.grid.max() == 0:
            return self.grid.copy()

        mask = self.grid > 0.05
        rows, cols = np.where(mask)
        cropped = self.grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

        target_box = int(self.profile.get("model_box_size", 20))
        scale = target_box / max(cropped.shape)
        new_height = max(1, int(round(cropped.shape[0] * scale)))
        new_width = max(1, int(round(cropped.shape[1] * scale)))
        resized = self.resize_image(cropped, new_height, new_width)

        output = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        mass = resized.sum()
        if mass > 0:
            row_coords, col_coords = np.indices(resized.shape)
            center_row = float((row_coords * resized).sum() / mass)
            center_col = float((col_coords * resized).sum() / mass)
        else:
            center_row = (new_height - 1) / 2
            center_col = (new_width - 1) / 2

        top = int(round((GRID_SIZE - 1) / 2 - center_row))
        left = int(round((GRID_SIZE - 1) / 2 - center_col))

        dest_r0 = max(0, top)
        dest_c0 = max(0, left)
        dest_r1 = min(GRID_SIZE, top + new_height)
        dest_c1 = min(GRID_SIZE, left + new_width)
        src_r0 = max(0, -top)
        src_c0 = max(0, -left)
        src_r1 = src_r0 + (dest_r1 - dest_r0)
        src_c1 = src_c0 + (dest_c1 - dest_c0)

        output[dest_r0:dest_r1, dest_c0:dest_c1] = resized[src_r0:src_r1, src_c0:src_c1]
        return np.clip(output, 0, 1)

    def paint_grid(self):
        self.canvas.delete("all")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = int(np.clip(self.grid[row, col], 0, 1) * 255)
                color = f"#{value:02x}{value:02x}{value:02x}"
                x0 = col * CELL_SIZE
                y0 = row * CELL_SIZE
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + CELL_SIZE,
                    y0 + CELL_SIZE,
                    fill=color,
                    outline="#242424",
                )

    def update_prediction(self):
        model_grid = self.model_input_grid()
        X = model_grid.reshape(1, -1).T
        _, probs, _ = self.model.predict(X)
        probs = probs[:, 0]
        ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)

        if self.grid.max() == 0:
            self.prediction_label.config(text="Draw a digit")
        else:
            self.prediction_label.config(text=f"Prediction: {ranked[0][0]}")

        for display_index, (digit, probability) in enumerate(ranked):
            row, digit_label, bar, prob_label = self.rank_rows[display_index]
            digit_label.config(text=str(digit))
            prob_label.config(text=f"{probability * 100:5.2f}%")
            bar.delete("all")
            width = int(160 * probability)
            color = "#2f80ed" if display_index == 0 else "#777"
            bar.create_rectangle(0, 0, width, 16, fill=color, outline="")
            row.pack_configure(fill="x", pady=2)

    def clear(self):
        self.grid.fill(0)
        self.paint_grid()
        self.update_prediction()


def run_app():
    model = NN()
    W1, b1, W2, b2 = load_weights()
    model.W1, model.b1, model.W2, model.b2 = W1, b1, W2, b2

    root = tk.Tk()
    MnistDrawingApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    run_app()
