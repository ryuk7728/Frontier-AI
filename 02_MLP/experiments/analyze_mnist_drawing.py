import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_data


def erode(mask):
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [
        padded[0:28, 0:28],
        padded[0:28, 1:29],
        padded[0:28, 2:30],
        padded[1:29, 0:28],
        padded[1:29, 1:29],
        padded[1:29, 2:30],
        padded[2:30, 0:28],
        padded[2:30, 1:29],
        padded[2:30, 2:30],
    ]
    return np.logical_and.reduce(neighbors)


def boundary(mask):
    return mask & ~erode(mask)


def run_lengths(mask):
    lengths = []
    for rows in (mask, mask.T):
        for row in rows:
            current = 0
            for value in row:
                if value:
                    current += 1
                elif current:
                    lengths.append(current)
                    current = 0
            if current:
                lengths.append(current)
    return lengths


def local_run_widths(mask):
    horizontal = np.zeros(mask.shape, dtype=int)
    vertical = np.zeros(mask.shape, dtype=int)

    for row in range(mask.shape[0]):
        col = 0
        while col < mask.shape[1]:
            if not mask[row, col]:
                col += 1
                continue
            start = col
            while col < mask.shape[1] and mask[row, col]:
                col += 1
            horizontal[row, start:col] = col - start

    for col in range(mask.shape[1]):
        row = 0
        while row < mask.shape[0]:
            if not mask[row, col]:
                row += 1
                continue
            start = row
            while row < mask.shape[0] and mask[row, col]:
                row += 1
            vertical[start:row, col] = row - start

    # The smaller local run is a better stroke-width proxy than whole-digit runs.
    return np.minimum(horizontal[mask], vertical[mask])


def summarize_mnist_drawing(data_path="data", max_samples=None, threshold=0.15):
    X_train, _, y_train, _ = load_data(str(PROJECT_ROOT / data_path))
    images = X_train.T.reshape(-1, 28, 28)
    labels = np.argmax(y_train, axis=0)

    if max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]

    active_values = []
    boundary_values = []
    interior_values = []
    edge_shell_values = []
    all_run_lengths = []
    bbox_widths = []
    bbox_heights = []
    ink_amounts = []
    erosion_depths = []
    local_widths = []

    for image in images:
        mask = image > threshold
        if not mask.any():
            continue

        rows, cols = np.where(mask)
        bbox_heights.append(int(rows.max() - rows.min() + 1))
        bbox_widths.append(int(cols.max() - cols.min() + 1))
        ink_amounts.append(float(image.sum()))

        active_values.extend(image[mask].tolist())
        current_boundary = boundary(mask)
        boundary_values.extend(image[current_boundary].tolist())
        interior = mask & ~current_boundary
        interior_values.extend(image[interior].tolist())
        all_run_lengths.extend(run_lengths(mask))
        local_widths.extend(local_run_widths(mask).tolist())

        grown_once = np.zeros_like(mask)
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        for r in range(3):
            for c in range(3):
                grown_once |= padded[r : r + 28, c : c + 28]
        edge_shell = grown_once & ~mask
        edge_shell_values.extend(image[edge_shell].tolist())

        depth = 0
        eroded = mask
        while eroded.any():
            depth += 1
            eroded = erode(eroded)
        erosion_depths.append(depth)

    nonzero = np.array(active_values)
    boundary_px = np.array(boundary_values) if boundary_values else np.array([0.0])
    interior_px = np.array(interior_values) if interior_values else np.array([0.0])
    shell_px = np.array(edge_shell_values) if edge_shell_values else np.array([0.0])
    runs = np.array(all_run_lengths) if all_run_lengths else np.array([1])
    local_width_px = np.array(local_widths) if local_widths else np.array([1])
    boundary_ratio = float(boundary_px.mean() / max(interior_px.mean(), 1e-8))
    local_width_median = float(np.median(local_width_px))

    profile = {
        "threshold_used": threshold,
        "samples_analyzed": int(len(images)),
        "pixel_intensity": {
            "active_mean": float(nonzero.mean()),
            "active_median": float(np.median(nonzero)),
            "active_p10": float(np.percentile(nonzero, 10)),
            "active_p25": float(np.percentile(nonzero, 25)),
            "active_p75": float(np.percentile(nonzero, 75)),
            "active_p90": float(np.percentile(nonzero, 90)),
            "max": float(nonzero.max()),
        },
        "stroke_shape": {
            "run_length_median": float(np.median(runs)),
            "run_length_p75": float(np.percentile(runs, 75)),
            "run_length_p90": float(np.percentile(runs, 90)),
            "local_width_median": local_width_median,
            "local_width_p25": float(np.percentile(local_width_px, 25)),
            "local_width_p75": float(np.percentile(local_width_px, 75)),
            "local_width_p90": float(np.percentile(local_width_px, 90)),
            "erosion_depth_median": float(np.median(erosion_depths)),
            "estimated_brush_radius_cells": float(np.clip((local_width_median - 2.5) / 2, 0.35, 0.9)),
        },
        "edge_fade": {
            "boundary_mean": float(boundary_px.mean()),
            "boundary_median": float(np.median(boundary_px)),
            "interior_mean": float(interior_px.mean()),
            "outside_shell_mean": float(shell_px.mean()),
            "boundary_to_interior_ratio": boundary_ratio,
        },
        "digit_extent": {
            "bbox_width_median": float(np.median(bbox_widths)),
            "bbox_height_median": float(np.median(bbox_heights)),
            "bbox_width_p90": float(np.percentile(bbox_widths, 90)),
            "bbox_height_p90": float(np.percentile(bbox_heights, 90)),
            "ink_sum_median": float(np.median(ink_amounts)),
        },
        "suggested_ui": {
            "brush_radius_cells": float(np.clip((local_width_median - 2.5) / 2, 0.35, 0.9)),
            "edge_sigma_cells": float(np.clip(0.35 + (1 - boundary_ratio) * 0.45, 0.35, 0.6)),
            "draw_strength": float(np.clip(np.percentile(nonzero, 90), 0.75, 1.0)),
            "model_box_size": 20,
        },
    }
    return profile


def main():
    profile = summarize_mnist_drawing()
    output_path = PROJECT_ROOT / "experiments" / "mnist_drawing_profile.json"
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("MNIST drawing profile")
    print(json.dumps(profile, indent=2))
    print(f"\nSaved profile to {output_path}")


if __name__ == "__main__":
    main()
