# ML Project

A clean, modular machine learning project structure designed for reproducibility and ease of use.

---

## Project Structure

```
project_name/
│
├── data/              # Raw and processed datasets
├── src/               # Core source code (model, training, inference)
├── configs/           # YAML configuration files for experiments
├── experiments/       # Logs, results, and metadata for each run
├── outputs/           # Saved models, plots, and predictions
│   ├── models/
│   ├── plots/
│   └── predictions/
├── notebooks/         # Exploratory and debugging notebooks
├── tests/             # Unit tests for src modules
├── main.py            # Entry point — ties everything together
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### `data/`
Place your raw datasets here. Processed or intermediate data can also live here in subdirectories (e.g., `data/raw/`, `data/processed/`). This folder is typically excluded from version control via `.gitignore` if data is large — use a data registry or download script instead.

### `src/`
The heart of the project. All importable modules live here:

| File | Purpose |
|------|---------|
| `model.py` | Model architecture definition |
| `train.py` | Training loop, loss computation, optimizer logic |
| `dataset.py` | Dataset class, data loading, and preprocessing |
| `inference.py` | Prediction and evaluation on new data |
| `utils.py` | Shared helper functions (logging, metrics, etc.) |

Add an `__init__.py` inside `src/` to make it a proper Python package so imports work cleanly across the project.

### `configs/`
YAML files that define hyperparameters, paths, and experiment settings. One config per experiment keeps runs reproducible and easy to compare. Example: `configs/baseline.yaml`, `configs/lr_sweep.yaml`.

### `experiments/`
Each run generates a subdirectory here (e.g., `experiments/run_001/`) containing the config used, training logs, and results. Keeps your workspace clean and makes it easy to compare runs.

### `outputs/`
All artifacts generated during training and inference:
- `models/` — saved checkpoints (`.pt`, `.pth`, etc.)
- `plots/` — loss curves, confusion matrices, visualizations
- `predictions/` — generated outputs from inference

### `notebooks/`
Jupyter notebooks for data exploration, debugging, and visualization. These are not part of the training pipeline — keep them as scratchpads, not production code.

### `tests/`
Unit tests for your `src/` modules. Even simple sanity checks (e.g., "does the dataset class return the right shape?") save a lot of debugging time. Run with `pytest`.

### `main.py`
The single entry point to run the full pipeline. Imports from `src/` and wires everything together — training, evaluation, inference — controlled by config flags or CLI args.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data

Place your dataset inside the `data/` directory. Update the paths in your config file under `configs/` accordingly.

### 5. Configure your run

Edit or create a config file in `configs/`. Set your hyperparameters, data paths, and output directories there.

### 6. Run training

```bash
python main.py --config configs/baseline.yaml
```

### 7. Run inference

```bash
python main.py --config configs/baseline.yaml --mode inference
```

Outputs (model checkpoints, plots, predictions) will be saved to the `outputs/` directory.

---

## Running Tests

```bash
pytest tests/
```

---

## Notes

- Keep `src/` free of side effects — it's a library, not a script. All execution starts from `main.py`.
- Commit your config files, not your outputs. Add `outputs/` and `data/` to `.gitignore`.
- Use `notebooks/` for exploration, but move any reusable logic back into `src/`.
