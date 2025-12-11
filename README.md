# ModelForge

A scaffolded computer vision project inspired by YOLO and DETR. The repository is structured for rapid experimentation with detection models, covering data preparation, model definitions, training, evaluation, and inference.

## Project layout

```
ModelForge/
├── data/                 # Datasets, transforms, and dataloaders
├── models/               # Backbone, heads, YOLO/DETR implementations
├── training/             # Trainers, losses, schedulers, configs
├── scripts/              # CLI entry points for train/eval/test/inference
├── logs/                 # Placeholder for training logs
├── output/               # Model checkpoints and results
├── requirements.txt      # Python dependencies
└── README.md
```

## Key components

- **Data pipeline** (`data/`): Includes dataset wrappers, composable transforms, and dataloader creation helpers for COCO-style annotations.
- **Models** (`models/`): Minimal YOLO-style and DETR-style architectures with shared ResNet backbones and task-specific heads.
- **Training loop** (`training/`): Trainer class, loss dispatching, learning-rate scheduler helpers, and central configuration in `config.py`.
- **Scripts** (`scripts/`): Ready-to-run entrypoints for training, validation, testing, and inference demonstrations.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset under `data/train`, `data/val`, and `data/test`, and supply annotation dictionaries to the loaders.
3. Adjust hyperparameters in `training/config.py`.
4. Launch training:

   ```bash
   python scripts/train.py
   ```

The provided code uses lightweight stubs for loss and evaluation to illustrate the project flow; extend these components with task-specific logic for real-world use.
