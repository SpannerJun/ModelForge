"""Central configuration for experiments."""

config = {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "num_epochs": 100,
    "num_classes": 80,
    "num_workers": 4,
    "device": "cuda",
    "model_name": "detr",
    "num_queries": 100,
}
