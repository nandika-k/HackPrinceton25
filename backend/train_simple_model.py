"""
Entry-point script to train and persist the lightweight PulseGuard model.

Usage:
    python backend/train_simple_model.py
"""

from simple_model import DEFAULT_MODEL_PATH
from simple_model.training import train_and_save


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    if model_path.exists():
        print(f"Model already exists at {model_path}. Re-training in place.")
    train_and_save(model_path)


if __name__ == "__main__":
    main()
