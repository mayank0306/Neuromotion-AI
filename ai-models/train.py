"""Validate a training configuration and reserve an explicit AI training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    """Load and report the configured experiment without claiming a completed model run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    print(f"Prepared experiment: {config['experiment_name']}")
    print("Connect a versioned dataset and MLflow tracking URI before training a clinical model.")


if __name__ == "__main__":
    main()
