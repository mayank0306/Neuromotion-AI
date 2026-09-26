"""
Rehabilitation Assessment Model Training Pipeline
Trains a multi-task MLP model on UI-PRMD (classification) and KIMORE (regression) datasets.
"""

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import structlog
from typing import Tuple, List, Dict, Any
import pandas as pd
import os

logger = structlog.get_logger(__name__)


class RehabDataset(Dataset):
    """Dataset for rehabilitation movement assessment."""

    def __init__(self, data_path: str, dataset_type: str, split: str = 'train',
                 transform=None, config: Dict[str, Any] = None):
        """
        Initialize dataset.

        Args:
            data_path: Root directory for dataset
            dataset_type: Either 'ui_prmd' or 'kimore'
            split: 'train', 'validation', or 'test'
            transform: Optional transform to apply to data
            config: Configuration dictionary with split specifications
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.config = config or {}

        # Load data based on dataset type
        if dataset_type == 'ui_prmd':
            self._load_ui_prmd()
        elif dataset_type == 'kimore':
            self._load_kimore()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        logger.info(f"Loaded {dataset_type} {split} dataset with {len(self.samples)} samples")

    def _load_ui_prmd(self):
        """Load UI-PRMD dataset (classification: correct/incorrect)."""
        # UI-PRMD provides CSV files with joint angles and labels
        # For demonstration, we'll simulate loading
        splits_config = self.config.get('datasets', {}).get('ui_prmd', {})

        # Determine subjects for this split
        if self.split == 'train':
            subject_list = splits_config.get('train_subjects', [1, 2, 3, 4, 5])
        elif self.split == 'validation':
            subject_list = splits_config.get('validation_subjects', [6, 7])
        else:  # test
            subject_list = splits_config.get('test_subjects', [8, 9, 10])

        # Simulate loading data
        self.samples = []
        self.labels = []  # 0 = incorrect, 1 = correct

        # In reality, we'd load actual CSV files for each subject/exercise
        # For now, generate simulated data
        num_samples_per_subject = 50  # Simulated
        for subject_id in subject_list:
            for exercise_id in range(1, 11):  # 10 exercises
                for rep in range(num_samples_per_subject):
                    # Simulate landmark features (99 dimensions)
                    features = np.random.rand(99).astype(np.float32)
                    # Simulate label (slight bias toward correct)
                    label = np.random.choice([0, 1], p=[0.3, 0.7])
                    self.samples.append(features)
                    self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)

    def _load_kimore(self):
        """Load KIMORE dataset (regression: quality score 0-100)."""
        # KIMORE provides skeleton data and clinician-rated quality scores
        splits_config = self.config.get('datasets', {}).get('kimore', {})

        # For stratified split, we'd need to load labels first
        # For simulation, we'll just use random split
        total_samples = 1000
        # In reality, we'd count actual samples
        if self.split == 'train':
            target_count = int(0.7 * 1000)  # Simulated
        elif self.split == 'validation':
            target_count = int(0.15 * 1000)
        else:  # test
            target_count = int(0.15 * 1000)

        # Simulate loading data
        self.samples = []
        self.labels = []  # Quality score 0-100

        for i in range(target_count):
            # Simulate landmark features (99 dimensions)
            features = np.random.rand(99).astype(np.float32)
            # Simulate quality score (higher scores for healthier subjects)
            quality_score = np.random.uniform(0, 100)
            self.samples.append(features)
            self.labels.append(quality_score)

        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = self.samples[idx]
        if self.transform:
            features = self.transform(features)

        if self.dataset_type == 'ui_prmd':
            label = self.labels[idx]
            return features, label
        else:  # kimore
            label = self.labels[idx]
            return features, label


class RehabQualityMLP(nn.Module):
    """Multi-Layer Perceptron with dual heads for classification and regression."""

    def __init__(self, input_features: int, hidden_layers: List[int],
                 num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize MLP with dual output heads.

        Args:
            input_features: Number of input features (99 for 33 landmarks * 3 coords)
            hidden_layers: List of hidden layer sizes
            num_classes: Number of classes for classification head (2 for correct/incorrect)
            dropout_rate: Dropout rate
        """
        super(RehabQualityMLP, self).__init__()

        # Shared feature extractor
        layers = []
        prev_size = input_features

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.shared_network = nn.Sequential(*layers)

        # Classification head (for UI-PRMD: correct/incorrect)
        self.classification_head = nn.Linear(prev_size, num_classes)

        # Regression head (for KIMORE: quality score 0-100)
        self.regression_head = nn.Linear(prev_size, 1)

    def forward(self, x):
        # Shared feature extraction
        features = self.shared_network(x)

        # Dual outputs
        classification_logits = self.classification_head(features)
        regression_output = self.regression_head(features)

        return classification_logits, regression_output.squeeze(-1)


def train_rehab_model(config_path: str):
    """
    Train rehabilitation assessment model.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Starting rehabilitation assessment model training",
                experiment=config['experiment_name'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize datasets for both UI-PRMD and KIMORE
    datasets_config = config.get('datasets', {})

    # UI-PRMD Dataset (Classification)
    ui_prmd_config = datasets_config.get('ui_prmd', {})
    ui_prmd_train = RehabDataset(
        data_path=ui_prmd_config.get('data_path', '${DATASET_PATH}/ui-prmd'),
        dataset_type='ui_prmd',
        split='train',
        config=config
    )
    ui_prmd_val = RehabDataset(
        data_path=ui_prmd_config.get('data_path', '${DATASET_PATH}/ui-prmd'),
        dataset_type='ui_prmd',
        split='validation',
        config=config
    )

    # KIMORE Dataset (Regression)
    kimore_config = datasets_config.get('kimore', {})
    kimore_train = RehabDataset(
        data_path=kimore_config.get('data_path', '${DATASET_PATH}/kimore'),
        dataset_type='kimore',
        split='train',
        config=config
    )
    kimore_val = RehabDataset(
        data_path=kimore_config.get('data_path', '${DATASET_PATH}/kimore'),
        dataset_type='kimore',
        split='validation',
        config=config
    )

    # Create data loaders
    batch_size = config['training']['batch_size']
    ui_prmd_train_loader = DataLoader(ui_prmd_train, batch_size=batch_size, shuffle=True)
    ui_prmd_val_loader = DataLoader(ui_prmd_val, batch_size=batch_size, shuffle=False)
    kimore_train_loader = DataLoader(kimore_train, batch_size=batch_size, shuffle=True)
    kimore_val_loader = DataLoader(kimore_val, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_config = config['model']
    model = RehabQualityMLP(
        input_features=model_config['input_features'],
        hidden_layers=model_config['hidden_layers'],
        num_classes=model_config['classification_head']['num_classes'],
        dropout_rate=0.3  # Default dropout rate
    ).to(device)

    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    num_epochs = config['training']['epochs']
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        ui_prmd_train_loss = 0.0
        kimore_train_loss = 0.0
        ui_prmd_correct = 0
        ui_prmd_total = 0

        # Train on UI-PRMD (classification)
        for batch_idx, (data, target) in enumerate(ui_prmd_train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            class_logits, reg_output = model(data)

            # Classification loss only for UI-PRMD
            class_loss = classification_criterion(class_logits, target)
            class_loss.backward(retain_graph=True)  # Retain graph for regression loss

            train_loss += class_loss.item()
            ui_prmd_train_loss += class_loss.item()

            _, predicted = torch.max(class_logits.data, 1)
            ui_prmd_total += target.size(0)
            ui_prmd_correct += (predicted == target).sum().item()

        # Train on KIMORE (regression)
        for batch_idx, (data, target) in enumerate(kimore_train_loader):
            data = data.to(device)
            target = target.to(device)

            # Zero gradients for regression pass
            optimizer.zero_grad()
            class_logits, reg_output = model(data)

            # Regression loss only for KIMORE
            reg_loss = regression_criterion(reg_output, target)
            reg_loss.backward()

            train_loss += reg_loss.item()
            kimore_train_loss += reg_loss.item()

        optimizer.step()

        # Calculate metrics
        ui_prmd_acc = 100 * ui_prmd_correct / ui_prmd_total if ui_prmd_total > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        ui_prmd_val_loss = 0.0
        kimore_val_loss = 0.0
        ui_prmd_val_correct = 0
        ui_prmd_val_total = 0
        kimore_val_mae = 0.0
        kimore_val_total = 0

        with torch.no_grad():
            # Validate UI-PRMD
            for data, target in ui_prmd_val_loader:
                data = data.to(device)
                target = target.to(device)
                class_logits, reg_output = model(data)
                class_loss = classification_criterion(class_logits, target)
                val_loss += class_loss.item()
                ui_prmd_val_loss += class_loss.item()

                _, predicted = torch.max(class_logits.data, 1)
                ui_prmd_val_total += target.size(0)
                ui_prmd_val_correct += (predicted == target).sum().item()

            # Validate KIMORE
            for data, target in kimore_val_loader:
                data = data.to(device)
                target = target.to(device)
                class_logits, reg_output = model(data)
                reg_loss = regression_criterion(reg_output, target)
                val_loss += reg_loss.item()
                kimore_val_loss += reg_loss.item()

                kimore_val_mae += torch.abs(reg_output - target).sum().item()
                kimore_val_total += target.size(0)

        # Calculate validation metrics
        ui_prmd_val_acc = 100 * ui_prmd_val_correct / ui_prmd_val_total if ui_prmd_val_total > 0 else 0
        kimore_val_mae = kimore_val_mae / kimore_val_total if kimore_val_total > 0 else 0
        avg_val_loss = val_loss / (len(ui_prmd_val_loader) + len(kimore_val_loader))

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}]",
            ui_prmd_train_loss=f"{ui_prmd_train_loss/len(ui_prmd_train_loader):.4f}",
            ui_prmd_val_loss=f"{ui_prmd_val_loss/len(ui_prmd_val_loader):.4f}",
            ui_prmd_train_acc=f"{ui_prmd_acc:.2f}%",
            ui_prmd_val_acc=f"{ui_prmd_val_acc:.2f}%",
            kimore_train_loss=f"{kimore_train_loss/len(kimore_train_loader):.4f}",
            kimore_val_loss=f"{kimore_val_loss/len(kimore_val_loader):.4f}",
            kimore_val_mae=f"{kimore_val_mae:.4f}",
            total_loss=f"{train_loss/(len(ui_prmd_train_loader)+len(kimore_train_loader)):.4f}",
            val_loss=f"{avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ui_prmd_val_acc': ui_prmd_val_acc,
                'kimore_val_mae': kimore_val_mae,
                'val_loss': avg_val_loss,
                'config': config
            }, f"best_rehab_model_{config['experiment_name']}.pth")
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")

    logger.info("Training completed",
                best_ui_prmd_acc=f"{ui_prmd_val_acc:.2f}%",
                best_kimore_mae=f"{kimore_val_mae:.4f}")
    return model


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rehab_assessment_train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    train_rehab_model(config_path)