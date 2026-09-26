"""
Yoga Pose Classifier Training Pipeline
Trains an MLP model on Kaggle Yoga 5-Class dataset.
"""

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import structlog
from typing import Tuple, List
import cv2
import os

logger = structlog.get_logger(__name__)


class YogaPoseDataset(Dataset):
    """Dataset for yoga pose classification."""

    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        """
        Initialize dataset.

        Args:
            images_dir: Directory containing yoga pose images
            labels_csv: CSV file with image filenames and labels
            transform: Optional transform to apply to images
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load labels
        import pandas as pd
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            self.image_paths.append(self.images_dir / row['filename'])
            self.labels.append(row['label'])

        # Create label mapping
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class YogaPoseMLP(nn.Module):
    """Multi-Layer Perceptron for yoga pose classification."""

    def __init__(self, input_features: int, hidden_layers: List[int], num_classes: int, dropout_rate: float = 0.3):
        """
        Initialize MLP.

        Args:
            input_features: Number of input features (99 for 33 landmarks * 3 coords)
            hidden_layers: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(YogaPoseMLP, self).__init__()

        layers = []
        prev_size = input_features

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def extract_landmarks_from_image(image: np.ndarray) -> np.ndarray:
    """
    Extract MediaPipe landmarks from image.
    This is a placeholder - in reality, we'd use the pose estimator service.

    Args:
        image: RGB image as numpy array

    Returns:
        Flattened landmark features (99 values: 33 landmarks * 3 coords)
    """
    # Placeholder implementation
    # In practice, this would use MediaPipe to extract landmarks
    # For now, return random landmarks for demonstration
    landmarks = np.random.rand(33, 3)  # 33 landmarks, x,y,z
    return landmarks.flatten()


def train_yoga_classifier(config_path: str):
    """
    Train yoga pose classifier.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Starting yoga pose classifier training",
                experiment=config['experiment_name'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize dataset
    dataset_config = config['dataset']
    dataset = YogaPoseDataset(
        images_dir=dataset_config['images_dir'],
        labels_csv=dataset_config['labels_csv']
    )

    # Split dataset
    train_size = int(config['training']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Initialize model
    model_config = config['model']
    model = YogaPoseMLP(
        input_features=model_config['input_features'],
        hidden_layers=model_config['hidden_layers'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    num_epochs = config['training']['epochs']
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # In reality, we'd extract landmarks here
            # For now, we'll simulate landmark extraction
            batch_features = []
            for img in data.numpy():
                landmarks = extract_landmarks_from_image(img)
                batch_features.append(landmarks)
            features = torch.FloatTensor(np.array(batch_features)).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                batch_features = []
                for img in data.numpy():
                    landmarks = extract_landmarks_from_image(img)
                    batch_features.append(landmarks)
                features = torch.FloatTensor(np.array(batch_features)).to(device)
                target = target.to(device)

                outputs = model(features)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_acc = 100 * val_correct / val_total

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}]",
            train_loss=f"{train_loss/len(train_loader):.4f}",
            train_acc=f"{train_acc:.2f}%",
            val_loss=f"{val_loss/len(val_loader):.4f}",
            val_acc=f"{val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': dataset.label_to_idx,
                'idx_to_label': dataset.idx_to_label,
                'config': config
            }, f"best_yoga_classifier_{config['experiment_name']}.pth")
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    logger.info("Training completed", best_validation_acc=f"{best_val_acc:.2f}%")
    return model


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python yoga_classifier_train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    train_yoga_classifier(config_path)