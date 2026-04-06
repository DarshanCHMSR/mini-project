from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.modeling import compute_binary_metrics

LOGGER = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MultiModalDataset(Dataset):
    def __init__(self, sequences: np.ndarray, tabular: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
        self.tabular = torch.from_numpy(tabular).float()
        self.targets = torch.from_numpy(targets.astype(np.float32)).float()

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.sequences[index], self.tabular[index], self.targets[index]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_output: Tensor) -> Tensor:
        weights = torch.softmax(self.score(sequence_output).squeeze(-1), dim=1)
        return torch.sum(sequence_output * weights.unsqueeze(-1), dim=1)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        tabular_input_dim: int,
        hidden_size: int,
        lstm_layers: int,
        tabular_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
        use_attention: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.use_attention = use_attention
        self.attention_pool = AttentionPooling(hidden_size * 2)
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tabular_head = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.BatchNorm1d(tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim // 2),
            nn.BatchNorm1d(tabular_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + tabular_hidden_dim // 2, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def encode_sequence(self, sequences: Tensor) -> Tensor:
        lstm_output, (hidden_state, _) = self.lstm(sequences)
        if self.use_attention:
            pooled = self.attention_pool(lstm_output)
        else:
            pooled = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        return self.sequence_head(pooled)

    def forward(self, sequences: Tensor, tabular: Tensor) -> Tensor:
        sequence_features = self.encode_sequence(sequences)
        tabular_features = self.tabular_head(tabular)
        fused = torch.cat((sequence_features, tabular_features), dim=1)
        logits = self.fusion_head(fused)
        return logits.squeeze(-1)


@dataclass
class TrainingResult:
    history: dict[str, list[float]]
    best_epoch: int
    best_val_f1: float


def build_dataloader(
    sequences: np.ndarray,
    tabular: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MultiModalDataset(sequences, tabular, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: FusionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    for sequences, tabular, targets in loader:
        sequences = sequences.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(sequences, tabular)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.append(probabilities)
        all_targets.append(targets.detach().cpu().numpy())
        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    average_loss = total_loss / max(total_examples, 1)
    return average_loss, scores, targets


def train_fusion_model(
    model: FusionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> TrainingResult:
    train_targets = train_loader.dataset.targets.numpy()
    pos_count = max(float(train_targets.sum()), 1.0)
    neg_count = max(float((train_targets == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_scores, train_targets_np = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_scores, val_targets_np = _run_epoch(model, val_loader, criterion, device)

        train_pred = (train_scores >= 0.5).astype(int)
        val_pred = (val_scores >= 0.5).astype(int)
        train_metrics = compute_binary_metrics(train_targets_np.astype(int), train_pred, train_scores)
        val_metrics = compute_binary_metrics(val_targets_np.astype(int), val_pred, val_scores)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1_score"])
        history["val_f1"].append(val_metrics["f1_score"])

        scheduler.step(val_loss)

        improved = val_metrics["f1_score"] > best_val_f1 or (
            abs(val_metrics["f1_score"] - best_val_f1) < 1e-6 and val_loss < best_val_loss
        )
        if improved:
            best_val_f1 = val_metrics["f1_score"]
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch}, checkpoint_path)
        else:
            epochs_without_improvement += 1

        epoch_duration = time.perf_counter() - epoch_start
        LOGGER.info(
            "Epoch %s/%s | train_loss=%.4f val_loss=%.4f train_f1=%.4f val_f1=%.4f train_acc=%.4f val_acc=%.4f lr=%.6f epoch_time=%s",
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            train_metrics["f1_score"],
            val_metrics["f1_score"],
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            optimizer.param_groups[0]["lr"],
            format_duration(epoch_duration),
            extra={"eta": format_duration((max_epochs - epoch) * epoch_duration)},
        )

        if epochs_without_improvement >= patience:
            LOGGER.info("Early stopping triggered at epoch %s", epoch, extra={"eta": "00:00:00"})
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(history=history, best_epoch=best_epoch, best_val_f1=best_val_f1)


@torch.no_grad()
def predict_fusion_model(model: FusionClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for sequences, tabular, targets in loader:
        logits = model(sequences.to(device), tabular.to(device))
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_targets.append(targets.numpy())
    return np.concatenate(all_scores), np.concatenate(all_targets)


def export_torchscript(model: FusionClassifier, example_sequences: np.ndarray, example_tabular: np.ndarray, destination: Path) -> None:
    model_cpu = model.to("cpu").eval()
    traced = torch.jit.trace(
        model_cpu,
        (
            torch.from_numpy(example_sequences[:1]).float(),
            torch.from_numpy(example_tabular[:1]).float(),
        ),
    )
    traced.save(str(destination))
    LOGGER.info("Saved TorchScript model to %s", destination)


def export_onnx(
    model: FusionClassifier,
    example_sequences: np.ndarray,
    example_tabular: np.ndarray,
    destination: Path,
) -> None:
    model_cpu = model.to("cpu").eval()
    torch.onnx.export(
        model_cpu,
        (
            torch.from_numpy(example_sequences[:1]).float(),
            torch.from_numpy(example_tabular[:1]).float(),
        ),
        str(destination),
        input_names=["sequences", "tabular"],
        output_names=["logits"],
        dynamic_axes={
            "sequences": {0: "batch_size"},
            "tabular": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    LOGGER.info("Saved ONNX model to %s", destination)
