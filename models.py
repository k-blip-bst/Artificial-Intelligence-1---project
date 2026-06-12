import torch
import torch.nn as nn
from torchvision import models
from src import NUM_CLASSES


def build_resnet18(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_efficientnet_b0(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


class PoseMLP(nn.Module):
    """3-layer MLP for 51-dim keypoint vector → 2 classes (neutral / non_neutral)."""

    def __init__(self, input_dim: int = 51, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseRNN(nn.Module):
    """LSTM on 17-keypoint sequence (17 timesteps × 3 features) → 2 classes.

    Input shape: (batch, 17, 3)  — each keypoint is one timestep.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 17, 3)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * directions, batch, hidden)
        # take last layer's hidden state (forward + backward concatenated)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.classifier(last)
