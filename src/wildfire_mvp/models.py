from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("torch and torchvision are required for model definitions") from exc
    return torch, nn, models


class StaticResNet18:
    def __new__(cls, in_channels: int = 4, dropout: float = 0.2):
        torch, nn, models = _require_torch()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, 1),
        )
        return backbone


class TemporalAttentionClassifier:
    def __new__(cls, in_channels: int = 4, dropout: float = 0.2, hidden_dim: int = 256, heads: int = 4):
        torch, nn, models = _require_torch()

        class _TemporalAttentionClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                backbone = models.resnet18(weights=None)
                backbone.conv1 = nn.Conv2d(
                    in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
                feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.encoder = backbone
                self.projection = nn.Linear(feature_dim, hidden_dim)
                self.temporal = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                batch, steps, channels, height, width = x.shape
                x = x.view(batch * steps, channels, height, width)
                embeddings = self.encoder(x)
                embeddings = self.projection(embeddings)
                embeddings = embeddings.view(batch, steps, -1)
                attended, _ = self.temporal(embeddings, embeddings, embeddings)
                pooled = self.norm(attended + embeddings).mean(dim=1)
                return self.classifier(pooled).squeeze(-1)

        return _TemporalAttentionClassifier()


def predict_with_mc_dropout(model, batch, passes: int = 30):
    torch, _, _ = _require_torch()
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(batch)
            preds.append(torch.sigmoid(logits).unsqueeze(0))
    stacked = torch.cat(preds, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0)
