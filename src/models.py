import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

MODEL_URLS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
    "xception": None,
}


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, hidden=64, mid=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, mid, kernel_size=1)
        self.conv3 = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x):
        att = F.relu(self.conv1(x))
        att = F.relu(self.conv2(att))
        att = torch.sigmoid(self.conv3(att))
        return att


class AttResNet50(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, dropout2=0.25, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=MODEL_URLS["resnet50"] if pretrained else None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.attention = SpatialAttention(2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
            nn.Dropout(dropout2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        att = self.attention(x)
        x = x * att
        x = self.gap(x).flatten(1)
        return self.classifier(x), x

    def extract_features(self, x):
        _, feats = self.forward(x)
        return feats


class AttentionModule(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.w_u = nn.Linear(in_dim, hidden)
        self.u_w = nn.Parameter(torch.randn(hidden))
        nn.init.xavier_uniform_(self.w_u.weight)

    def forward(self, h):
        u = F.tanh(self.w_u(h))
        scores = torch.einsum("bh,h->b", u, self.u_w)
        alpha = F.softmax(scores, dim=-1)
        context = alpha.unsqueeze(-1) * h
        return context


class AttCNN(nn.Module):
    def __init__(self, in_dim=500, hidden=256, mid=128, num_classes=1, dropout=0.5, dropout2=0.3):
        super().__init__()
        self.attention = AttentionModule(in_dim, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(mid, num_classes),
        )

    def forward(self, x):
        x = self.attention(x)
        return self.classifier(x)