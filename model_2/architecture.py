import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    """
    Standard ResNet Basic Block for 1D signal data.
    Consists of two Conv1d layers with batch normalization and ReLU activations.
    Downsampling is performed using stride in the first convolution and the shortcut projection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # First convolutional layer in the block (handles downsampling via stride)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Second convolutional layer (maintains channel counts and length)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut pathway: projects channels and performs downsampling if dimensions mismatch
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        return out


class GlobalViewBranch(nn.Module):
    """
    Global View Pipeline:
    - 5 residual conv 1D with progressive filter counts: 16, 32, 64, 128, 256 and stride 2.
    - An 8-head multi-head self-attention layer operating over the temporal dimension of the resulting feature map.
    """
    def __init__(self, in_channels=1, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 5 Residual Conv 1D blocks with progressive filter counts
        self.blocks = nn.Sequential(
            ResidualBlock1D(in_channels, 16, stride=2),
            ResidualBlock1D(16, 32, stride=2),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 256, stride=2)
        )
        
        # 8-head Multi-Head Self-Attention operating over the temporal sequence
        self.mha = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(256)
        
    def forward(self, x):
        # Input shape: (B, 1, 1001) or (B, in_channels, sequence_length)
        features = self.blocks(x)
        # Shape after blocks: (B, 256, L_pooled) where L_pooled = 32
        
        # Permute to (B, L_pooled, channels) for sequence-based attention -> (B, 32, 256)
        features_perm = features.permute(0, 2, 1)
        
        # Self-attention pass
        attn_out, attn_weights = self.mha(features_perm, features_perm, features_perm)
        
        # Residual addition + Layer normalization
        out = self.layer_norm(features_perm + attn_out)
        
        return out, attn_weights


class LocalViewBranch(nn.Module):
    """
    Local View Pipeline:
    - 5 residual conv 1D with progressive filter counts: 16, 32, 64, 128, 256 and stride 2.
    - An 8-head multi-head self-attention layer operating over the temporal dimension of the resulting feature map.
    """
    def __init__(self, in_channels=1, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 5 Residual Conv 1D blocks with progressive filter counts
        self.blocks = nn.Sequential(
            ResidualBlock1D(in_channels, 16, stride=2),
            ResidualBlock1D(16, 32, stride=2),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 256, stride=2)
        )
        
        # 8-head Multi-Head Self-Attention operating over the temporal sequence
        self.mha = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(256)
        
    def forward(self, x):
        # Input shape: (B, 1, 1001) or (B, in_channels, sequence_length)
        features = self.blocks(x)
        # Shape after blocks: (B, 256, L_pooled) where L_pooled = 32
        
        # Permute to (B, L_pooled, channels) for sequence-based attention -> (B, 32, 256)
        features_perm = features.permute(0, 2, 1)
        
        # Self-attention pass
        attn_out, attn_weights = self.mha(features_perm, features_perm, features_perm)
        
        # Residual addition + Layer normalization
        out = self.layer_norm(features_perm + attn_out)
        
        return out, attn_weights


class StellarFeaturesBranch(nn.Module):
    """
    Stellar Features Pipeline:
    - Two fully connected layers (64 then 128 units) with LayerNorm and GELU activations.
    - Maps the 8-dimensional stellar feature vector to a 128-dimensional embedding.
    """
    def __init__(self, in_features=8):
        super().__init__()
        
        # Layer 1: 8 -> 64 dimensions
        self.fc1 = nn.Linear(in_features, 64)
        self.ln1 = nn.LayerNorm(64)
        self.gelu1 = nn.GELU()
        
        # Layer 2: 64 -> 128 dimensions
        self.fc2 = nn.Linear(64, 128)
        self.ln2 = nn.LayerNorm(128)
        self.gelu2 = nn.GELU()
        
    def forward(self, x):
        # Input shape: (B, 8)
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.gelu1(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.gelu2(out)
        
        # Output shape: (B, 128)
        return out


    
