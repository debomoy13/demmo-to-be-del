import torch
import torch.nn as nn

try:
    from model_2.architecture import GlobalViewBranch, LocalViewBranch, StellarFeaturesBranch
except ImportError:
    from architecture import GlobalViewBranch, LocalViewBranch, StellarFeaturesBranch

class ResidualLateFusionHead(nn.Module):
    """
    Residual Late Fusion Head:
    - Receives a 640-dimensional concatenated joint representation of:
      - Global view pooled embedding (256-d)
      - Local view pooled embedding (256-d)
      - Stellar parameter embedding (128-d)
    - Refines this representation using a 2-layer MLP (512 then 256 units, dropout p = 0.4).
    - Utilizes a linear shortcut R640 -> 256 for residual learning.
    """
    def __init__(self, in_features=640, hidden_features=512, out_features=256, dropout=0.4):
        super().__init__()
        
        # 2-layer MLP Pathway
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(dropout)
        
        # Linear Shortcut: R640 -> 256 units
        self.shortcut = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # Shortcut pathway
        identity = self.shortcut(x)
        
        # Main MLP pathway
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Merge residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


class ClassifierHead(nn.Module):
    """
    Classifier Head:
    - FC 256 -> 64 -> num_classes (default 5 classes, representing:
      0: transit, 1: stellar_eclipse, 2: not_transit, 3: centroid_offset, 4: Ephemeris match)
    """
    def __init__(self, in_features=256, hidden_features=64, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ExoplanetLateFusionModel(nn.Module):
    """
    Unified Exoplanet Model combining:
    1. Global View Pipeline (GlobalViewBranch)
    2. Local View Pipeline (LocalViewBranch)
    3. Stellar Features Pipeline (StellarFeaturesBranch)
    4. Late Fusion & Residual Refinement (ResidualLateFusionHead)
    5. Multi-task Heads:
       - Classification (ClassifierHead with 5 classes)
       - Regression (4 parameters: depth, duration, period, epoch)
       - Confidence (1 transit probability parameter)
    """
    def __init__(self, num_classes=5, temperature=1.0):
        super().__init__()
        
        # Instantiate three parallel feature extraction pipelines
        self.global_branch = GlobalViewBranch()
        self.local_branch = LocalViewBranch()
        self.stellar_branch = StellarFeaturesBranch()
        
        # Instantiate late-fusion residual MLP head (640-d input -> 256-d output)
        self.late_fusion_head = ResidualLateFusionHead(
            in_features=640,
            hidden_features=512,
            out_features=256,
            dropout=0.4
        )
        
        # Instantiate multi-class classifier head (256-d input -> num_classes output)
        self.classifier_head = ClassifierHead(
            in_features=256,
            hidden_features=64,
            num_classes=num_classes
        )
        
        # Multi-task Regression Head (256 -> 4 parameters)
        self.reg_head = nn.Linear(256, 4)
        
        # Multi-task Confidence Head (256 -> 1 output logit)
        self.conf_head = nn.Linear(256, 1)
        
        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        
    def forward(self, x_global, x_local, x_stellar):
        # 1. Process Global and Local curves -> outputs are shape (B, 32, 256)
        global_features, global_attn = self.global_branch(x_global)
        local_features, local_attn = self.local_branch(x_local)
        
        # 2. Extract Stellar parameter embedding -> output shape (B, 128)
        stellar_emb = self.stellar_branch(x_stellar)
        
        # 3. Global Average Pooling (GAP) over temporal sequence lengths
        global_emb = torch.mean(global_features, dim=1) # shape (B, 256)
        local_emb = torch.mean(local_features, dim=1)   # shape (B, 256)
        
        # 4. Concatenate embeddings to produce 640-dimensional joint representation
        fused = torch.cat([global_emb, local_emb, stellar_emb], dim=1) # shape (B, 640)
        
        # 5. Pass through late fusion residual head
        refined = self.late_fusion_head(fused) # shape (B, 256)
        
        # 6. Pass through final classifier and multi-task heads
        class_logits = self.classifier_head(refined)    # shape (B, num_classes)
        reg_outputs = self.reg_head(refined)            # shape (B, 4)
        confidence_logits = self.conf_head(refined)     # shape (B, 1)
        confidence = torch.sigmoid(confidence_logits)    # shape (B, 1)
        
        return class_logits, reg_outputs, confidence, global_attn, local_attn
        
    def get_probabilities(self, class_logits):
        """
        Applies temperature scaling and softmax on class logits to get probability distribution.
        """
        scaled_logits = class_logits / self.temperature
        return torch.softmax(scaled_logits, dim=-1)


if __name__ == "__main__":
    print("=" * 60)
    print("MOCK TENSOR FUSION MODEL DIMENSIONALITY CHECKS")
    print("=" * 60)
    
    batch_size = 4
    num_classes = 5
    model = ExoplanetLateFusionModel(num_classes=num_classes)
    model.eval()
    
    x_global = torch.randn(batch_size, 1, 1001)
    x_local = torch.randn(batch_size, 1, 1001)
    x_stellar = torch.randn(batch_size, 8)
    
    class_logits, reg_outputs, confidence, global_attn, local_attn = model(x_global, x_local, x_stellar)
    probs = model.get_probabilities(class_logits)
    
    print(f"Global Input Shape:               {x_global.shape}")
    print(f"Local Input Shape:                {x_local.shape}")
    print(f"Stellar Input Shape:              {x_stellar.shape}")
    print(f"\nClassification Logits Shape:      {class_logits.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Probabilities Output Shape:       {probs.shape} (Expected: [{batch_size}, {num_classes}])")
    print(f"Regression Outputs Shape:         {reg_outputs.shape} (Expected: [{batch_size}, 4])")
    print(f"Confidence Output Shape:          {confidence.shape} (Expected: [{batch_size}, 1])")
    print(f"Global Attention Weights Shape:   {global_attn.shape} (Expected: [{batch_size}, 32, 32])")
    print(f"Local Attention Weights Shape:    {local_attn.shape} (Expected: [{batch_size}, 32, 32])")
    
    assert class_logits.shape == (batch_size, num_classes)
    assert probs.shape == (batch_size, num_classes)
    assert reg_outputs.shape == (batch_size, 4)
    assert confidence.shape == (batch_size, 1)
    assert global_attn.shape == (batch_size, 32, 32)
    assert local_attn.shape == (batch_size, 32, 32)
    
    print("=" * 60)
    print("STATUS: FUSION MODEL VERIFIED SUCCESSFULLY!")
    print("=" * 60)
