"""Sequence Labeling LSTM Model.

Outputs per-timestep predictions for each position in the input sequence.
Designed for online inference where each new datapoint gets a prediction.

Architecture:
    Input (batch, seq_len, 4) -> LSTM -> Linear -> Output (batch, seq_len, 7)
    
Unlike window classification, this model produces a label for EVERY timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from .config import CONFIG


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, 
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SeqLabelingLSTM(BaseModel):
    """LSTM for per-timestep sequence labeling.
    
    Args:
        input_size: Number of input features (default: 4 for OHLC)
        hidden_size: LSTM hidden dimension (default: 24)
        num_layers: Number of LSTM layers (default: 2)
        num_classes: Number of output classes (default: 7)
        dropout: Dropout probability (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: False for online)
        seq_len: Sequence length / window size (default: 256)
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        hidden_size: int = CONFIG["hidden_size"],
        num_layers: int = CONFIG["num_layers"],
        num_classes: int = CONFIG["num_classes"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        seq_len: int = 256,
    ):
        # Call BaseModel.__init__ with required args
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "seq_len": seq_len,
        }
        
        self.feature_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Per-timestep classifier
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
        
        # Hidden state for online inference
        self._hidden_state = None
    
    def _init_weights(self):
        """Initialize LSTM and classifier weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        # LSTM outputs at every timestep
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden * directions)
        
        # Classify each timestep
        logits = self.classifier(lstm_out)  # (batch, seq_len, num_classes)
        
        return logits
    
    def forward_online(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        """Online inference - process one timestep at a time.
        
        Args:
            x: Input tensor of shape (batch, 1, input_size) - single timestep
            reset: Whether to reset hidden state
            
        Returns:
            Logits of shape (batch, 1, num_classes)
        """
        batch_size = x.size(0)
        
        if reset or self._hidden_state is None:
            self._hidden_state = self._init_hidden(batch_size, x.device)
        
        # Single timestep forward
        lstm_out, self._hidden_state = self.lstm(x, self._hidden_state)
        logits = self.classifier(lstm_out)
        
        return logits
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """Initialize hidden state for online inference."""
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset hidden state for new sequence."""
        self._hidden_state = None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class for each timestep.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Predicted classes of shape (batch, seq_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities for each timestep.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Probabilities of shape (batch, seq_len, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> dict:
        """Return model configuration dict."""
        return {
            "name": CONFIG["name"],
            "description": CONFIG["description"],
            "input_size": self.feature_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "num_parameters": self.num_parameters,
        }
    
    def get_loss_fn(
        self, 
        class_weights: torch.Tensor | None = None,
        use_focal: bool = CONFIG["focal_loss"],
        focal_gamma: float = CONFIG["focal_gamma"],
        label_smoothing: float = CONFIG["label_smoothing"],
    ) -> nn.Module:
        """Get loss function for training.
        
        Args:
            class_weights: Optional class weights for imbalanced data
            use_focal: Use focal loss instead of cross-entropy
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing for regularization
            
        Returns:
            Loss function module
        """
        if use_focal:
            return FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )


def compute_class_weights(
    y: torch.Tensor, 
    num_classes: int = 7,
    max_weight: float = 10.0,  # Higher cap for imbalanced sequence labeling
    balance_mode: str = "inverse_freq",  # "inverse_freq" or "balanced"
) -> torch.Tensor:
    """Compute class weights for imbalanced sequence labeling.
    
    For highly imbalanced per-timestep labeling (e.g., 90% background, 10% patterns),
    proper class weights are critical to prevent the model from always predicting
    the majority class.
    
    Args:
        y: Labels tensor (can be any shape, will be flattened)
        num_classes: Number of classes
        max_weight: Maximum weight for any class (to avoid instability)
        balance_mode: 
            - "inverse_freq": weight = total / (count * num_present_classes)
            - "balanced": weight = max_count / count (so all classes have equal importance)
        
    Returns:
        Class weights tensor of shape (num_classes,)
    """
    y_flat = y.flatten().long()
    total = len(y_flat)
    counts = torch.bincount(y_flat, minlength=num_classes).float()
    
    # Present classes mask
    present_mask = counts > 0
    
    weights = torch.ones(num_classes)
    
    if present_mask.any():
        if balance_mode == "balanced":
            # Make all classes have equal total weight
            # weight = max_count / count (so rare classes have high weight)
            max_count = counts[present_mask].max()
            weights[present_mask] = max_count / counts[present_mask]
        else:  # inverse_freq
            # Standard inverse frequency weighting
            num_present = present_mask.sum()
            weights[present_mask] = total / (counts[present_mask] * num_present)
    
    # For missing classes, use median weight of present classes
    if (~present_mask).any() and present_mask.any():
        median_weight = weights[present_mask].median()
        weights[~present_mask] = median_weight
    
    # Cap maximum weight to avoid instability
    weights = weights.clamp(max=max_weight)
    
    # Normalize so mean weight = 1 (doesn't change relative importance)
    weights = weights / weights.mean()
    
    return weights


if __name__ == "__main__":
    # Quick test
    model = SeqLabelingLSTM()
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.num_parameters:,}")
    
    # Test forward pass
    x = torch.randn(8, 256, 4)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    
    # Test online inference
    model.reset_hidden()
    for t in range(5):
        x_t = torch.randn(8, 1, 4)
        y_t = model.forward_online(x_t)
        print(f"Online step {t}: input {x_t.shape} -> output {y_t.shape}")
    
    # Test with class weights
    y_labels = torch.randint(0, 7, (8, 256))
    weights = compute_class_weights(y_labels)
    print(f"\nClass weights: {weights}")
    
    loss_fn = model.get_loss_fn(class_weights=weights)
    logits = model(x)
    loss = loss_fn(logits.view(-1, 7), y_labels.view(-1))
    print(f"Loss: {loss.item():.4f}")
