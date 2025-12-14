"""Hierarchical classification models.

Two-stage approach:
- Stage 1: Detect pattern direction (None/Bearish/Bullish)
- Stage 2: Classify pattern subtype (Normal/Wedge/Pennant)

This separates the easier task (direction) from the harder task (subtype).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.lstm_v2.model import SeqLabelingLSTM
from models.hierarchical_v1.config import (
    CONFIG_STAGE1,
    CONFIG_STAGE2,
)


class HierarchicalClassifier(nn.Module):
    """Hierarchical two-stage classifier.
    
    Stage 1: Predicts direction (None=0, Bearish=1, Bullish=2)
    Stage 2: Predicts subtype (Normal=0, Wedge=1, Pennant=2)
    
    Final prediction combines both stages.
    """
    
    def __init__(
        self,
        stage1_config: Optional[Dict] = None,
        stage2_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.stage1_config = stage1_config or CONFIG_STAGE1
        self.stage2_config = stage2_config or CONFIG_STAGE2
        
        # Stage 1: Direction classifier
        self.stage1 = SeqLabelingLSTM(
            input_size=self.stage1_config["input_size"],
            hidden_size=self.stage1_config["hidden_size"],
            num_layers=self.stage1_config["num_layers"],
            num_classes=3,  # None, Bearish, Bullish
            dropout=self.stage1_config["dropout"],
            bidirectional=self.stage1_config.get("bidirectional", False),
        )
        
        # Stage 2: Subtype classifier
        self.stage2 = SeqLabelingLSTM(
            input_size=self.stage2_config["input_size"],
            hidden_size=self.stage2_config["hidden_size"],
            num_layers=self.stage2_config["num_layers"],
            num_classes=3,  # Normal, Wedge, Pennant
            dropout=self.stage2_config["dropout"],
            bidirectional=self.stage2_config.get("bidirectional", False),
        )
    
    def forward_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for stage 1 only.
        
        Args:
            x: Input tensor (batch, seq_len, 4)
            
        Returns:
            Logits (batch, seq_len, 3) for None/Bearish/Bullish
        """
        return self.stage1(x)
    
    def forward_stage2(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for stage 2 only.
        
        Args:
            x: Input tensor (batch, seq_len, 4)
            
        Returns:
            Logits (batch, seq_len, 3) for Normal/Wedge/Pennant
        """
        return self.stage2(x)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_both: bool = False,
    ) -> torch.Tensor:
        """Full forward pass combining both stages.
        
        Args:
            x: Input tensor (batch, seq_len, 4)
            return_both: If True, return (stage1_logits, stage2_logits)
            
        Returns:
            If return_both: Tuple of (stage1_logits, stage2_logits)
            Otherwise: Combined 7-class logits (batch, seq_len, 7)
        """
        stage1_logits = self.stage1(x)  # (batch, seq_len, 3)
        stage2_logits = self.stage2(x)  # (batch, seq_len, 3)
        
        if return_both:
            return stage1_logits, stage2_logits
        
        # Combine into 7-class predictions
        return self._combine_logits(stage1_logits, stage2_logits)
    
    def _combine_logits(
        self, 
        stage1_logits: torch.Tensor, 
        stage2_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Combine stage 1 and stage 2 logits into 7-class logits.
        
        Uses log-probability addition for proper combination.
        
        Args:
            stage1_logits: (batch, seq_len, 3) - None/Bearish/Bullish
            stage2_logits: (batch, seq_len, 3) - Normal/Wedge/Pennant
            
        Returns:
            combined: (batch, seq_len, 7) - 7-class logits
        """
        batch, seq_len, _ = stage1_logits.shape
        device = stage1_logits.device
        
        # Convert to log probabilities
        stage1_logprob = torch.log_softmax(stage1_logits, dim=-1)  # (B, S, 3)
        stage2_logprob = torch.log_softmax(stage2_logits, dim=-1)  # (B, S, 3)
        
        # Build 7-class log probabilities
        combined = torch.zeros(batch, seq_len, 7, device=device)
        
        # Class 0: None = P(stage1=0)
        combined[:, :, 0] = stage1_logprob[:, :, 0]
        
        # Class 1: Bearish Normal = P(stage1=1) * P(stage2=0)
        combined[:, :, 1] = stage1_logprob[:, :, 1] + stage2_logprob[:, :, 0]
        
        # Class 2: Bearish Wedge = P(stage1=1) * P(stage2=1)
        combined[:, :, 2] = stage1_logprob[:, :, 1] + stage2_logprob[:, :, 1]
        
        # Class 3: Bearish Pennant = P(stage1=1) * P(stage2=2)
        combined[:, :, 3] = stage1_logprob[:, :, 1] + stage2_logprob[:, :, 2]
        
        # Class 4: Bullish Normal = P(stage1=2) * P(stage2=0)
        combined[:, :, 4] = stage1_logprob[:, :, 2] + stage2_logprob[:, :, 0]
        
        # Class 5: Bullish Wedge = P(stage1=2) * P(stage2=1)
        combined[:, :, 5] = stage1_logprob[:, :, 2] + stage2_logprob[:, :, 1]
        
        # Class 6: Bullish Pennant = P(stage1=2) * P(stage2=2)
        combined[:, :, 6] = stage1_logprob[:, :, 2] + stage2_logprob[:, :, 2]
        
        return combined
    
    @torch.no_grad()
    def predict(
        self, 
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate predictions from both stages.
        
        Args:
            x: Input tensor (batch, seq_len, 4)
            
        Returns:
            Tuple of:
            - stage1_preds: (batch, seq_len) - 0/1/2
            - stage2_preds: (batch, seq_len) - 0/1/2
            - final_preds: (batch, seq_len) - 0-6
        """
        self.eval()
        
        stage1_logits, stage2_logits = self.forward(x, return_both=True)
        
        stage1_preds = stage1_logits.argmax(dim=-1)
        stage2_preds = stage2_logits.argmax(dim=-1)
        
        # Combine predictions
        combined_logits = self._combine_logits(stage1_logits, stage2_logits)
        final_preds = combined_logits.argmax(dim=-1)
        
        return stage1_preds, stage2_preds, final_preds


def convert_labels_to_stage1(Y: np.ndarray) -> np.ndarray:
    """Convert 7-class labels to Stage 1 (3-class direction).
    
    0 -> 0 (None)
    1,2,3 -> 1 (Bearish)
    4,5,6 -> 2 (Bullish)
    """
    Y_stage1 = np.zeros_like(Y)
    Y_stage1[(Y >= 1) & (Y <= 3)] = 1  # Bearish
    Y_stage1[(Y >= 4) & (Y <= 6)] = 2  # Bullish
    return Y_stage1


def convert_labels_to_stage2(Y: np.ndarray) -> np.ndarray:
    """Convert 7-class labels to Stage 2 (3-class subtype).
    
    Only meaningful for pattern regions (Y > 0).
    
    1,4 -> 0 (Normal)
    2,5 -> 1 (Wedge)
    3,6 -> 2 (Pennant)
    """
    Y_stage2 = np.zeros_like(Y)
    Y_stage2[(Y == 1) | (Y == 4)] = 0  # Normal
    Y_stage2[(Y == 2) | (Y == 5)] = 1  # Wedge
    Y_stage2[(Y == 3) | (Y == 6)] = 2  # Pennant
    return Y_stage2


def get_pattern_mask(Y: np.ndarray) -> np.ndarray:
    """Get boolean mask for pattern regions (Y > 0).
    
    Used to train Stage 2 only on pattern regions.
    """
    return Y > 0
