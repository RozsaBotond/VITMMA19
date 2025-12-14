"""Configuration for hierarchical models.

Hierarchical Classification:
- Stage 1: 3-class (None, Bearish, Bullish)
- Stage 2: 3-class subtype (Normal, Wedge, Pennant)

Final output: 7 classes

Key improvements:
- Bidirectional LSTM for better context
- Larger Stage 2 to handle harder subtype classification
- Focal loss option to handle class imbalance
- More aggressive label smoothing for Stage 2
"""

# Stage 1: Direction classifier (None vs Bearish vs Bullish)
CONFIG_STAGE1 = {
    "name": "hierarchical_stage1",
    "description": "Stage 1: Direction classifier (None/Bearish/Bullish)",
    
    # Model architecture
    "input_size": 4,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 3,  # None, Bearish, Bullish
    "dropout": 0.3,
    "bidirectional": True,  # IMPROVED: bidirectional for better context
    
    # Training
    "learning_rate": 0.0003,  # IMPROVED: slightly lower for stability
    "batch_size": 16,
    "epochs": 300,
    "weight_decay": 1e-4,  # IMPROVED: stronger regularization
    "use_class_weights": True,
    "max_class_weight": 5.0,  # IMPROVED: cap class weights
    "patience": 50,
    "min_delta": 0.0001,
    
    # Scheduler
    "scheduler": "cosine",
    "warmup_epochs": 10,  # IMPROVED: longer warmup
    
    # Loss
    "label_smoothing": 0.1,  # IMPROVED: more smoothing
    "focal_loss": False,
    "focal_gamma": 2.0,
}

# Stage 2: Subtype classifier (Normal vs Wedge vs Pennant)
CONFIG_STAGE2 = {
    "name": "hierarchical_stage2",
    "description": "Stage 2: Subtype classifier (Normal/Wedge/Pennant)",
    
    # Model architecture - IMPROVED: larger model for harder task
    "input_size": 4,
    "hidden_size": 128,  # IMPROVED: same size as Stage 1
    "num_layers": 2,
    "num_classes": 3,  # Normal, Wedge, Pennant
    "dropout": 0.4,  # IMPROVED: higher dropout for small data
    "bidirectional": True,  # IMPROVED: bidirectional
    
    # Training
    "learning_rate": 0.0003,  # IMPROVED: lower for stability
    "batch_size": 16,
    "epochs": 300,  # IMPROVED: more epochs
    "weight_decay": 1e-4,  # IMPROVED: stronger regularization
    "use_class_weights": True,
    "max_class_weight": 3.0,  # IMPROVED: lower cap
    "patience": 50,  # IMPROVED: more patience
    "min_delta": 0.0001,
    
    # Scheduler
    "scheduler": "cosine",
    "warmup_epochs": 10,  # IMPROVED: longer warmup
    
    # Loss
    "label_smoothing": 0.15,  # IMPROVED: more smoothing for noisy labels
    "focal_loss": True,  # IMPROVED: focal loss for imbalance
    "focal_gamma": 2.0,
}

# Class name mappings
STAGE1_CLASSES = ["None", "Bearish", "Bullish"]
STAGE2_CLASSES = ["Normal", "Wedge", "Pennant"]

# Mapping from 7-class to hierarchical
# 0: None -> Stage1=0 (None)
# 1: Bearish Normal -> Stage1=1 (Bearish), Stage2=0 (Normal)
# 2: Bearish Wedge -> Stage1=1 (Bearish), Stage2=1 (Wedge)
# 3: Bearish Pennant -> Stage1=1 (Bearish), Stage2=2 (Pennant)
# 4: Bullish Normal -> Stage1=2 (Bullish), Stage2=0 (Normal)
# 5: Bullish Wedge -> Stage1=2 (Bullish), Stage2=1 (Wedge)
# 6: Bullish Pennant -> Stage1=2 (Bullish), Stage2=2 (Pennant)

LABEL_7CLASS_TO_STAGE1 = {
    0: 0,  # None -> None
    1: 1,  # Bearish Normal -> Bearish
    2: 1,  # Bearish Wedge -> Bearish
    3: 1,  # Bearish Pennant -> Bearish
    4: 2,  # Bullish Normal -> Bullish
    5: 2,  # Bullish Wedge -> Bullish
    6: 2,  # Bullish Pennant -> Bullish
}

LABEL_7CLASS_TO_STAGE2 = {
    1: 0,  # Bearish Normal -> Normal
    2: 1,  # Bearish Wedge -> Wedge
    3: 2,  # Bearish Pennant -> Pennant
    4: 0,  # Bullish Normal -> Normal
    5: 1,  # Bullish Wedge -> Wedge
    6: 2,  # Bullish Pennant -> Pennant
}

# Inverse mapping: (Stage1, Stage2) -> 7-class
HIERARCHICAL_TO_7CLASS = {
    (0, None): 0,  # None
    (1, 0): 1,     # Bearish Normal
    (1, 1): 2,     # Bearish Wedge
    (1, 2): 3,     # Bearish Pennant
    (2, 0): 4,     # Bullish Normal
    (2, 1): 5,     # Bullish Wedge
    (2, 2): 6,     # Bullish Pennant
}
