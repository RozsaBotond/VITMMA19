"""Statistical Baseline Model Configuration.

This baseline uses handcrafted features (trend, volatility) with a simple
classifier. No neural network - pure statistical/ML approach.

This serves as the reference to beat with deep learning models.
"""

# Model identification
MODEL_NAME = "Statistical Baseline"
MODEL_VERSION = "1.0"
MODEL_DESCRIPTION = "Trend + Volatility features with Logistic Regression/Random Forest"

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
# Features extracted from OHLC data:
# 1. Trend direction (price change from start to end)
# 2. Trend strength (magnitude of price change)
# 3. Volatility (std of returns)
# 4. Range ratio (high-low range relative to close)
# 5. Body ratio (open-close relative to high-low)
# 6. Upper/lower wick ratios

FEATURE_NAMES = [
    "trend_direction",      # 1 if bullish, 0 if bearish
    "trend_strength",       # Normalized price change
    "volatility",           # Std of log returns
    "avg_range_ratio",      # Average (high-low)/close
    "avg_body_ratio",       # Average |open-close|/(high-low)
    "consolidation_ratio",  # Ratio of consolidation phase to trend
    "slope_first_half",     # Trend in first half (pole)
    "slope_second_half",    # Trend in second half (flag)
]

# =============================================================================
# CLASSIFIER
# =============================================================================
CLASSIFIER = "logistic_regression"  # Options: "logistic_regression", "random_forest"

# Logistic Regression params
LR_C = 1.0
LR_MAX_ITER = 1000

# Random Forest params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def get_config() -> dict:
    """Return all configuration as a dictionary."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "classifier": CLASSIFIER,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
    }
