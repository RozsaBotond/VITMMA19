"""
Data Augmentation Module

Augments time series segment data with various noise and transformation techniques.
Primarily focuses on adding zero-mean Gaussian noise with parameterized amplitude.
"""

import numpy as np
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    noise_amplitude: float = 0.01  # Noise amplitude as fraction of price range
    seed: int = 42  # Random seed for reproducibility
    num_augmented_copies: int = 1  # Number of augmented copies per original sample


def add_gaussian_noise(
    segment: np.ndarray,
    amplitude: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to a segment.
    
    Args:
        segment: OHLC numpy array with shape (n_timesteps, 4).
        amplitude: Noise amplitude as fraction of the segment's price range.
        rng: NumPy random generator for reproducibility.
        
    Returns:
        Augmented segment with added noise.
    """
    if segment.size == 0:
        return segment.copy()
    
    # Calculate the price range for scaling the noise
    price_range = np.max(segment) - np.min(segment)
    if price_range == 0:
        price_range = 1.0  # Avoid division by zero
    
    # Generate zero-mean Gaussian noise
    noise = rng.normal(loc=0.0, scale=amplitude * price_range, size=segment.shape)
    
    # Add noise to the segment
    augmented = segment + noise
    
    # Ensure OHLC consistency: high >= open, close, low and low <= open, close, high
    # After adding noise, we need to fix any inconsistencies
    for i in range(len(augmented)):
        o, h, l, c = augmented[i]
        # Ensure high is the maximum and low is the minimum
        new_high = max(o, h, c)
        new_low = min(o, l, c)
        augmented[i, 1] = new_high  # high
        augmented[i, 2] = new_low   # low
    
    return augmented


def augment_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    config: AugmentationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset by adding noise to all samples.
    
    Args:
        data: Object array of OHLC segments.
        labels: Array of label strings.
        config: Augmentation configuration.
        
    Returns:
        Tuple of (augmented_data, augmented_labels) including original data.
    """
    rng = np.random.default_rng(config.seed)
    
    augmented_data_list = []
    augmented_labels_list = []
    
    # First, include all original data
    for segment, label in zip(data, labels):
        augmented_data_list.append(segment.copy())
        augmented_labels_list.append(label)
    
    # Then, create augmented copies
    for copy_idx in range(config.num_augmented_copies):
        for segment, label in zip(data, labels):
            augmented_segment = add_gaussian_noise(segment, config.noise_amplitude, rng)
            augmented_data_list.append(augmented_segment)
            augmented_labels_list.append(label)
    
    augmented_data = np.array(augmented_data_list, dtype=object)
    augmented_labels = np.array(augmented_labels_list)
    
    return augmented_data, augmented_labels


def create_augmented_dataset(
    input_data_path: str,
    input_labels_path: str,
    output_dir: str,
    noise_amplitude: float = 0.01,
    seed: int = 42,
    num_copies: int = 1,
    prefix: str = 'augmented'
) -> Tuple[str, str]:
    """
    Load data, augment it, and save to output directory.
    
    Args:
        input_data_path: Path to input data numpy file.
        input_labels_path: Path to input labels numpy file.
        output_dir: Output directory for augmented data.
        noise_amplitude: Noise amplitude (fraction of price range).
        seed: Random seed for reproducibility.
        num_copies: Number of augmented copies per original sample.
        prefix: Filename prefix for output files.
        
    Returns:
        Tuple of (data_path, labels_path) for saved files.
    """
    # Load original data
    print(f"Loading data from {input_data_path}...")
    data = np.load(input_data_path, allow_pickle=True)
    labels = np.load(input_labels_path, allow_pickle=True)
    print(f"  Loaded {len(data)} samples")
    
    # Create augmentation config
    config = AugmentationConfig(
        noise_amplitude=noise_amplitude,
        seed=seed,
        num_augmented_copies=num_copies
    )
    
    print(f"\nAugmenting data...")
    print(f"  Noise amplitude: {noise_amplitude}")
    print(f"  Random seed: {seed}")
    print(f"  Augmented copies per sample: {num_copies}")
    
    # Augment data
    augmented_data, augmented_labels = augment_dataset(data, labels, config)
    
    print(f"  Original samples: {len(data)}")
    print(f"  Augmented samples: {len(augmented_data)} (original + {num_copies}x augmented)")
    
    # Save augmented data
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(output_dir, f'{prefix}_data.npy')
    labels_path = os.path.join(output_dir, f'{prefix}_labels.npy')
    
    np.save(data_path, augmented_data)
    np.save(labels_path, augmented_labels)
    
    print(f"\nSaved augmented data:")
    print(f"  Data: {data_path}")
    print(f"  Labels: {labels_path}")
    
    return data_path, labels_path


def augment_train_val_test(
    input_dir: str,
    output_dir: str,
    noise_amplitude: float = 0.01,
    seed: int = 42,
    num_copies: int = 1,
    input_prefix: str = 'bullflag',
    output_prefix: str = 'augmented'
) -> None:
    """
    Augment train, validation, and test sets separately.
    
    Only augments training data to avoid data leakage.
    Validation and test sets are copied without augmentation.
    
    Args:
        input_dir: Directory containing input data files.
        output_dir: Output directory for augmented data.
        noise_amplitude: Noise amplitude (fraction of price range).
        seed: Random seed for reproducibility.
        num_copies: Number of augmented copies per training sample.
        input_prefix: Prefix of input files (e.g., 'bullflag').
        output_prefix: Prefix for output files (e.g., 'augmented').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Augment training data
    train_data_path = os.path.join(input_dir, f'{input_prefix}_train_data.npy')
    train_labels_path = os.path.join(input_dir, f'{input_prefix}_train_labels.npy')
    
    if os.path.exists(train_data_path):
        print("=" * 60)
        print("AUGMENTING TRAINING DATA")
        print("=" * 60)
        create_augmented_dataset(
            train_data_path,
            train_labels_path,
            output_dir,
            noise_amplitude=noise_amplitude,
            seed=seed,
            num_copies=num_copies,
            prefix=f'{output_prefix}_train'
        )
    
    # Copy validation data without augmentation
    val_data_path = os.path.join(input_dir, f'{input_prefix}_val_data.npy')
    val_labels_path = os.path.join(input_dir, f'{input_prefix}_val_labels.npy')
    
    if os.path.exists(val_data_path):
        print("\n" + "=" * 60)
        print("COPYING VALIDATION DATA (no augmentation)")
        print("=" * 60)
        val_data = np.load(val_data_path, allow_pickle=True)
        val_labels = np.load(val_labels_path, allow_pickle=True)
        
        np.save(os.path.join(output_dir, f'{output_prefix}_val_data.npy'), val_data)
        np.save(os.path.join(output_dir, f'{output_prefix}_val_labels.npy'), val_labels)
        print(f"  Copied {len(val_data)} validation samples")
    
    # Copy test data without augmentation
    test_data_path = os.path.join(input_dir, f'{input_prefix}_test_data.npy')
    test_labels_path = os.path.join(input_dir, f'{input_prefix}_test_labels.npy')
    
    if os.path.exists(test_data_path):
        print("\n" + "=" * 60)
        print("COPYING TEST DATA (no augmentation)")
        print("=" * 60)
        test_data = np.load(test_data_path, allow_pickle=True)
        test_labels = np.load(test_labels_path, allow_pickle=True)
        
        np.save(os.path.join(output_dir, f'{output_prefix}_test_data.npy'), test_data)
        np.save(os.path.join(output_dir, f'{output_prefix}_test_labels.npy'), test_labels)
        print(f"  Copied {len(test_data)} test samples")
    
    # Also copy/augment the full dataset
    full_data_path = os.path.join(input_dir, f'{input_prefix}_data.npy')
    full_labels_path = os.path.join(input_dir, f'{input_prefix}_labels.npy')
    
    if os.path.exists(full_data_path):
        print("\n" + "=" * 60)
        print("AUGMENTING FULL DATASET")
        print("=" * 60)
        create_augmented_dataset(
            full_data_path,
            full_labels_path,
            output_dir,
            noise_amplitude=noise_amplitude,
            seed=seed,
            num_copies=num_copies,
            prefix=output_prefix
        )


if __name__ == '__main__':
    import sys
    
    # Default values
    input_dir = 'processed'
    output_dir = 'processed_augmented'
    noise_amplitude = 0.01
    seed = 42
    num_copies = 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        noise_amplitude = float(sys.argv[3])
    if len(sys.argv) > 4:
        seed = int(sys.argv[4])
    if len(sys.argv) > 5:
        num_copies = int(sys.argv[5])
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Noise amplitude: {noise_amplitude}")
    print(f"Random seed: {seed}")
    print(f"Augmented copies: {num_copies}")
    print()
    
    augment_train_val_test(
        input_dir=input_dir,
        output_dir=output_dir,
        noise_amplitude=noise_amplitude,
        seed=seed,
        num_copies=num_copies
    )
