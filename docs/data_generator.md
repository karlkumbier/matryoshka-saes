# Hierarchical Data Generator Model Summary

## Overview

The `HierarchicalDataGenerator` implements a data generating process (DGP) that creates synthetic data with hierarchical sparse feature structure. This model is designed to test Sparse Autoencoders (SAEs) on data where features have meaningful hierarchical dependencies, mimicking real-world scenarios where concepts build upon each other.

## Data Generating Process (DGP)

### 1. Hierarchical Feature Structure

The core of the DGP is a **tree structure** that defines feature dependencies. Each node in the tree represents a feature or feature group with:

- **Activation probability**: Probability of becoming active given parent is active
- **Read-out status**: Whether the node contributes to the final feature vector
- **Mutual exclusivity**: Whether children compete (only one can be active)
- **Hierarchical dependencies**: Children can only activate if parents are active

Example structure:
```
Root (always active)
├── Feature Group A (prob = 0.15, mutually exclusive children)
│   ├── Feature A1 (prob = 0.2)
│   ├── Feature A2 (prob = 0.2)
│   └── Feature A3 (prob = 0.2)
├── Feature Group B (prob = 0.1, hierarchical children)
│   ├── Subgroup B1 (prob = 0.5)
│   │   ├── Feature B1a (prob = 0.8)
│   │   └── Feature B1b (prob = 0.6)
│   └── Subgroup B2 (prob = 0.3)
│       ├── Feature B2a (prob = 0.7)
│       └── Feature B2b (prob = 0.4)
└── Feature Group C (prob = 0.05, independent children)
    ├── Feature C1 (prob = 0.3)
    ├── Feature C2 (prob = 0.4)
    └── Feature C3 (prob = 0.2)
```

### 2. Feature Generation

#### 2.1 Activation Sampling
For each data sample, the model samples binary/continuous activations following the tree structure:

1. **Root-to-Leaf Propagation**: Starting from the root, each node's activation is sampled based on:
   - Its intrinsic activation probability
   - Whether its parent is active
   - Mutual exclusivity constraints (if applicable)

2. **Activation Rules**:
   - If a parent is inactive, all children are forced inactive
   - For mutually exclusive children, only one child can be active when the parent is active
   - For independent children, each child's activation is sampled independently given the parent is active

3. **Binary vs Continuous**: Features can be binary (0 or 1) or continuous (0 to 1 when active)

#### 2.2 Direction Generation
The model generates `d_model`-dimensional feature directions that represent how each latent feature manifests in the observed data:

1. **Orthogonal Initialization** (default): Uses QR decomposition to create orthogonal unit vectors
2. **Random Initialization**: Generates random normalized directions when orthogonality is disabled
3. **Correlation Injection**: Adds controlled correlations between feature directions
4. **Scale Variation**: Applies random scaling to feature magnitudes

### 3. Data Vector Construction

Final data vectors are generated as:

```
x = Σᵢ aᵢ × fᵢ + ε
```

Where:
- `aᵢ` is the activation of feature i (from tree sampling)
- `fᵢ` is the feature direction for feature i
- `ε` is optional Gaussian noise

## Parameter Guide

### 1. Hierarchical Feature Structure Parameters

| Parameter | Default | Description | DGP Impact |
|-----------|---------|-------------|------------|
| `tree_config` | **REQUIRED** | Hierarchical structure specification | Defines feature dependencies and activation probabilities |

#### Tree Configuration Format
The tree structure is defined by nested dictionaries with these keys:

| Key | Type | Description | DGP Impact |
|-----|------|-------------|------------|
| `active_prob` | float [0,1] | Probability of activation given parent is active | Controls feature sparsity at each level |
| `is_read_out` | bool | Whether this node contributes to final feature vector | Determines if node generates observable features |
| `mutually_exclusive_children` | bool | Whether only one child can be active | Creates competition between features |
| `children` | list | Child nodes in the hierarchy | Defines the tree structure depth and branching |

### 2. Feature Generation Parameters

| Parameter | Default | Description | DGP Impact |
|-----------|---------|-------------|------------|
| `orthogonal_features` | True | Generate orthogonal feature directions | True: Features are linearly independent<br>False: Features may be correlated in representation |
| `feature_correlation` | 0.05 | Amount of correlation between feature directions | Higher values create more entangled representations |
| `feature_scale_variation` | 0.05 | Random variation in feature magnitudes | Controls heterogeneity in feature importance |

### 3. Data Vector Construction Parameters

| Parameter | Default | Description | DGP Impact |
|-----------|---------|-------------|------------|
| `d_model` | 512 | Dimensionality of data vectors | Controls the ambient space size; affects feature identifiability |
| `noise_level` | 0.0 | Gaussian noise standard deviation | Higher values make feature recovery harder |
| `random_seed` | None | Random seed for reproducibility | Ensures deterministic generation |

## Example Tree Configurations

### 1. Simple Two-Level Hierarchy

**Configuration file** (`simple_hierarchy.json`):
```json
{
  "active_prob": 1.0,
  "is_read_out": false,
  "children": [
    {
      "active_prob": 0.2,
      "is_read_out": true,
      "children": [
        {"active_prob": 0.8, "is_read_out": true},
        {"active_prob": 0.6, "is_read_out": true}
      ]
    }
  ]
}
```

**Complete parameter file** (`simple_params.json`):
```json
{
  "tree_config_file": "simple_hierarchy.json",
  "d_model": 256,
  "feature_correlation": 0.0,
  "noise_level": 0.0,
  "orthogonal_features": true,
  "feature_scale_variation": 0.05,
  "random_seed": 42
}
```

**Usage commands**:
```bash
# Generate data using parameter file
python data_generator.py --config simple_params.json --n_samples 1000 --output simple_data.pt

# Generate data with command line overrides
python data_generator.py --config simple_params.json --d_model 512 --noise_level 0.01 --n_samples 2000
```

**Effect**: Creates 3 features where 2 sub-features activate only when the parent feature is active.

### 2. Mutually Exclusive Groups

**Configuration file** (`exclusive_groups.json`):
```json
{
  "active_prob": 1.0,
  "is_read_out": false,
  "children": [
    {
      "active_prob": 0.3,
      "is_read_out": true,
      "mutually_exclusive_children": true,
      "children": [
        {"active_prob": 0.4, "is_read_out": true},
        {"active_prob": 0.6, "is_read_out": true}
      ]
    }
  ]
}
```

**Complete parameter file** (`exclusive_params.json`):
```json
{
  "tree_config_file": "exclusive_groups.json",
  "d_model": 128,
  "feature_correlation": 0.1,
  "noise_level": 0.02,
  "orthogonal_features": true,
  "feature_scale_variation": 0.1,
  "random_seed": 123
}
```

**Usage commands**:
```bash
# Generate data and save statistics
python data_generator.py --config exclusive_params.json --n_samples 5000 --output exclusive_data.pt --save_stats

# Generate multiple datasets with different seeds
for seed in 42 123 456; do
    python data_generator.py --config exclusive_params.json --random_seed $seed --output exclusive_data_${seed}.pt
done
```

**Effect**: Creates competition where only one child feature can be active at a time.

### 3. Repository Default (tree.json)

**Usage commands**:
```bash
# Use existing tree.json directly
python data_generator.py --tree_config tree.json --d_model 512 --n_samples 10000

# Create parameter file for tree.json
python data_generator.py --create_config tree_params.json --tree_config tree.json --d_model 512

# Generate data using the created parameter file
python data_generator.py --config tree_params.json --n_samples 10000 --output tree_data.pt
```

## Custom Tree Generation

The `create_custom_tree_config()` function allows programmatic tree creation:

```python
from data_generator import create_custom_tree_config, HierarchicalDataGenerator
import json

# Create custom tree structure
tree = create_custom_tree_config(
    n_levels=3,           # Depth of hierarchy
    branching_factor=2,   # Average children per node
    root_prob=0.1,        # Base activation probability
    decay_factor=0.7,     # How much probability decreases per level
    mutually_exclusive_prob=0.3  # Chance of mutual exclusivity
)

# Save tree configuration
with open('custom_tree.json', 'w') as f:
    json.dump(tree, f, indent=2)

# Create complete parameter file
params = {
    "tree_config_file": "custom_tree.json",
    "d_model": 256,
    "feature_correlation": 0.05,
    "noise_level": 0.01,
    "orthogonal_features": True,
    "feature_scale_variation": 0.05,
    "random_seed": 42
}

with open('custom_params.json', 'w') as f:
    json.dump(params, f, indent=2)

# Generate data
generator = HierarchicalDataGenerator.from_config('custom_params.json')
data, activations = generator.generate_batch(1000, return_activations=True)
```

**Command line usage**:
```bash
# Generate custom tree and data in one step
python data_generator.py --generate_custom_tree --n_levels 4 --branching_factor 3 --root_prob 0.15 --d_model 512 --n_samples 5000 --output custom_data.pt
```

## Extras

## Extras

### Statistical Properties

#### Expected Sparsity (L0)
The expected number of active features depends on the tree structure:
- Root features: `Σᵢ P(root_i active)`
- Child features: `Σᵢ P(parent_i active) × P(child_i | parent_i active)`

#### Feature Co-occurrence Patterns
- **Hierarchical dependencies**: Child features never activate without parents
- **Mutual exclusivity**: Creates negative correlations between competing features
- **Independent features**: Create sparse, uncorrelated activations

### Usage Patterns

#### For SAE Testing
1. **Easy case**: Low noise, orthogonal features, simple hierarchy
2. **Medium case**: Moderate correlation, deeper hierarchies
3. **Hard case**: High noise, non-orthogonal features, complex mutual exclusivity

#### Parameter Recommendations

| Scenario | `d_model` | `feature_correlation` | `noise_level` | `orthogonal_features` |
|----------|-----------|----------------------|---------------|----------------------|
| **Proof of concept** | 64-128 | 0.0 | 0.0 | True |
| **Realistic testing** | 256-512 | 0.05-0.1 | 0.01-0.05 | True |
| **Challenging evaluation** | 512-1024 | 0.1-0.3 | 0.05-0.1 | False |

### Ground Truth Metrics

The generator provides several metrics for evaluation:

1. **Expected L0**: `get_ground_truth_l0()` - theoretical sparsity
2. **Feature frequencies**: How often each feature activates
3. **Feature statistics**: Mean and standard deviation of activations
4. **Hierarchical structure**: Tree visualization and analysis

### Integration with Matryoshka SAE

The generated data is designed to test the Matryoshka SAE's ability to:

1. **Discover hierarchical structure**: Learn that some features depend on others
2. **Handle sparsity**: Work with very sparse activation patterns
3. **Scale efficiently**: Use prefix-based training to learn features at different scales
4. **Preserve dependencies**: Maintain hierarchical relationships in learned representations

The data generator creates the perfect testbed for these capabilities by providing ground truth hierarchical structure that can be compared against learned representations.
