import torch
import numpy as np
import json
import argparse
import sys
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
import os

from toy_model import Tree, TreeDataset


class HierarchicalDataGenerator:
    """
    Data generator for hierarchical sparse features following the matryoshka-saes structure.
    
    This generator creates synthetic data that follows the hierarchical tree structure
    used in the matryoshka SAE experiments, where features have dependencies and
    different activation probabilities based on their position in the hierarchy.
    """
    
    def __init__(
        self,
        tree_config: Optional[Union[Dict, str]] = None,
        d_model: int = 512,
        feature_correlation: float = 0.05,
        noise_level: float = 0.0,
        random_seed: Optional[int] = None,
        orthogonal_features: bool = True,
        feature_scale_variation: float = 0.05
    ):

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
        self.d_model = d_model
        self.feature_correlation = feature_correlation
        self.noise_level = noise_level
        self.orthogonal_features = orthogonal_features
        self.feature_scale_variation = feature_scale_variation
        
        # Tree config must be provided
        if tree_config is None:
            raise ValueError("tree_config must be provided. Use HierarchicalDataGenerator.from_json() or provide a tree configuration dictionary.")
        
        # If tree_config is a string, treat it as a path and load the JSON
        if isinstance(tree_config, str):
            with open(tree_config, 'r', encoding='utf-8') as f:
                tree_config = json.load(f)
        
        self.tree = Tree(tree_config)
        self.n_features = self.tree.n_features
        
        # Generate feature directions
        self.true_feats = self._generate_feature_directions()
    
    def create_dataset(
        self,
        batch_size: int = 200,
        num_batches: int = 1000,
        device: str = "cpu"
    ) -> TreeDataset:
        """Create a TreeDataset compatible with the training loop."""
        return TreeDataset(
            tree=self.tree,
            true_feats=self.true_feats.to(device),
            batch_size=batch_size,
            num_batches=num_batches
        )

    def _generate_feature_directions(self) -> torch.Tensor:
        """Generate feature directions with optional correlation and scaling."""
        if self.orthogonal_features and self.n_features <= self.d_model:
            Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.d_model))
            true_feats = Q[:self.n_features].T
        else:
            true_feats = torch.randn(self.d_model, self.n_features)
            true_feats = torch.nn.functional.normalize(true_feats, dim=0)
        if self.feature_correlation > 0:
            correlation_matrix = torch.eye(self.n_features) + \
                self.feature_correlation * torch.randn(self.n_features, self.n_features)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            true_feats = true_feats @ correlation_matrix
        if self.feature_scale_variation > 0:
            random_scaling = 1 + torch.randn(self.n_features) * self.feature_scale_variation
            true_feats = true_feats * random_scaling[None, :]
        
        return true_feats.T
        
    def get_ground_truth_l0(self, n_samples: int = 10000) -> float:
        """Calculate the expected L0 sparsity of the ground truth features."""
        sample = self.tree.sample(n_samples)
        if isinstance(sample, list):
            sample = torch.tensor(sample, dtype=torch.float32)
        return float((sample > 0).float().mean())
        
    def get_feature_statistics(self, n_samples: int = 10000) -> Dict:
        """Get statistics about feature activations."""
        samples = self.tree.sample(n_samples)
        if isinstance(samples, list):
            samples = torch.tensor(samples, dtype=torch.float32)
        feature_stats = {
            "mean_l0": float((samples > 0).float().mean()),
            "feature_frequencies": (samples > 0).float().mean(dim=0).tolist(),
            "feature_means": samples.mean(dim=0).tolist(),
            "feature_stds": samples.std(dim=0).tolist(),
            "total_features": self.n_features,
            "d_model": self.d_model
        }
        return feature_stats
        
    def get_tree_config(self) -> Dict:
        """Get the tree configuration as a dictionary."""
        return self._tree_to_dict(self.tree)
        
    def _tree_to_dict(self, tree: Tree) -> Dict:
        """Convert Tree object back to dictionary format."""
        result = {
            "active_prob": tree.active_prob,
            "is_read_out": tree.is_read_out,
            "mutually_exclusive_children": tree.mutually_exclusive_children,
        }
        if hasattr(tree, 'is_binary'):
            result["is_binary"] = tree.is_binary
        if tree.children:
            result["children"] = [self._tree_to_dict(child) for child in tree.children]
        return result
        
    def save_features(self, path: str):
        """Save the generated feature directions to a file."""
        torch.save({
            'true_feats': self.true_feats,
            'd_model': self.d_model,
            'n_features': self.n_features
        }, path)
        
    @classmethod
    def load_features(cls, path: str, tree_config: Optional[Dict] = None, **kwargs):
        """Load a generator with pre-saved feature directions."""
        saved_data = torch.load(path)
        generator = cls(
            tree_config=tree_config,
            d_model=saved_data['d_model'],
            **kwargs
        )
        generator.true_feats = saved_data['true_feats']
        return generator


def create_custom_tree_config(
    n_levels: int = 3,
    branching_factor: int = 3,
    root_prob: float = 0.1,
    decay_factor: float = 0.7,
    mutually_exclusive_prob: float = 0.3
) -> Dict:
    """
    Create a custom tree configuration with specified parameters.
    
    Args:
        n_levels: Number of hierarchy levels
        branching_factor: Average number of children per node
        root_prob: Activation probability for root nodes
        decay_factor: How much activation probability decreases per level
        mutually_exclusive_prob: Probability that children are mutually exclusive
        
    Returns:
        Tree configuration dictionary
    """
    def create_node(level: int, is_root: bool = False) -> Dict:
        # Calculate activation probability based on level
        if is_root:
            prob = 1.0
        else:
            prob = root_prob * (decay_factor ** level)
            
        node = {
            "active_prob": prob,
            "is_read_out": not is_root,
            "children": []
        }
        
        # Add mutually exclusive flag randomly
        if not is_root and np.random.random() < mutually_exclusive_prob:
            node["mutually_exclusive_children"] = True
            
        # Create children if not at max level
        if level < n_levels:
            n_children = np.random.poisson(branching_factor)
            n_children = max(0, min(n_children, 8))  # Cap at reasonable number
            
            for _ in range(n_children):
                child = create_node(level + 1)
                node["children"].append(child)
                
        return node
    
    return create_node(0, is_root=True)


# Example usage and testing
if __name__ == "__main__":
    
    # Input configuration    
    parser = argparse.ArgumentParser(description="Hierarchical Data Generator")
    parser.add_argument("--config", type=str, help="Path to parameter configuration file")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()
    
    
    # Validate that we have a tree configuration
    assert args.config, "Error: Must provide either --config"
    print("Use --help for more information")

    # Create data generator
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    generator = HierarchicalDataGenerator(**config)
    print(f"Generator created with {generator.n_features} features")
    print(f"Ground truth L0: {generator.get_ground_truth_l0():.4f}")
    
    # Generate data using TreeDataset
    dataset = generator.create_dataset(batch_size=args.n_samples, num_batches=1)
    data = dataset[0]
    
    # For activations, sample directly from the tree
    activations = generator.tree.sample(args.n_samples)
    if isinstance(activations, list):
        activations = torch.tensor(activations, dtype=torch.float32)
    
    print(f"Generated data shape: {data.shape}")
    print(f"Activations shape: {activations.shape}")
    print(f"Actual L0 in batch: {(activations > 0).float().mean():.4f}")