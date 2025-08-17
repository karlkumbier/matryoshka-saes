import torch
import numpy as np
import json
import argparse
import sys
from typing import Dict, Optional, Union
import os

from tree import Tree, TreeDataset


class HierarchicalDataGenerator:
    """
    Data generator for hierarchical sparse features following the matryoshka-saes structure.
    
    Generates samples as a linear combination of feature vectors. Ground truth features are generated given specified correlation levels and scaling. Hierarchical feature activations are generated according
    to the tree structure defined in `tree_config`.
    """
    
    def __init__(
        self,
        tree_config: Optional[Union[Dict, str]] = None,
        d_model: int = 512,
        feature_correlation: float = 0.05,
        random_seed: Optional[int] = None,
        orthogonal_features: bool = True,
        feature_scale_variation: float = 0.05
    ):

        print("Initializing HierarchicalDataGenerator")
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
        self.d_model = d_model
        self.feature_correlation = feature_correlation
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

# Example usage and testing
if __name__ == "__main__":
    
    # Input configuration    
    config = "/Users/kkumbier/github/matryoshka-saes/tree_params/simple_params.json"
    
    # Create data generator
    with open(config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    generator = HierarchicalDataGenerator(**config)
    print(f"Generator created with {generator.n_features} features")
    
    # Generate data using TreeDataset
    n_samples = 10
    dataset = generator.create_dataset(batch_size=n_samples, num_batches=10)
    x, acts = dataset.__getitem__(0)
    
    print(dataset.true_feats.shape)
    