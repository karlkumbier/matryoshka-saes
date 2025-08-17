import torch
from torch.utils.data import Dataset
import jsonschema
import json
import numpy as np


class Tree:
    def __init__(self, tree_dict, start_idx=0):
        assert "active_prob" in tree_dict
        self.active_prob = tree_dict["active_prob"]
        self.is_read_out = tree_dict.get("is_read_out", True)
        self.mutually_exclusive_children = tree_dict.get(
            "mutually_exclusive_children", False
        )
        self.id = tree_dict.get("id", None)

        self.is_binary = tree_dict.get("is_binary", True)
        if self.is_read_out:
            self.index = start_idx
            start_idx += 1
        else:
            self.index = False

        self.children = []
        for child_dict in tree_dict.get("children", []):
            child = Tree(child_dict, start_idx)
            start_idx = child.next_index
            self.children.append(child)

        self.next_index = start_idx

        if self.mutually_exclusive_children:
            assert len(self.children) >= 2

    def __repr__(self, indent=0):
        s = " " * (indent * 2)
        s += (
            str(self.index) + " "
            if self.index is not False
            else " " * len(str(self.next_index)) + " "
        )
        s += "B" if self.is_binary else " "
        s += "x" if self.mutually_exclusive_children else " "
        s += f" {self.active_prob}"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s

    @property
    def n_features(self):
        return len(self.sample())

    @property
    def child_probs(self):
        # Return normalized numpy array of child active_probs that sum to 1
        probs = np.array([child.active_prob for child in self.children], dtype=np.float64)
        total = probs.sum()
        if total == 0:
            # Avoid division by zero; fallback to uniform
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / total
        return probs

    def sample(self, shape=None, force_inactive=False, force_active=False):
        assert not (force_inactive and force_active)

        # special sampling for shape argument
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            n_samples = np.prod(shape)
            samples = [self.sample() for _ in range(n_samples)]
            return torch.tensor(samples).view(*shape, -1).float()

        sample = []

        # is this feature active?
        is_active = (
            (torch.rand(1) <= self.active_prob).item() * (1 - (force_inactive))
            if not force_active
            else 1
        )

        # append something if this is a readout
        if self.is_read_out:
            if self.is_binary:
                sample.append(is_active)
            else:
                sample.append((is_active * torch.rand(1)))

        if self.mutually_exclusive_children:
            active_child = (
                np.random.choice(self.children, p=self.child_probs)
                if is_active
                else None
            )

        for child in self.children:
            child_force_inactive = not bool(is_active) or (
                self.mutually_exclusive_children and child != active_child
            )

            child_force_active = (
                self.mutually_exclusive_children and child == active_child
            )

            sample += child.sample(
                force_inactive=child_force_inactive, force_active=child_force_active
            )

        return sample


class TreeDataset(Dataset):
    def __init__(self, tree, true_feats, batch_size, num_batches):
        self.tree = tree
        self.true_feats = true_feats.cpu()
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        true_acts = self.tree.sample(self.batch_size)
        x = true_acts @ self.true_feats
        return x, true_acts
