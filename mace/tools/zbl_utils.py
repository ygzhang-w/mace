###########################################################################################
# ZBL Utilities for Exclusive Mode
# This module provides utilities for the exclusive ZBL pair repulsion mode.
###########################################################################################

import logging
from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from mace.tools.utils import AtomicNumberTable


def compute_element_pair_min_distances(
    train_loader: DataLoader,
    z_table: AtomicNumberTable,
) -> Dict[Tuple[int, int], float]:
    """
    Scan training data to find minimum interatomic distances per element pair.

    This function iterates through all training batches and tracks the minimum
    distance observed for each element pair (Z_i, Z_j). The returned r_max values
    are used as the ZBL cutoff distances in exclusive mode.

    Args:
        train_loader: DataLoader containing training data
        z_table: AtomicNumberTable with atomic numbers

    Returns:
        Dictionary mapping (Z_i, Z_j) tuples to minimum distances.
        Keys are sorted (Z_i <= Z_j) to avoid duplicates.
    """
    min_distances: Dict[Tuple[int, int], float] = defaultdict(lambda: float("inf"))
    atomic_numbers_tensor = torch.tensor(z_table.zs, dtype=torch.int64)

    logging.info("Computing element-pair minimum distances from training data...")

    for batch in train_loader:
        if hasattr(batch, "to_dict"):
            batch = batch.to_dict()

        edge_index = batch["edge_index"]
        positions = batch["positions"]
        node_attrs = batch["node_attrs"]

        # Compute edge lengths
        sender = edge_index[0]
        receiver = edge_index[1]
        shifts = batch.get("shifts", torch.zeros((edge_index.shape[1], 3)))

        vectors = positions[receiver] - positions[sender] + shifts
        lengths = torch.linalg.norm(vectors, dim=-1)

        # Get atomic numbers for each node
        node_z_indices = torch.argmax(node_attrs, dim=1)
        sender_z = atomic_numbers_tensor[node_z_indices[sender]]
        receiver_z = atomic_numbers_tensor[node_z_indices[receiver]]

        # Track minimum distance per element pair
        for i in range(len(lengths)):
            z1, z2 = int(sender_z[i].item()), int(receiver_z[i].item())
            # Sort to ensure (Z_i, Z_j) with Z_i <= Z_j
            pair = (min(z1, z2), max(z1, z2))
            dist = lengths[i].item()
            if dist > 0 and dist < min_distances[pair]:
                min_distances[pair] = dist

    # Convert defaultdict to regular dict
    result = dict(min_distances)

    # Log the computed values
    for pair, dist in sorted(result.items()):
        logging.info(f"  Element pair {pair}: min distance = {dist:.4f} Å")

    return result


def pair_r_max_to_tensors(
    pair_r_max: Dict[Tuple[int, int], float],
    max_z: int = 100,
) -> torch.Tensor:
    """
    Convert pair_r_max dictionary to a 2D tensor for efficient lookup.

    Args:
        pair_r_max: Dictionary mapping (Z_i, Z_j) to r_max values
        max_z: Maximum atomic number to support

    Returns:
        Tensor of shape (max_z, max_z) where tensor[Z_i, Z_j] = r_max
        for the pair (Z_i, Z_j). Entries without data are set to 0.
    """
    r_max_matrix = torch.zeros((max_z, max_z), dtype=torch.get_default_dtype())

    for (z1, z2), r_max in pair_r_max.items():
        r_max_matrix[z1, z2] = r_max
        r_max_matrix[z2, z1] = r_max  # Symmetric

    return r_max_matrix
