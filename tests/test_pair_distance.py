"""Tests for pair distance computation utilities used in ZBL potential."""

import numpy as np
import pytest
import torch

from mace import data, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import (
    compute_min_pair_distances,
    create_pair_r_max_tensor,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture
def simple_configs():
    """Create simple atomic configurations for testing."""
    # Configuration 1: H2O molecule
    config1 = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],  # H at ~0.96 Å from O
                [0.0, 0.96, 0.0],  # H at ~0.96 Å from O
            ]
        ),
        properties={
            "energy": -1.5,
        },
        property_weights={
            "energy": 1.0,
        },
    )
    # Configuration 2: H2O molecule with different distances
    config2 = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # H at 1.0 Å from O
                [0.0, 1.0, 0.0],  # H at 1.0 Å from O
            ]
        ),
        properties={
            "energy": -1.3,
        },
        property_weights={
            "energy": 1.0,
        },
    )
    return [config1, config2]


@pytest.fixture
def z_table():
    """Create atomic number table for H and O."""
    return tools.AtomicNumberTable([1, 8])


@pytest.fixture
def train_loader(simple_configs, z_table):
    """Create a data loader from the simple configs."""
    cutoff = 5.0
    atomic_data_list = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)
        for config in simple_configs
    ]
    return torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )


def test_compute_min_pair_distances(train_loader, z_table):
    """Test computing minimum pair distances from training data."""
    epsilon = 0.1
    min_distances = compute_min_pair_distances(train_loader, z_table, epsilon=epsilon)

    # Should have computed distances for pairs: (1, 1), (1, 8), (8, 8)
    # H-O pair should have minimum distance around 0.96 Å + epsilon
    # H-H pair distance should be sqrt(0.96^2 + 0.96^2) ≈ 1.36 Å for config1
    assert (1, 8) in min_distances or (8, 1) in min_distances
    assert (1, 1) in min_distances

    # Check H-O distance is approximately 0.96 + epsilon
    if (1, 8) in min_distances:
        assert abs(min_distances[(1, 8)] - (0.96 + epsilon)) < 0.01
    else:
        assert abs(min_distances[(8, 1)] - (0.96 + epsilon)) < 0.01


def test_compute_min_pair_distances_zero_epsilon(train_loader, z_table):
    """Test that epsilon=0 gives raw minimum distances."""
    min_distances = compute_min_pair_distances(train_loader, z_table, epsilon=0.0)

    # H-O pair should have minimum distance around 0.96 Å
    if (1, 8) in min_distances:
        assert abs(min_distances[(1, 8)] - 0.96) < 0.01
    else:
        assert abs(min_distances[(8, 1)] - 0.96) < 0.01


def test_create_pair_r_max_tensor():
    """Test creating pair r_max tensor from distance dictionary."""
    min_distances = {
        (1, 1): 1.2,  # H-H
        (1, 6): 1.5,  # H-C
        (6, 6): 2.0,  # C-C
    }

    pair_r_max = create_pair_r_max_tensor(min_distances, max_z=10)

    # Check shape
    assert pair_r_max.shape == (10, 10)

    # Check symmetric values
    assert pair_r_max[1, 1] == 1.2
    assert pair_r_max[1, 6] == 1.5
    assert pair_r_max[6, 1] == 1.5  # Should be symmetric
    assert pair_r_max[6, 6] == 2.0

    # Check default value for unset pairs
    assert pair_r_max[2, 3] == -1.0  # He-Li not in dict


def test_create_pair_r_max_tensor_default_max_z():
    """Test that default max_z is 119."""
    min_distances = {(1, 1): 1.2}
    pair_r_max = create_pair_r_max_tensor(min_distances)

    assert pair_r_max.shape == (119, 119)


def test_integration_min_distance_to_pair_r_max(train_loader, z_table):
    """Test the full workflow from training data to pair_r_max tensor."""
    epsilon = 0.05
    min_distances = compute_min_pair_distances(train_loader, z_table, epsilon=epsilon)
    pair_r_max = create_pair_r_max_tensor(min_distances)

    # Verify the tensor can be used with ZBLBasis
    from mace.modules.radial import ZBLBasis

    zbl = ZBLBasis(p=6, pair_r_max=pair_r_max)
    assert zbl.pair_r_max is not None


if __name__ == "__main__":
    pytest.main([__file__])
