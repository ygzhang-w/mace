"""Unit tests for ZBL utilities for exclusive mode."""

import pytest
import torch

from mace.tools.zbl_utils import compute_element_pair_min_distances, pair_r_max_to_tensors
from mace.tools.utils import AtomicNumberTable


class MockBatch:
    """Mock batch object for testing compute_element_pair_min_distances."""

    def __init__(self, positions, edge_index, node_attrs, shifts=None):
        self.data = {
            "positions": positions,
            "edge_index": edge_index,
            "node_attrs": node_attrs,
        }
        if shifts is not None:
            self.data["shifts"] = shifts

    def to_dict(self):
        return self.data

    def __getitem__(self, key):
        return self.data[key]


class TestPairRMaxToTensors:
    """Tests for pair_r_max_to_tensors function."""

    def test_single_pair(self):
        """Test conversion of single element pair."""
        pair_r_max = {(1, 6): 2.5}  # H-C pair
        matrix = pair_r_max_to_tensors(pair_r_max, max_z=10)

        assert matrix.shape == (10, 10)
        assert matrix[1, 6].item() == pytest.approx(2.5)
        assert matrix[6, 1].item() == pytest.approx(2.5)  # Symmetric
        assert matrix[0, 0].item() == 0.0  # Unset entry

    def test_multiple_pairs(self):
        """Test conversion of multiple element pairs."""
        pair_r_max = {
            (1, 1): 1.5,  # H-H
            (1, 6): 2.0,  # H-C
            (6, 6): 2.5,  # C-C
            (6, 8): 1.8,  # C-O
        }
        matrix = pair_r_max_to_tensors(pair_r_max, max_z=10)

        assert matrix[1, 1].item() == pytest.approx(1.5)
        assert matrix[1, 6].item() == pytest.approx(2.0)
        assert matrix[6, 1].item() == pytest.approx(2.0)  # Symmetric
        assert matrix[6, 6].item() == pytest.approx(2.5)
        assert matrix[6, 8].item() == pytest.approx(1.8)
        assert matrix[8, 6].item() == pytest.approx(1.8)  # Symmetric

    def test_empty_dict(self):
        """Test with empty dictionary."""
        matrix = pair_r_max_to_tensors({}, max_z=10)
        assert matrix.shape == (10, 10)
        assert torch.all(matrix == 0)

    def test_default_max_z(self):
        """Test default max_z value."""
        matrix = pair_r_max_to_tensors({(1, 1): 1.0})
        assert matrix.shape == (100, 100)

    def test_dtype(self):
        """Test output dtype matches default dtype."""
        matrix = pair_r_max_to_tensors({(1, 1): 1.0})
        assert matrix.dtype == torch.get_default_dtype()


class TestComputeElementPairMinDistances:
    """Tests for compute_element_pair_min_distances function."""

    @pytest.fixture
    def z_table(self):
        """Create AtomicNumberTable for H and C."""
        return AtomicNumberTable([1, 6])

    def test_single_batch(self, z_table):
        """Test with single batch containing one structure."""
        # Create simple structure: H at origin, C at (2.0, 0, 0)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_attrs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype()
        )  # H, C

        batch = MockBatch(positions, edge_index, node_attrs)
        loader = [batch]

        result = compute_element_pair_min_distances(loader, z_table)

        # Only H-C pair should be present
        assert (1, 6) in result
        assert result[(1, 6)] == pytest.approx(2.0, rel=1e-5)

    def test_multiple_structures(self, z_table):
        """Test tracking minimum distance across multiple batches."""
        # First batch: H-C distance = 3.0
        batch1 = MockBatch(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
            ),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            node_attrs=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype()
            ),
        )

        # Second batch: H-C distance = 1.5 (smaller)
        batch2 = MockBatch(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.get_default_dtype()
            ),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            node_attrs=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype()
            ),
        )

        loader = [batch1, batch2]
        result = compute_element_pair_min_distances(loader, z_table)

        # Should track minimum distance
        assert result[(1, 6)] == pytest.approx(1.5, rel=1e-5)

    def test_same_element_pair(self, z_table):
        """Test same-element pairs (H-H)."""
        # Two H atoms at distance 1.0
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_attrs = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], dtype=torch.get_default_dtype()
        )  # H, H

        batch = MockBatch(positions, edge_index, node_attrs)
        loader = [batch]

        result = compute_element_pair_min_distances(loader, z_table)

        assert (1, 1) in result
        assert result[(1, 1)] == pytest.approx(1.0, rel=1e-5)

    def test_pair_key_ordering(self, z_table):
        """Test that pair keys are always ordered (Z_i <= Z_j)."""
        # C at origin, H at (2.0, 0, 0) - reversed order in node_attrs
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_attrs = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.get_default_dtype()
        )  # C, H

        batch = MockBatch(positions, edge_index, node_attrs)
        loader = [batch]

        result = compute_element_pair_min_distances(loader, z_table)

        # Key should be (1, 6) not (6, 1)
        assert (1, 6) in result
        assert (6, 1) not in result

    def test_with_shifts(self, z_table):
        """Test that shifts are correctly handled for periodic boundaries."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_attrs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype()
        )  # H, C
        shifts = torch.tensor(
            [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=torch.get_default_dtype()
        )

        batch = MockBatch(positions, edge_index, node_attrs, shifts=shifts)
        loader = [batch]

        result = compute_element_pair_min_distances(loader, z_table)

        assert (1, 6) in result
        assert result[(1, 6)] == pytest.approx(2.0, rel=1e-5)

    def test_multiple_element_pairs(self):
        """Test with structure containing H, C, and O."""
        z_table = AtomicNumberTable([1, 6, 8])

        # H at origin, C at (1.5, 0, 0), O at (0, 2.0, 0)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.get_default_dtype(),
        )
        edge_index = torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
        )
        node_attrs = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.get_default_dtype(),
        )  # H, C, O

        batch = MockBatch(positions, edge_index, node_attrs)
        loader = [batch]

        result = compute_element_pair_min_distances(loader, z_table)

        # Check all pairs
        assert (1, 6) in result  # H-C
        assert (1, 8) in result  # H-O
        assert (6, 8) in result  # C-O
        assert result[(1, 6)] == pytest.approx(1.5, rel=1e-5)
        assert result[(1, 8)] == pytest.approx(2.0, rel=1e-5)
        # C-O distance: sqrt((1.5-0)^2 + (0-2.0)^2) = sqrt(2.25 + 4) = 2.5
        assert result[(6, 8)] == pytest.approx(2.5, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
