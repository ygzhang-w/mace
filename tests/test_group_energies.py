"""
Tests for group_energies functionality.

This module tests:
1. GroupEnergyBlock computation correctness
2. Gaussian weight calculation
3. Gradient flow (differentiability)
4. Loss function computation
5. compute_group_energies_from_atomic utility function
"""

import numpy as np
import pytest
import torch

from mace.modules import GroupEnergyBlock, PolynomialCutoff
from mace.modules.loss import weighted_mean_squared_error_group_energies
from mace.modules.utils import compute_group_energies_from_atomic
from mace.tools.scatter import scatter_sum


@pytest.fixture(name="simple_graph")
def _simple_graph():
    """Create a simple graph with 4 atoms and known distances."""
    # 4 atoms: 0 is central, 1,2,3 are neighbors
    # Distances: 0-1: 1.0, 0-2: 2.0, 0-3: 3.0
    edge_index = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],  # sender
            [0, 0, 0, 1, 2, 3],  # receiver
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    node_energy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    return edge_index, lengths, node_energy


@pytest.fixture(name="r_max")
def _r_max():
    return 5.0


class TestGroupEnergyBlock:
    """Tests for GroupEnergyBlock."""

    def test_init(self, r_max):
        """Test GroupEnergyBlock initialization."""
        block = GroupEnergyBlock(r_max=r_max)
        assert block.sigma == r_max / 3.0
        assert block.coeff == -0.5 / (block.sigma**2)
        assert block.r_max.item() == r_max
        assert block.p.item() == 6

    def test_init_custom_sigma(self, r_max):
        """Test GroupEnergyBlock with custom sigma."""
        custom_sigma = 1.5
        block = GroupEnergyBlock(r_max=r_max, sigma=custom_sigma)
        assert block.sigma == custom_sigma
        assert block.coeff == -0.5 / (custom_sigma**2)

    def test_forward_shape(self, simple_graph, r_max):
        """Test that forward returns correct shape."""
        edge_index, lengths, node_energy = simple_graph
        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )
        assert group_energy.shape == node_energy.shape

    def test_forward_includes_self_energy(self, r_max):
        """Test that group_energy includes self energy."""
        # Single isolated atom (no neighbors)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        lengths = torch.tensor([])
        node_energy = torch.tensor([5.0])

        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        # For isolated atom, group_energy should equal node_energy
        assert torch.allclose(group_energy, node_energy)

    def test_forward_gaussian_weighting(self, simple_graph, r_max):
        """Test Gaussian weighting computation."""
        edge_index, lengths, node_energy = simple_graph
        sigma = r_max / 3.0
        block = GroupEnergyBlock(r_max=r_max)

        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        # Manually compute expected group_energy for atom 0
        # Neighbors: 1 (d=1.0), 2 (d=2.0), 3 (d=3.0)
        coeff = -0.5 / (sigma**2)
        weights = torch.exp(coeff * lengths[:3].pow(2))
        cutoffs = PolynomialCutoff.calculate_envelope(lengths[:3], r_max, 6)
        full_weights = weights * cutoffs

        expected_neighbor_contrib = (
            full_weights[0] * node_energy[1]
            + full_weights[1] * node_energy[2]
            + full_weights[2] * node_energy[3]
        )
        expected_group_energy_0 = node_energy[0] + expected_neighbor_contrib

        assert torch.allclose(group_energy[0], expected_group_energy_0, rtol=1e-5)

    def test_gradient_flow(self, simple_graph, r_max):
        """Test that gradients flow through the computation."""
        edge_index, lengths, node_energy = simple_graph
        node_energy = node_energy.clone().requires_grad_(True)

        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        # Compute loss and backward
        loss = group_energy.sum()
        loss.backward()

        # Check gradients exist
        assert node_energy.grad is not None
        assert not torch.all(node_energy.grad == 0)

    def test_repr(self, r_max):
        """Test string representation."""
        block = GroupEnergyBlock(r_max=r_max)
        repr_str = repr(block)
        assert "GroupEnergyBlock" in repr_str
        assert "r_max" in repr_str
        assert "sigma" in repr_str


class TestComputeGroupEnergiesFromAtomic:
    """Tests for compute_group_energies_from_atomic utility function."""

    def test_basic_computation(self, simple_graph, r_max):
        """Test basic computation matches GroupEnergyBlock."""
        edge_index, lengths, atomic_energies = simple_graph

        # Using utility function
        group_energies = compute_group_energies_from_atomic(
            atomic_energies=atomic_energies,
            edge_index=edge_index,
            lengths=lengths,
            r_max=r_max,
        )

        # Using block
        block = GroupEnergyBlock(r_max=r_max)
        expected = block(
            node_energy=atomic_energies, edge_index=edge_index, lengths=lengths
        )

        assert torch.allclose(group_energies, expected)

    def test_custom_sigma(self, simple_graph, r_max):
        """Test with custom sigma parameter."""
        edge_index, lengths, atomic_energies = simple_graph
        custom_sigma = 2.0

        group_energies = compute_group_energies_from_atomic(
            atomic_energies=atomic_energies,
            edge_index=edge_index,
            lengths=lengths,
            r_max=r_max,
            sigma=custom_sigma,
        )

        # Using block with same sigma
        block = GroupEnergyBlock(r_max=r_max, sigma=custom_sigma)
        expected = block(
            node_energy=atomic_energies, edge_index=edge_index, lengths=lengths
        )

        assert torch.allclose(group_energies, expected)


class TestGaussianWeights:
    """Tests for Gaussian weight properties."""

    def test_weight_at_zero_distance(self, r_max):
        """Test that weight at distance 0 is maximum (1.0 before cutoff)."""
        sigma = r_max / 3.0
        coeff = -0.5 / (sigma**2)
        distance = torch.tensor([0.0])

        gaussian_weight = torch.exp(coeff * distance.pow(2))
        assert torch.allclose(gaussian_weight, torch.tensor([1.0]))

    def test_weight_decay_with_distance(self, r_max):
        """Test that weight decreases with distance."""
        sigma = r_max / 3.0
        coeff = -0.5 / (sigma**2)
        distances = torch.tensor([0.0, 1.0, 2.0, 3.0])

        gaussian_weights = torch.exp(coeff * distances.pow(2))

        # Weights should be monotonically decreasing
        for i in range(len(gaussian_weights) - 1):
            assert gaussian_weights[i] > gaussian_weights[i + 1]

    def test_cutoff_at_r_max(self, r_max):
        """Test that cutoff is zero at r_max."""
        p = 6
        distance = torch.tensor([r_max])
        cutoff = PolynomialCutoff.calculate_envelope(distance, r_max, p)
        assert torch.allclose(cutoff, torch.tensor([0.0]), atol=1e-6)

    def test_cutoff_beyond_r_max(self, r_max):
        """Test that cutoff is zero beyond r_max."""
        p = 6
        distance = torch.tensor([r_max + 1.0])
        cutoff = PolynomialCutoff.calculate_envelope(distance, r_max, p)
        assert torch.allclose(cutoff, torch.tensor([0.0]))


class TestDifferentiability:
    """Tests for differentiability of group_energy computations."""

    def test_second_order_gradients(self, simple_graph, r_max):
        """Test that second-order gradients can be computed with respect to lengths."""
        edge_index, lengths, node_energy = simple_graph
        lengths = lengths.clone().requires_grad_(True)

        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        # First derivative
        (grad,) = torch.autograd.grad(
            group_energy.sum(), lengths, create_graph=True
        )

        # Second derivative (Hessian diagonal) - should work for lengths
        # since weights depend non-linearly on lengths
        (grad2,) = torch.autograd.grad(grad.sum(), lengths)

        assert grad2 is not None

    def test_gradient_wrt_lengths(self, simple_graph, r_max):
        """Test gradients with respect to edge lengths."""
        edge_index, lengths, node_energy = simple_graph
        lengths = lengths.clone().requires_grad_(True)

        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        loss = group_energy.sum()
        loss.backward()

        assert lengths.grad is not None


class TestBatchProcessing:
    """Tests for batch processing of multiple graphs."""

    def test_multiple_graphs(self, r_max):
        """Test computation with multiple graphs in a batch."""
        # Two separate graphs
        # Graph 1: atoms 0,1 (edge 1->0)
        # Graph 2: atoms 2,3 (edge 3->2)
        edge_index = torch.tensor(
            [
                [1, 0, 3, 2],  # sender
                [0, 1, 2, 3],  # receiver
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([1.0, 1.0, 2.0, 2.0])
        node_energy = torch.tensor([1.0, 2.0, 3.0, 4.0])

        block = GroupEnergyBlock(r_max=r_max)
        group_energy = block(
            node_energy=node_energy, edge_index=edge_index, lengths=lengths
        )

        # Each graph should be processed independently
        assert group_energy.shape == node_energy.shape
        assert len(group_energy) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
