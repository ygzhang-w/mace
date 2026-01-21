"""Tests for EnergyForcesLoss class."""

import pytest
import torch

from mace.modules import EnergyForcesLoss
from mace.tools.torch_geometric import Batch


@pytest.fixture
def sample_batch_and_pred():
    """Create sample batch and prediction data for testing."""
    torch.set_default_dtype(torch.float64)

    batch = Batch(
        energy=torch.tensor([1.0, 2.0]),
        forces=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        ptr=torch.tensor([0, 2, 3]),
        weight=torch.tensor([1.0, 1.0]),
        energy_weight=torch.tensor([1.0, 1.0]),
        forces_weight=torch.tensor([1.0, 1.0]),
    )

    pred = {
        "energy": torch.tensor([1.1, 2.1]),
        "forces": torch.tensor(
            [[0.11, 0.21, 0.31], [0.41, 0.51, 0.61], [0.71, 0.81, 0.91]]
        ),
    }

    return batch, pred


class TestEnergyForcesLoss:
    """Test suite for EnergyForcesLoss class."""

    def test_energy_forces_loss_init(self):
        """Test EnergyForcesLoss initialization with default values."""
        loss_fn = EnergyForcesLoss()
        assert loss_fn.energy_weight == 1.0
        assert loss_fn.forces_weight == 1.0

    def test_energy_forces_loss_init_custom_weights(self):
        """Test EnergyForcesLoss initialization with custom weights."""
        loss_fn = EnergyForcesLoss(energy_weight=2.0, forces_weight=50.0)
        assert loss_fn.energy_weight == 2.0
        assert loss_fn.forces_weight == 50.0

    def test_energy_forces_loss_forward(self, sample_batch_and_pred):
        """Test EnergyForcesLoss forward pass returns a tensor."""
        batch, pred = sample_batch_and_pred
        loss_fn = EnergyForcesLoss()
        loss = loss_fn(batch, pred)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss >= 0  # loss should be non-negative

    def test_energy_forces_loss_repr(self):
        """Test EnergyForcesLoss __repr__ method."""
        loss_fn = EnergyForcesLoss(energy_weight=1.5, forces_weight=100.0)
        repr_str = repr(loss_fn)
        assert "EnergyForcesLoss" in repr_str
        assert "energy_weight=1.500" in repr_str
        assert "forces_weight=100.000" in repr_str

    def test_energy_forces_loss_is_module(self):
        """Test EnergyForcesLoss is a torch.nn.Module."""
        loss_fn = EnergyForcesLoss()
        assert isinstance(loss_fn, torch.nn.Module)

    def test_energy_forces_loss_buffers_registered(self):
        """Test that weights are registered as buffers."""
        loss_fn = EnergyForcesLoss(energy_weight=2.0, forces_weight=50.0)
        buffer_names = [name for name, _ in loss_fn.named_buffers()]
        assert "energy_weight" in buffer_names
        assert "forces_weight" in buffer_names
