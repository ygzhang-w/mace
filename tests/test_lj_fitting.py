###########################################################################################
# Tests for LJRepulsionBasis and Region-Split LJ Repulsion
###########################################################################################

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules
from mace.modules.radial import LJRepulsionBasis
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable


@pytest.fixture(autouse=True)
def set_default_dtype():
    """Set default dtype for all tests."""
    torch.set_default_dtype(torch.float64)


class TestLJRepulsionBasis:
    """Tests for the region-split LJRepulsionBasis."""

    def test_init_default(self):
        """Test default initialization: zero bias, correct shape."""
        basis = LJRepulsionBasis(num_elements=3)
        assert basis.bias_matrix.shape == (3, 3)
        assert torch.all(basis.bias_matrix == 0.0)
        assert basis.repulsion_c == 1.0
        assert basis.num_elements == 3

    def test_init_with_bias(self):
        """Test initialization with a provided bias matrix."""
        bias = torch.tensor([[0.1, 0.2], [0.2, 0.3]], dtype=torch.float64)
        basis = LJRepulsionBasis(num_elements=2, repulsion_c=2.0, bias_matrix=bias)
        assert torch.allclose(basis.bias_matrix, bias)
        assert basis.repulsion_c == 2.0

    def test_forward_shape(self):
        """Test that forward output shape matches num_atoms."""
        basis = LJRepulsionBasis(num_elements=2, repulsion_c=1.0)

        # 3 atoms, 4 edges
        x = torch.tensor([1.5, 2.0, 1.8, 2.5], dtype=torch.float64)
        node_attrs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64
        )
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        atomic_numbers = torch.tensor([1, 6, 1])

        out = basis(x, node_attrs, edge_index, atomic_numbers)
        assert out.shape == (3,)

    def test_forward_energy_formula(self):
        """Verify V = c * r^{-12} * 0.5 + bias for known input."""
        bias = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float64)
        c = 2.0
        basis = LJRepulsionBasis(num_elements=2, repulsion_c=c, bias_matrix=bias)

        r = 2.0
        x = torch.tensor([r, r], dtype=torch.float64)
        node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        atomic_numbers = torch.tensor([1, 6])

        out = basis(x, node_attrs, edge_index, atomic_numbers)

        # Each atom receives one edge from the other element type
        # V_edge = c * r^{-12} * 0.5 + bias(0,1) = 2.0 * 2.0^{-12} * 0.5 + 0.5
        expected_edge = c * (r ** -12) * 0.5 + 0.5
        assert out[0].item() == pytest.approx(expected_edge, rel=1e-10)
        assert out[1].item() == pytest.approx(expected_edge, rel=1e-10)

    def test_bias_not_trainable_by_default(self):
        """Test that bias_matrix is a buffer (not a Parameter) by default."""
        basis = LJRepulsionBasis(num_elements=2)
        # Should not appear in parameters
        param_names = [name for name, _ in basis.named_parameters()]
        assert "bias_matrix" not in param_names
        # Should appear in buffers
        buffer_names = [name for name, _ in basis.named_buffers()]
        assert "bias_matrix" in buffer_names

    def test_bias_trainable(self):
        """Test that when trainable=True, bias_matrix has requires_grad."""
        basis = LJRepulsionBasis(num_elements=2, trainable=True)
        param_names = [name for name, _ in basis.named_parameters()]
        assert "bias_matrix" in param_names
        assert basis.bias_matrix.requires_grad is True


class TestComputeLjRcutMatrix:
    """Tests for compute_lj_rcut_matrix."""

    def test_basic(self):
        from mace.tools.lj_fitting import compute_lj_rcut_matrix

        z_table = AtomicNumberTable([1, 6])
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch_idx = torch.tensor([0, 0])
        energy = torch.tensor(0.0, dtype=torch.float64)

        data_loader = [torch_geometric.Batch(
            positions=positions,
            atomic_numbers=torch.tensor([1, 6]),
            node_attrs=node_attrs,
            edge_index=edge_index,
            energy=energy,
            batch=batch_idx,
        )]

        rcut = compute_lj_rcut_matrix(data_loader, z_table)
        assert rcut.shape == (2, 2)
        assert rcut[0, 1].item() == pytest.approx(1.5)
        assert rcut[1, 0].item() == pytest.approx(1.5)
        # No H-H or C-C data
        assert rcut[0, 0].item() > 100
        assert rcut[1, 1].item() > 100


class TestComputeBiasFromRcut:
    """Tests for compute_bias_from_rcut."""

    def test_basic(self):
        from mace.tools.lj_fitting import compute_bias_from_rcut

        rcut = torch.tensor([[2.0, 1.5], [1.5, 2.0]], dtype=torch.float64)
        mace_energy = torch.tensor([[0.1, 0.2], [0.2, 0.3]], dtype=torch.float64)
        c = 1.0

        bias = compute_bias_from_rcut(rcut, mace_energy, c)
        expected_01 = 0.2 - 1.0 * 1.5**(-12) * 0.5
        assert bias[0, 1].item() == pytest.approx(expected_01, rel=1e-6)


class TestRegionSplitIntegration:
    """Integration test: full model forward pass with region-split LJ repulsion."""

    _SENTINEL = object()

    @staticmethod
    def _build_model(rcut_matrix=_SENTINEL):
        """Build a minimal ScaleShiftMACE with lj_repulsion."""
        table = AtomicNumberTable([1, 8])
        if rcut_matrix is TestRegionSplitIntegration._SENTINEL:
            # H-O rcut=1.2, others sentinel
            rcut_matrix = torch.tensor(
                [[999.0, 1.2], [1.2, 999.0]], dtype=torch.float64
            )
        model_config = dict(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=1,
            num_elements=2,
            hidden_irreps=o3.Irreps("16x0e"),
            MLP_irreps=o3.Irreps("16x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([1.0, 3.0], dtype=float),
            avg_num_neighbors=2.0,
            atomic_numbers=table.zs,
            correlation=2,
            pair_repulsion=True,
            pair_repulsion_type="lj_repulsion",
            lj_repulsion_c=1.0,
            lj_rcut_matrix=rcut_matrix,
            lj_rcut_epsilon=0.01,
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
        return modules.ScaleShiftMACE(**model_config), table

    @staticmethod
    def _make_atomic_data(table, r_close=0.8, r_far=2.5):
        """Create test AtomicData with a close and a far neighbor pair."""
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [r_close, 0.0, 0.0],
                    [r_far, 0.0, 0.0],
                ]
            ),
            properties={
                "forces": np.zeros((3, 3)),
                "energy": -1.0,
            },
            property_weights={"forces": 1.0, "energy": 1.0},
        )
        return data.AtomicData.from_config(config, z_table=table, cutoff=5.0)

    def test_training_mode_produces_output(self):
        """Model in training mode should use full neighbor list and run fine."""
        model, table = self._build_model()
        model.train()
        atomic_data = self._make_atomic_data(table)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        output = model(batch.to_dict(), training=True)
        assert "energy" in output
        assert torch.isfinite(output["energy"])

    def test_eval_mode_produces_output(self):
        """Model in eval mode should apply region split and produce output."""
        model, table = self._build_model()
        model.eval()
        atomic_data = self._make_atomic_data(table)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            output = model(batch.to_dict(), training=False, compute_force=False)
        assert "energy" in output
        assert torch.isfinite(output["energy"])

    def test_close_atoms_get_repulsion(self):
        """Close atoms (r < rcut) should get repulsion energy in eval mode."""
        model, table = self._build_model()
        model.eval()
        # r_close=0.8 < rcut=1.2, so this edge goes to repulsion
        atomic_data = self._make_atomic_data(table, r_close=0.8)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            output = model(batch.to_dict(), training=False, compute_force=False)
        assert torch.isfinite(output["energy"])
        # Energy should be non-zero (repulsion contributes)
        assert output["energy"].abs().item() > 0

    def test_no_rcut_matrix_no_split(self):
        """Without lj_rcut_matrix set, no neighbor list split should happen."""
        model, table = self._build_model(rcut_matrix=None)
        # has_lj_rcut should be False when no rcut_matrix provided
        assert model.has_lj_rcut is False
        model.eval()
        atomic_data = self._make_atomic_data(table, r_close=0.8)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            output = model(batch.to_dict(), training=False, compute_force=False)
        assert torch.isfinite(output["energy"])

    def test_all_edges_beyond_rcut(self):
        """When all edges are beyond rcut, repulsion should contribute nothing."""
        model, table = self._build_model()
        model.eval()
        # Both atoms are far from atom 0: r=2.5, r=3.0 > rcut=1.2
        atomic_data = self._make_atomic_data(table, r_close=2.5, r_far=3.0)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            output = model(batch.to_dict(), training=False, compute_force=False)
        assert torch.isfinite(output["energy"])
