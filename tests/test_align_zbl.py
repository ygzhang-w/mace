"""Unit tests for ZBL alignment utilities for exclusive mode."""

import pytest
import torch
import numpy as np

from mace.modules.radial import ExclusiveZBLBasis
from mace.tools.align_zbl import align_exclusive_zbl, _compute_energy_offset
from mace.tools.utils import AtomicNumberTable
from mace.tools.zbl_utils import pair_r_max_to_tensors
from mace.tools.scatter import scatter_sum


class MockAtomicEnergiesBlock(torch.nn.Module):
    """Mock atomic energies block for testing."""

    def __init__(self, atomic_energies: np.ndarray):
        super().__init__()
        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        # node_attrs: [n_nodes, n_elements]
        # Return atomic energies for each node based on one-hot encoding
        # Output shape: [n_nodes, n_heads] where n_heads=1
        e0 = torch.matmul(node_attrs, self.atomic_energies.unsqueeze(-1))
        return e0  # [n_nodes, 1]


class MockMACEModel(torch.nn.Module):
    """Mock MACE model for testing align_exclusive_zbl.
    
    This mock model includes a simple forward method that computes:
    - Atomic energies (E0)
    - A simple distance-based interaction energy
    """

    def __init__(
        self,
        pair_r_max_matrix: torch.Tensor,
        atomic_numbers: list,
        pair_repulsion_type: str = "exclusive",
        atomic_energies: np.ndarray = None,
    ):
        super().__init__()
        self.pair_repulsion_type = pair_repulsion_type
        self.pair_repulsion_fn = ExclusiveZBLBasis(
            pair_r_max_matrix=pair_r_max_matrix, p=6
        )
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        # Add a dummy parameter so next(model.parameters()) works
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))
        self.pair_repulsion = True
        
        # Add atomic energies function
        if atomic_energies is None:
            # Default: small atomic energies for H and C
            atomic_energies = np.array([-1.0, -5.0])  # H, C
        self.atomic_energies_fn = MockAtomicEnergiesBlock(atomic_energies)
        
        # Simple interaction coefficient (mimics MACE interaction)
        self.register_buffer(
            "interaction_coeff", 
            torch.tensor(-0.1, dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        data: dict,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> dict:
        """Simple forward pass for testing.
        
        Computes:
        - E0: atomic energies
        - E_interaction: simple distance-based interaction (1/r decay)
        - E_ZBL: ZBL pair repulsion (if enabled)
        """
        positions = data["positions"]
        node_attrs = data["node_attrs"]
        edge_index = data["edge_index"]
        batch = data["batch"]
        shifts = data.get("shifts", torch.zeros((edge_index.shape[1], 3)))
        
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        
        # Compute atomic energies (E0)
        node_e0 = self.atomic_energies_fn(node_attrs)[:, 0]  # [n_nodes]
        e0 = scatter_sum(
            src=node_e0, index=batch, dim=0, dim_size=num_graphs
        )
        
        # Compute edge vectors and lengths
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        
        # Simple interaction energy: coefficient * sum(1/r)
        # This mimics MACE's learned interaction
        interaction_energy_per_edge = self.interaction_coeff / lengths
        interaction_energy_per_node = scatter_sum(
            src=0.5 * interaction_energy_per_edge.squeeze(-1),
            index=receiver,
            dim=0,
            dim_size=positions.shape[0],
        )
        interaction_energy = scatter_sum(
            src=interaction_energy_per_node,
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        
        # Compute ZBL energy if enabled
        if hasattr(self, "pair_repulsion") and self.pair_repulsion:
            zbl_node_energy = self.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.atomic_numbers
            )
            zbl_energy = scatter_sum(
                src=zbl_node_energy, index=batch, dim=0, dim_size=num_graphs
            )
        else:
            zbl_energy = torch.zeros_like(e0)
        
        total_energy = e0 + interaction_energy + zbl_energy
        
        return {
            "energy": total_energy,
            "node_energy": node_e0 + interaction_energy_per_node,
            "forces": None,
            "virials": None,
            "stress": None,
        }


class MockMACEModelNonExclusive(torch.nn.Module):
    """Mock MACE model without exclusive ZBL mode."""

    def __init__(self):
        super().__init__()
        self.pair_repulsion_type = "additional"
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))


class TestComputeEnergyOffset:
    """Tests for _compute_energy_offset function."""

    @pytest.fixture
    def z_table(self):
        """Create AtomicNumberTable for H and C."""
        return AtomicNumberTable([1, 6])

    @pytest.fixture
    def pair_r_max_matrix(self):
        """Create pair r_max matrix."""
        pair_r_max_dict = {
            (1, 1): 1.5,  # H-H
            (1, 6): 2.0,  # H-C
            (6, 6): 2.5,  # C-C
        }
        return pair_r_max_to_tensors(pair_r_max_dict, max_z=10)

    @pytest.fixture
    def mock_model(self, pair_r_max_matrix):
        """Create mock MACE model."""
        return MockMACEModel(
            pair_r_max_matrix=pair_r_max_matrix,
            atomic_numbers=[1, 6],
        )

    def test_compute_energy_offset_returns_float(self, mock_model, z_table):
        """Test that _compute_energy_offset returns a float."""
        device = next(mock_model.parameters()).device
        dtype = torch.get_default_dtype()

        offset = _compute_energy_offset(
            model=mock_model,
            z1=1,
            z2=6,
            r_max=2.0,
            z_table=z_table,
            device=device,
            dtype=dtype,
        )

        assert isinstance(offset, float)

    def test_compute_energy_offset_h_c_pair(self, mock_model, z_table):
        """Test energy offset computation for H-C pair.
        
        The offset should equal the MACE interaction energy at r_max.
        """
        device = next(mock_model.parameters()).device
        dtype = torch.get_default_dtype()
        r_max = 2.0

        offset = _compute_energy_offset(
            model=mock_model,
            z1=1,
            z2=6,
            r_max=r_max,
            z_table=z_table,
            device=device,
            dtype=dtype,
        )

        # The mock model has interaction_coeff = -0.1
        # At r = r_max = 2.0, the MACE interaction energy = -0.1 / 2.0 = -0.05
        # The offset should equal this value (not divided by 2, as ZBL forward
        # already handles the 0.5 factor for bidirectional edges)
        expected_offset = mock_model.interaction_coeff.item() / r_max
        
        assert isinstance(offset, float)
        assert abs(offset - expected_offset) < 1e-5

    def test_compute_energy_offset_same_element(self, mock_model, z_table):
        """Test energy offset computation for same element pair (H-H)."""
        device = next(mock_model.parameters()).device
        dtype = torch.get_default_dtype()
        r_max = 1.5

        offset = _compute_energy_offset(
            model=mock_model,
            z1=1,
            z2=1,
            r_max=r_max,
            z_table=z_table,
            device=device,
            dtype=dtype,
        )

        # Expected offset for H-H at r=1.5
        expected_offset = mock_model.interaction_coeff.item() / r_max
        
        assert isinstance(offset, float)
        assert abs(offset - expected_offset) < 1e-5

    def test_compute_energy_offset_different_r_max(self, mock_model, z_table):
        """Test that different r_max values give different offsets."""
        device = next(mock_model.parameters()).device
        dtype = torch.get_default_dtype()

        # Compute offset at r_max = 1.5
        offset_15 = _compute_energy_offset(
            model=mock_model,
            z1=1,
            z2=6,
            r_max=1.5,
            z_table=z_table,
            device=device,
            dtype=dtype,
        )

        # Compute offset at r_max = 2.5
        offset_25 = _compute_energy_offset(
            model=mock_model,
            z1=1,
            z2=6,
            r_max=2.5,
            z_table=z_table,
            device=device,
            dtype=dtype,
        )

        # Different r_max should give different offsets
        assert offset_15 != offset_25
        # Larger r_max should give smaller absolute offset (1/r decay)
        assert abs(offset_15) > abs(offset_25)


class TestAlignExclusiveZBL:
    """Tests for align_exclusive_zbl function."""

    @pytest.fixture
    def z_table(self):
        """Create AtomicNumberTable for H and C."""
        return AtomicNumberTable([1, 6])

    @pytest.fixture
    def pair_r_max_matrix(self):
        """Create pair r_max matrix."""
        pair_r_max_dict = {
            (1, 1): 1.5,  # H-H
            (1, 6): 2.0,  # H-C
            (6, 6): 2.5,  # C-C
        }
        return pair_r_max_to_tensors(pair_r_max_dict, max_z=10)

    @pytest.fixture
    def mock_model(self, pair_r_max_matrix):
        """Create mock MACE model."""
        return MockMACEModel(
            pair_r_max_matrix=pair_r_max_matrix,
            atomic_numbers=[1, 6],
        )

    def test_align_exclusive_zbl_returns_model(self, mock_model, z_table):
        """Test that align_exclusive_zbl returns a model."""
        result = align_exclusive_zbl(mock_model, z_table)
        assert isinstance(result, torch.nn.Module)

    def test_align_exclusive_zbl_updates_offset_matrix(self, mock_model, z_table):
        """Test that energy_offset_matrix is updated with MACE interaction energies."""
        # Initially all zeros
        assert torch.all(mock_model.pair_repulsion_fn.energy_offset_matrix == 0)

        align_exclusive_zbl(mock_model, z_table)

        offset_matrix = mock_model.pair_repulsion_fn.energy_offset_matrix

        # Matrix should have correct shape
        assert offset_matrix.shape == mock_model.pair_repulsion_fn.pair_r_max_matrix.shape

        # Offsets should be non-zero (MACE interaction energy at r_max)
        # For H-C at r_max=2.0: offset = -0.1 / 2.0 / 2 = -0.025
        assert offset_matrix[1, 6].item() != 0.0
        assert offset_matrix[1, 6].item() == offset_matrix[6, 1].item()  # Symmetric

    def test_align_exclusive_zbl_symmetric_offsets(self, mock_model, z_table):
        """Test that offset matrix is symmetric."""
        align_exclusive_zbl(mock_model, z_table)

        offset_matrix = mock_model.pair_repulsion_fn.energy_offset_matrix

        # H-C should equal C-H
        assert offset_matrix[1, 6].item() == offset_matrix[6, 1].item()

    def test_align_exclusive_zbl_continuity_at_boundary(self, mock_model, z_table):
        """Test that energy is continuous at r_max boundary after alignment.
        
        At r = r_max:
        - ZBL envelope = 0, so raw ZBL contribution = 0
        - With new formula: E_zbl = raw_zbl * envelope + offset * (1 - envelope)
        - At r_max: E_zbl = 0 + offset * 1 = offset
        - offset = E_mace(r_max), so E_zbl(r_max) = E_mace(r_max) ✓
        """
        align_exclusive_zbl(mock_model, z_table)
        
        device = next(mock_model.parameters()).device
        dtype = torch.get_default_dtype()
        r_max = 2.0  # H-C r_max
        
        # Create dimer at r_max
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [r_max, 0.0, 0.0]], dtype=dtype, device=device
        )
        node_attrs = torch.zeros((2, 2), dtype=dtype, device=device)
        node_attrs[0, 0] = 1.0  # H (index 0 in z_table)
        node_attrs[1, 1] = 1.0  # C (index 1 in z_table)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
        
        # Get vectors and lengths
        vectors = positions[edge_index[1]] - positions[edge_index[0]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        
        # Compute ZBL energy at r_max (includes offset now)
        zbl_energy_per_node = mock_model.pair_repulsion_fn(
            lengths, node_attrs, edge_index, mock_model.atomic_numbers
        )
        zbl_total = zbl_energy_per_node.sum().item()
        
        # Compute expected MACE interaction energy at r_max
        # interaction_coeff / r_max for full bidirectional interaction
        expected_mace_interaction = mock_model.interaction_coeff.item() / r_max
        
        # ZBL energy at boundary should equal MACE interaction energy
        assert abs(zbl_total - expected_mace_interaction) < 1e-5

    def test_align_exclusive_zbl_non_exclusive_model(self, z_table):
        """Test that non-exclusive models are skipped with warning."""
        model = MockMACEModelNonExclusive()

        # Should return model unchanged
        result = align_exclusive_zbl(model, z_table)
        assert result is model

    def test_align_exclusive_zbl_no_pair_repulsion_type(self, z_table):
        """Test model without pair_repulsion_type attribute."""

        class ModelWithoutAttr(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(1.0))

        model = ModelWithoutAttr()
        result = align_exclusive_zbl(model, z_table)
        assert result is model

    def test_align_exclusive_zbl_model_mode_restored(self, mock_model, z_table):
        """Test that model training mode is restored after alignment."""
        mock_model.train()
        assert mock_model.training

        align_exclusive_zbl(mock_model, z_table)

        # Model should be back in training mode
        assert mock_model.training

    def test_align_exclusive_zbl_preserves_other_params(
        self, pair_r_max_matrix, z_table
    ):
        """Test that alignment doesn't change other model parameters."""
        model = MockMACEModel(
            pair_r_max_matrix=pair_r_max_matrix,
            atomic_numbers=[1, 6],
        )

        # Store original values
        original_c = model.pair_repulsion_fn.c.clone()
        original_a_exp = model.pair_repulsion_fn.a_exp.clone()
        original_a_prefactor = model.pair_repulsion_fn.a_prefactor.clone()
        original_pair_r_max = model.pair_repulsion_fn.pair_r_max_matrix.clone()

        align_exclusive_zbl(model, z_table)

        # These should be unchanged
        assert torch.allclose(model.pair_repulsion_fn.c, original_c)
        assert torch.allclose(model.pair_repulsion_fn.a_exp, original_a_exp)
        assert torch.allclose(model.pair_repulsion_fn.a_prefactor, original_a_prefactor)
        assert torch.allclose(
            model.pair_repulsion_fn.pair_r_max_matrix, original_pair_r_max
        )


class TestAlignExclusiveZBLWithZeroPairs:
    """Tests for align_exclusive_zbl with some pairs having r_max=0."""

    @pytest.fixture
    def z_table(self):
        """Create AtomicNumberTable for H, C, and O."""
        return AtomicNumberTable([1, 6, 8])

    @pytest.fixture
    def sparse_pair_r_max_matrix(self):
        """Create pair r_max matrix with some zero entries."""
        pair_r_max_dict = {
            (1, 6): 2.0,  # H-C only
            # H-H, C-C, H-O, C-O, O-O are all 0 (no data)
        }
        return pair_r_max_to_tensors(pair_r_max_dict, max_z=10)

    def test_align_skips_zero_r_max_pairs(self, sparse_pair_r_max_matrix, z_table):
        """Test that pairs with r_max=0 are skipped."""
        model = MockMACEModel(
            pair_r_max_matrix=sparse_pair_r_max_matrix,
            atomic_numbers=[1, 6, 8],
            atomic_energies=np.array([-1.0, -5.0, -3.0]),  # H, C, O
        )

        align_exclusive_zbl(model, z_table)

        offset_matrix = model.pair_repulsion_fn.energy_offset_matrix

        # Pairs with r_max=0 should definitely have offset=0 (not processed)
        assert offset_matrix[1, 1].item() == 0.0  # H-H (r_max=0)
        assert offset_matrix[6, 6].item() == 0.0  # C-C (r_max=0)
        assert offset_matrix[8, 8].item() == 0.0  # O-O (r_max=0)
        assert offset_matrix[1, 8].item() == 0.0  # H-O (r_max=0)
        assert offset_matrix[6, 8].item() == 0.0  # C-O (r_max=0)

        # H-C was processed (r_max=2.0), offset should be non-zero
        assert offset_matrix[1, 6].item() != 0.0
        assert offset_matrix[1, 6].item() == offset_matrix[6, 1].item()  # Symmetric


if __name__ == "__main__":
    pytest.main([__file__])
