import pytest
import torch

from mace.modules.radial import AgnesiTransform, ExclusiveZBLBasis, ZBLBasis
from mace.tools.zbl_utils import pair_r_max_to_tensors


@pytest.fixture
def zbl_basis():
    return ZBLBasis(p=6, trainable=False)


def test_zbl_basis_initialization(zbl_basis):
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert not zbl_basis.a_exp.requires_grad
    assert not zbl_basis.a_prefactor.requires_grad


def test_trainable_zbl_basis_initialization(zbl_basis):
    zbl_basis = ZBLBasis(p=6, trainable=True)
    assert zbl_basis.p == torch.tensor(6.0)
    assert torch.allclose(zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))

    assert zbl_basis.a_exp == torch.tensor(0.300)
    assert zbl_basis.a_prefactor == torch.tensor(0.4543)
    assert zbl_basis.a_exp.requires_grad
    assert zbl_basis.a_prefactor.requires_grad


def test_forward(zbl_basis):
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]]
    )  # [n_nodes, n_node_features] - one_hot encoding of atomic numbers
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]
    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)

    assert output.shape == torch.Size([node_attrs.shape[0]])
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor([0.0031, 0.0031], dtype=torch.get_default_dtype()),
        rtol=1e-2,
    )


@pytest.fixture
def agnesi():
    return AgnesiTransform(trainable=False)


def test_agnesi_transform_initialization(agnesi: AgnesiTransform):
    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert not agnesi.a.requires_grad
    assert not agnesi.q.requires_grad
    assert not agnesi.p.requires_grad


def test_trainable_agnesi_transform_initialization():
    agnesi = AgnesiTransform(trainable=True)

    assert agnesi.q.item() == pytest.approx(0.9183, rel=1e-4)
    assert agnesi.p.item() == pytest.approx(4.5791, rel=1e-4)
    assert agnesi.a.item() == pytest.approx(1.0805, rel=1e-4)
    assert agnesi.a.requires_grad
    assert agnesi.q.requires_grad
    assert agnesi.p.requires_grad


def test_agnesi_transform_forward():
    agnesi = AgnesiTransform()
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.get_default_dtype()).unsqueeze(-1)
    node_attrs = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    atomic_numbers = torch.tensor([1, 6, 8])
    output = agnesi(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == x.shape
    assert torch.is_tensor(output)
    assert torch.allclose(
        output,
        torch.tensor(
            [0.3646, 0.2175, 0.2089], dtype=torch.get_default_dtype()
        ).unsqueeze(-1),
        rtol=1e-2,
    )


# --- ExclusiveZBLBasis Tests ---


@pytest.fixture
def pair_r_max_matrix():
    """Create pair r_max matrix for H (Z=1) and C (Z=6)."""
    pair_r_max_dict = {
        (1, 1): 1.5,  # H-H
        (1, 6): 2.0,  # H-C
        (6, 6): 2.5,  # C-C
    }
    return pair_r_max_to_tensors(pair_r_max_dict, max_z=10)


@pytest.fixture
def exclusive_zbl_basis(pair_r_max_matrix):
    return ExclusiveZBLBasis(pair_r_max_matrix=pair_r_max_matrix, p=6, trainable=False)


def test_exclusive_zbl_basis_initialization(exclusive_zbl_basis, pair_r_max_matrix):
    """Test ExclusiveZBLBasis initializes correctly."""
    assert exclusive_zbl_basis.p.item() == 6
    assert torch.allclose(
        exclusive_zbl_basis.c, torch.tensor([0.1818, 0.5099, 0.2802, 0.02817])
    )
    assert exclusive_zbl_basis.a_exp.item() == pytest.approx(0.300)
    assert exclusive_zbl_basis.a_prefactor.item() == pytest.approx(0.4543)
    assert not exclusive_zbl_basis.a_exp.requires_grad
    assert not exclusive_zbl_basis.a_prefactor.requires_grad
    # Check pair_r_max_matrix is stored
    assert torch.allclose(exclusive_zbl_basis.pair_r_max_matrix, pair_r_max_matrix)
    # Check energy_offset_matrix is initialized to zeros
    assert torch.all(exclusive_zbl_basis.energy_offset_matrix == 0)


def test_trainable_exclusive_zbl_basis():
    """Test ExclusiveZBLBasis with trainable parameters."""
    pair_r_max = pair_r_max_to_tensors({(1, 1): 1.5}, max_z=10)
    zbl = ExclusiveZBLBasis(pair_r_max_matrix=pair_r_max, p=6, trainable=True)
    assert zbl.a_exp.requires_grad
    assert zbl.a_prefactor.requires_grad


def test_exclusive_zbl_get_pair_r_max(exclusive_zbl_basis):
    """Test get_pair_r_max returns correct r_max for each edge."""
    # Setup: 2 nodes (H and C)
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # H, C
    edge_index = torch.tensor([[0, 1], [1, 0]])  # H->C, C->H
    atomic_numbers = torch.tensor([1, 6])  # H, C

    r_max = exclusive_zbl_basis.get_pair_r_max(node_attrs, edge_index, atomic_numbers)

    # H-C pair should have r_max = 2.0
    expected = torch.tensor([[2.0], [2.0]], dtype=torch.get_default_dtype())
    assert torch.allclose(r_max, expected)


def test_exclusive_zbl_get_edge_mask(exclusive_zbl_basis):
    """Test get_edge_mask correctly filters edges based on r_max."""
    # Setup: 2 nodes (H and C), H-C pair r_max = 2.0
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # H, C
    edge_index = torch.tensor([[0, 1], [1, 0]])  # H->C, C->H
    atomic_numbers = torch.tensor([1, 6])  # H, C

    # Edge length = 1.5 (within r_max=2.0) -> should be masked (False for MACE)
    x_within = torch.tensor([[1.5], [1.5]], dtype=torch.get_default_dtype())
    mask_within = exclusive_zbl_basis.get_edge_mask(
        x_within, node_attrs, edge_index, atomic_numbers
    )
    assert torch.all(~mask_within)  # All False (within ZBL range)

    # Edge length = 2.5 (outside r_max=2.0) -> should NOT be masked (True for MACE)
    x_outside = torch.tensor([[2.5], [2.5]], dtype=torch.get_default_dtype())
    mask_outside = exclusive_zbl_basis.get_edge_mask(
        x_outside, node_attrs, edge_index, atomic_numbers
    )
    assert torch.all(mask_outside)  # All True (outside ZBL range)

    # Edge length = 2.0 (exactly at r_max) -> should NOT be masked (True for MACE)
    x_at = torch.tensor([[2.0], [2.0]], dtype=torch.get_default_dtype())
    mask_at = exclusive_zbl_basis.get_edge_mask(
        x_at, node_attrs, edge_index, atomic_numbers
    )
    assert torch.all(mask_at)  # At boundary, goes to MACE


def test_exclusive_zbl_forward(exclusive_zbl_basis):
    """Test ExclusiveZBLBasis forward pass computes ZBL energy."""
    # Setup: 2 nodes (H and C)
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # H, C
    edge_index = torch.tensor([[0, 1], [1, 0]])  # H->C, C->H
    atomic_numbers = torch.tensor([1, 6])  # H, C

    # Edge length = 1.5 (within r_max=2.0)
    x = torch.tensor([[1.5], [1.5]], dtype=torch.get_default_dtype())

    output = exclusive_zbl_basis(x, node_attrs, edge_index, atomic_numbers)

    # Check output shape: per-node energy
    assert output.shape == torch.Size([2])
    assert torch.is_tensor(output)
    # ZBL energy should be positive (repulsive)
    assert torch.all(output > 0)
    # Both nodes should have same contribution (symmetric pair)
    assert torch.allclose(output[0], output[1], rtol=1e-5)


def test_exclusive_zbl_forward_outside_rmax(exclusive_zbl_basis):
    """Test that ZBL energy goes to zero outside r_max due to cutoff."""
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # H, C
    edge_index = torch.tensor([[0, 1], [1, 0]])  # H->C, C->H
    atomic_numbers = torch.tensor([1, 6])  # H, C

    # Edge length = 2.5 (outside r_max=2.0)
    x = torch.tensor([[2.5], [2.5]], dtype=torch.get_default_dtype())

    output = exclusive_zbl_basis(x, node_attrs, edge_index, atomic_numbers)

    # Outside r_max, cutoff envelope should make energy ~0
    assert torch.allclose(
        output, torch.zeros(2, dtype=torch.get_default_dtype()), atol=1e-6
    )


def test_exclusive_zbl_energy_offset(pair_r_max_matrix):
    """Test that energy_offset_matrix modifies the output."""
    zbl = ExclusiveZBLBasis(pair_r_max_matrix=pair_r_max_matrix, p=6)

    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # H, C
    edge_index = torch.tensor([[0, 1], [1, 0]])  # H->C, C->H
    atomic_numbers = torch.tensor([1, 6])  # H, C
    x = torch.tensor([[1.5], [1.5]], dtype=torch.get_default_dtype())

    # Get baseline energy
    energy_no_offset = zbl(x, node_attrs, edge_index, atomic_numbers).clone()

    # Set energy offset for H-C pair
    offset_value = 1.0
    zbl.energy_offset_matrix[1, 6] = offset_value
    zbl.energy_offset_matrix[6, 1] = offset_value

    energy_with_offset = zbl(x, node_attrs, edge_index, atomic_numbers)

    # Energy should be different with offset
    assert not torch.allclose(energy_no_offset, energy_with_offset)


def test_exclusive_zbl_repr(exclusive_zbl_basis):
    """Test ExclusiveZBLBasis string representation."""
    repr_str = repr(exclusive_zbl_basis)
    assert "ExclusiveZBLBasis" in repr_str
    assert "p=6" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
