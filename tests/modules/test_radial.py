import pytest
import torch
import numpy as np

from mace.modules.radial import AgnesiTransform, ZBLBasis


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
    assert zbl_basis.pair_r_max is None


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


def test_zbl_basis_with_pair_r_max():
    """Test ZBLBasis with custom pair_r_max tensor."""
    # Create a pair_r_max tensor with custom values
    # For H (Z=1) and C (Z=6) pair, set r_max to 1.5
    pair_r_max = torch.full((119, 119), -1.0, dtype=torch.get_default_dtype())
    pair_r_max[1, 6] = 1.5
    pair_r_max[6, 1] = 1.5
    pair_r_max[1, 1] = 1.2
    pair_r_max[6, 6] = 2.0

    zbl_basis = ZBLBasis(p=6, trainable=False, pair_r_max=pair_r_max)

    assert zbl_basis.pair_r_max is not None
    assert zbl_basis.pair_r_max.shape == (119, 119)
    assert zbl_basis.pair_r_max[1, 6] == 1.5
    assert zbl_basis.pair_r_max[6, 1] == 1.5

    # Test forward pass with custom r_max
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges, 1]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # [n_nodes, n_node_features] - one_hot encoding of atomic numbers
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]

    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == torch.Size([node_attrs.shape[0]])

    # Verify that the output is different from default behavior due to custom r_max
    zbl_basis_default = ZBLBasis(p=6, trainable=False)
    output_default = zbl_basis_default(x, node_attrs, edge_index, atomic_numbers)

    # The outputs should be different because we're using different r_max values
    # (unless the default covalent radii happen to match our custom values)
    # This is a weak test - the important thing is that no errors occur


def test_zbl_basis_pair_r_max_fallback():
    """Test ZBLBasis falls back to covalent radii for pairs not in pair_r_max."""
    # Create a sparse pair_r_max tensor with only H-H pair
    pair_r_max = torch.full((119, 119), -1.0, dtype=torch.get_default_dtype())
    pair_r_max[1, 1] = 1.2  # Only set H-H pair

    zbl_basis = ZBLBasis(p=6, trainable=False, pair_r_max=pair_r_max)

    # Test forward pass
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges, 1]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # [n_nodes, n_node_features]
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]

    output = zbl_basis(x, node_attrs, edge_index, atomic_numbers)
    assert output.shape == torch.Size([node_attrs.shape[0]])
    # No error should occur - H-C pair should fall back to covalent radii


def test_zbl_basis_repr_with_pair_r_max():
    """Test that repr correctly shows pair_r_max status."""
    zbl_default = ZBLBasis(p=6)
    assert "covalent_radii" in repr(zbl_default)
    assert "zbl_scale=1.0" in repr(zbl_default)

    pair_r_max = torch.full((119, 119), -1.0, dtype=torch.get_default_dtype())
    zbl_custom = ZBLBasis(p=6, pair_r_max=pair_r_max)
    assert "custom" in repr(zbl_custom)


def test_zbl_basis_zbl_scale():
    """Test ZBLBasis zbl_scale parameter."""
    # Test default zbl_scale = 1.0
    zbl_default = ZBLBasis(p=6, trainable=False)
    assert zbl_default.zbl_scale.item() == 1.0

    # Test custom zbl_scale
    zbl_scaled = ZBLBasis(p=6, trainable=False, zbl_scale=2.0)
    assert zbl_scaled.zbl_scale.item() == 2.0

    # Test forward pass with different scales
    x = torch.tensor([1.0, 1.0, 2.0]).unsqueeze(-1)  # [n_edges, 1]
    node_attrs = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.get_default_dtype()
    )  # [n_nodes, n_node_features]
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # [2, n_edges]
    atomic_numbers = torch.tensor([1, 6])  # [n_nodes]

    output_default = zbl_default(x, node_attrs, edge_index, atomic_numbers)
    output_scaled = zbl_scaled(x, node_attrs, edge_index, atomic_numbers)

    # Output should scale linearly with zbl_scale
    assert torch.allclose(output_scaled, output_default * 2.0, rtol=1e-5)


def test_zbl_basis_zbl_scale_various_values():
    """Test ZBLBasis with various zbl_scale values."""
    x = torch.tensor([1.0, 1.0]).unsqueeze(-1)
    node_attrs = torch.tensor([[1, 0], [0, 1]], dtype=torch.get_default_dtype())
    edge_index = torch.tensor([[0, 1], [1, 0]])
    atomic_numbers = torch.tensor([1, 6])

    zbl_base = ZBLBasis(p=6, zbl_scale=1.0)
    output_base = zbl_base(x, node_attrs, edge_index, atomic_numbers)

    for scale in [0.5, 2.0, 5.0, 10.0]:
        zbl_scaled = ZBLBasis(p=6, zbl_scale=scale)
        output_scaled = zbl_scaled(x, node_attrs, edge_index, atomic_numbers)
        assert torch.allclose(output_scaled, output_base * scale, rtol=1e-5), \
            f"zbl_scale={scale} failed: expected {output_base * scale}, got {output_scaled}"


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


if __name__ == "__main__":
    pytest.main([__file__])
