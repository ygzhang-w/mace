###########################################################################################
# LJ Repulsion Post-Training Bias Fitting Module
# Computes per-element-pair bias by constructing dimers at the boundary distance
# and using MACE inference for alignment.
###########################################################################################

import logging
from typing import Dict, Tuple

import torch

from mace.tools.utils import AtomicNumberTable

logger = logging.getLogger(__name__)


def compute_lj_rcut_matrix(
    data_loader,
    z_table: AtomicNumberTable,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Scan training data to find minimum inter-atomic distance per element pair.

    Args:
        data_loader: Iterable of batch data (torch_geometric Batch objects).
        z_table: AtomicNumberTable defining element types.

    Returns:
        rcut_matrix: [num_elements, num_elements] tensor of minimum distances.
            Element pairs not found in training data get sentinel value 999.0.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    num_elements = len(z_table)
    rcut_matrix = torch.full((num_elements, num_elements), 999.0, dtype=dtype)

    for batch in data_loader:
        positions = batch.positions
        edge_index = batch.edge_index
        node_attrs = batch.node_attrs

        sender = edge_index[0]
        receiver = edge_index[1]
        edge_vec = positions[receiver] - positions[sender]

        # Handle periodic boundary conditions
        if hasattr(batch, "shifts") and batch.shifts is not None:
            edge_vec = edge_vec + batch.shifts

        edge_length = torch.norm(edge_vec, dim=1)

        element_indices = torch.argmax(node_attrs, dim=1)
        sender_z = element_indices[sender]
        receiver_z = element_indices[receiver]

        for e in range(len(edge_length)):
            i = sender_z[e].item()
            j = receiver_z[e].item()
            r = edge_length[e].item()
            if r < rcut_matrix[i, j].item():
                rcut_matrix[i, j] = r

    logger.info(
        "Computed rcut matrix: min=%.4f, max finite=%.4f",
        (
            rcut_matrix[rcut_matrix < 999.0].min().item()
            if (rcut_matrix < 999.0).any()
            else 0.0
        ),
        (
            rcut_matrix[rcut_matrix < 999.0].max().item()
            if (rcut_matrix < 999.0).any()
            else 0.0
        ),
    )

    return rcut_matrix


def fit_lj_repulsion_bias(
    model,
    data_loader,
    z_table: AtomicNumberTable,
    repulsion_c: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = None,
    epsilon: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Dimer-based post-training fitting for per-pair c_ij and bias_ij.

    For each element pair, constructs a dimer at distance lj_rcut - epsilon,
    runs MACE inference with force computation, and analytically solves for
    c_ij (from force continuity) and bias_ij (from energy continuity).

    Math:
        c_ij = F_proj * r_b^{13} / 6
        bias_ij = delta_E / 2 - c_ij * r_b^{-12} * 0.5

    Args:
        model: Trained MACE or ScaleShiftMACE model.
        data_loader: Training data loader (used to compute rcut_matrix).
        z_table: AtomicNumberTable defining element types.
        repulsion_c: Default repulsion constant for pairs without data (default: 1.0).
        device: Device for inference (default: "cpu").
        dtype: Data type (default: torch.get_default_dtype()).
        epsilon: Boundary offset (default: 0.5).

    Returns:
        rcut_matrix: [num_elements, num_elements] minimum distance tensor.
        c_matrix: [num_elements, num_elements] per-pair repulsion constants.
        bias_matrix: [num_elements, num_elements] bias tensor.
        diagnostics: Dictionary with fitting diagnostics.
    """
    import numpy as np

    from mace.data import AtomicData, Configuration
    from mace.tools import torch_geometric

    if dtype is None:
        dtype = torch.get_default_dtype()
    num_elements = len(z_table)

    # Step 1: Scan training data for minimum distances per element pair
    logger.info("Step 1: Scanning data for minimum inter-atomic distances...")
    rcut_matrix = compute_lj_rcut_matrix(data_loader, z_table, dtype=dtype)

    # Step 2: Build dimers and run MACE inference for alignment
    logger.info(
        "Step 2: Building dimers and running MACE inference (energy + force)..."
    )

    c_matrix = torch.full((num_elements, num_elements), repulsion_c, dtype=dtype)
    bias_matrix = torch.zeros((num_elements, num_elements), dtype=dtype)
    mace_dimer_energy = torch.zeros((num_elements, num_elements), dtype=dtype)
    mace_dimer_force = torch.zeros((num_elements, num_elements), dtype=dtype)

    model_device = next(model.parameters()).device
    r_max = float(model.r_max) if hasattr(model, "r_max") else 5.0
    was_training = model.training
    model.eval()

    no_data_mask = rcut_matrix >= 999.0

    # Enable the split logic so dimer inference at boundary + delta
    # is handled purely by MACE (no repulsion contribution).
    fitting_rcut = rcut_matrix.clone()
    fitting_rcut[no_data_mask] = 0.0
    old_has_lj_rcut = model.has_lj_rcut
    model.lj_rcut_matrix.copy_(fitting_rcut.to(model_device))
    model.has_lj_rcut = True

    # Small offset to ensure the dimer edge lands on the MACE side
    boundary_delta = max(1e-5, float(torch.finfo(torch.float32).eps))

    def _run_dimer(z_i, z_j, distance, compute_force=False):
        """Run MACE inference on a dimer and return (energy, forces)."""
        config = Configuration(
            atomic_numbers=np.array([z_i, z_j]),
            positions=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            pbc=np.array([False, False, False]),
            cell=np.eye(3) * 100.0,
            properties={"forces": np.zeros((2, 3)), "energy": 0.0},
            property_weights={"forces": 1.0, "energy": 1.0},
        )
        atomic_data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[atomic_data], batch_size=1, shuffle=False
        )
        batch = next(iter(loader))
        batch_dict = batch.to_dict()
        batch_dict = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }
        output = model(batch_dict, training=False, compute_force=compute_force)
        energy = output["energy"].detach().cpu().to(dtype).item()
        forces = None
        if compute_force and output["forces"] is not None:
            forces = output["forces"].detach().cpu().to(dtype)
        return energy, forces

    # Distance beyond cutoff: no edges, gives E_no_edge = e0 + N*shift
    r_far = r_max + 1.0

    for i in range(num_elements):
        for j in range(i, num_elements):
            if no_data_mask[i, j]:
                continue

            r_boundary = rcut_matrix[i, j].item() - epsilon

            if r_boundary <= 0:
                logger.warning(
                    "Pair (%d, %d) has rcut=%.4f <= epsilon=%.4f, skipping",
                    i,
                    j,
                    rcut_matrix[i, j].item(),
                    epsilon,
                )
                continue

            z_i = z_table.zs[i]
            z_j = z_table.zs[j]

            # Dimer at boundary + delta: edge is on MACE side (no repulsion)
            # compute_force=True for force continuity fitting
            e_with_edge, forces_with = _run_dimer(
                z_i, z_j, r_boundary + boundary_delta, compute_force=True
            )
            # Dimer beyond cutoff: no edges at all (baseline e0 + N*shift)
            e_no_edge, _ = _run_dimer(z_i, z_j, r_far, compute_force=False)

            # Extract F_proj: atom 1's x-component of force
            f_proj = forces_with[1, 0].item()

            # Solve analytically:
            #   Force continuity: F_proj = 6 * c_ij * r_b^{-13}
            #   Energy continuity: delta_E / 2 = c_ij * r_b^{-12} * 0.5 + bias_ij
            delta_e = e_with_edge - e_no_edge
            r_safe = max(r_boundary, 0.5)
            c_val = f_proj * (r_safe**13) / 6.0
            bias_val = delta_e / 2.0 - c_val * (r_safe**-12) * 0.5

            if c_val < 0:
                raise ValueError(
                    f"Negative c_ij={c_val:.6f} for element pair ({i},{j}) "
                    f"z=({z_i},{z_j}) at r_boundary={r_boundary:.4f}. "
                    f"This means MACE predicts attractive force at the boundary. "
                    f"Try reducing --lj_rcut_epsilon (current: {epsilon})."
                )

            c_matrix[i, j] = c_val
            c_matrix[j, i] = c_val
            bias_matrix[i, j] = bias_val
            bias_matrix[j, i] = bias_val
            mace_dimer_energy[i, j] = e_with_edge
            mace_dimer_energy[j, i] = e_with_edge
            mace_dimer_force[i, j] = f_proj
            mace_dimer_force[j, i] = f_proj

            logger.info(
                "Pair (%d,%d) z=(%d,%d): rcut=%.4f, r_boundary=%.4f, "
                "E_with=%.6f, E_no=%.6f, delta=%.6f, F_proj=%.6f, "
                "c_ij=%.6f, bias=%.6f",
                i,
                j,
                z_i,
                z_j,
                rcut_matrix[i, j].item(),
                r_boundary,
                e_with_edge,
                e_no_edge,
                delta_e,
                f_proj,
                c_val,
                bias_val,
            )

    # Restore model state before setting final values
    model.has_lj_rcut = old_has_lj_rcut

    # Zero out bias for pairs with no training data;
    # keep c_matrix at repulsion_c default for no-data pairs
    bias_matrix[no_data_mask] = 0.0

    # Set sentinel rcut values to 0.0 so boundary becomes negative
    rcut_matrix[no_data_mask] = 0.0

    if was_training:
        model.train()

    num_pairs_found = int((~no_data_mask).sum().item())
    logger.info(
        "Found training data for %d / %d element pairs",
        num_pairs_found,
        num_elements * num_elements,
    )

    diagnostics = {
        "rcut_matrix": rcut_matrix,
        "mace_dimer_energy": mace_dimer_energy,
        "mace_dimer_force": mace_dimer_force,
        "num_pairs_with_data": num_pairs_found,
        "num_pairs_total": num_elements * num_elements,
        "repulsion_c": repulsion_c,
        "epsilon": epsilon,
    }

    return rcut_matrix, c_matrix, bias_matrix, diagnostics
