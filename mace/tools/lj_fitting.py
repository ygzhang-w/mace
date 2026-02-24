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
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Dimer-based post-training bias fitting.

    For each element pair, constructs a dimer at distance lj_rcut - epsilon,
    runs full MACE inference, and computes bias so that repulsion matches
    MACE prediction at the boundary.

    Args:
        model: Trained MACE or ScaleShiftMACE model.
        data_loader: Training data loader (used to compute rcut_matrix).
        z_table: AtomicNumberTable defining element types.
        repulsion_c: Repulsion coefficient (default: 1.0).
        device: Device for inference (default: "cpu").
        dtype: Data type (default: torch.get_default_dtype()).
        epsilon: Boundary offset (default: 0.01).

    Returns:
        rcut_matrix: [num_elements, num_elements] minimum distance tensor.
            Element pairs without data get 0.0 (so boundary becomes negative,
            routing all edges to MACE).
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
    logger.info("Step 2: Building dimers and running MACE inference...")

    bias_matrix = torch.zeros((num_elements, num_elements), dtype=dtype)
    mace_dimer_energy = torch.zeros((num_elements, num_elements), dtype=dtype)

    model_device = next(model.parameters()).device
    r_max = float(model.r_max) if hasattr(model, "r_max") else 5.0
    was_training = model.training
    model.eval()

    no_data_mask = rcut_matrix >= 999.0

    with torch.no_grad():
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

                # Build dimer: atom 0 at origin, atom 1 at [r_boundary, 0, 0]
                z_i = z_table.zs[i]
                z_j = z_table.zs[j]

                config = Configuration(
                    atomic_numbers=np.array([z_i, z_j]),
                    positions=np.array([[0.0, 0.0, 0.0], [r_boundary, 0.0, 0.0]]),
                    pbc=np.array([False, False, False]),
                    cell=np.eye(3) * 100.0,
                    properties={"forces": np.zeros((2, 3)), "energy": 0.0},
                    property_weights={"forces": 1.0, "energy": 1.0},
                )
                atomic_data = AtomicData.from_config(
                    config, z_table=z_table, cutoff=r_max
                )
                loader = torch_geometric.dataloader.DataLoader(
                    dataset=[atomic_data], batch_size=1, shuffle=False
                )
                batch = next(iter(loader))
                batch_dict = batch.to_dict()
                batch_dict = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_dict.items()
                }

                output = model(batch_dict, training=False, compute_force=False)
                e_mace = output["energy"].detach().cpu().to(dtype).item()

                # bias_ij = E_mace - repulsion_c * r^{-12} * 0.5
                r_safe = max(r_boundary, 0.5)
                repulsion_term = repulsion_c * r_safe ** (-12) * 0.5
                bias_val = e_mace - repulsion_term

                bias_matrix[i, j] = bias_val
                bias_matrix[j, i] = bias_val
                mace_dimer_energy[i, j] = e_mace
                mace_dimer_energy[j, i] = e_mace

                logger.info(
                    "Pair (%d,%d) z=(%d,%d): rcut=%.4f, r_boundary=%.4f, "
                    "E_mace=%.6f, repulsion=%.6f, bias=%.6f",
                    i,
                    j,
                    z_i,
                    z_j,
                    rcut_matrix[i, j].item(),
                    r_boundary,
                    e_mace,
                    repulsion_term,
                    bias_val,
                )

    # Zero out bias for pairs with no training data
    bias_matrix[no_data_mask] = 0.0

    # Set sentinel rcut values to 0.0 so boundary becomes negative
    # and all edges for unseen pairs go to MACE (not repulsion)
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
        "num_pairs_with_data": num_pairs_found,
        "num_pairs_total": num_elements * num_elements,
        "repulsion_c": repulsion_c,
        "epsilon": epsilon,
    }

    return rcut_matrix, bias_matrix, diagnostics
