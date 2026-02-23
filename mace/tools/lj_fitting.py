###########################################################################################
# LJ Repulsion Post-Training Bias Fitting Module
# Computes per-element-pair bias by analyzing minimum inter-atomic distances
# and MACE total energies at those configurations.
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
    rcut_matrix = torch.full(
        (num_elements, num_elements), 999.0, dtype=dtype
    )

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

    logger.info("Computed rcut matrix: min=%.4f, max finite=%.4f",
                rcut_matrix[rcut_matrix < 999.0].min().item() if (rcut_matrix < 999.0).any() else 0.0,
                rcut_matrix[rcut_matrix < 999.0].max().item() if (rcut_matrix < 999.0).any() else 0.0)

    return rcut_matrix


def compute_bias_from_rcut(
    rcut_matrix: torch.Tensor,
    mace_total_energy: torch.Tensor,
    repulsion_c: float,
) -> torch.Tensor:
    """Compute bias matrix from rcut distances and MACE total energies.

    Formula: bias_ij = mace_total_energy[i,j] - repulsion_c * rcut[i,j]^{-12} * 0.5

    Args:
        rcut_matrix: [num_elements, num_elements] minimum distance per pair.
        mace_total_energy: [num_elements, num_elements] MACE total energy
            of the configuration containing the shortest-distance edge.
        repulsion_c: Repulsion coefficient.

    Returns:
        bias_matrix: [num_elements, num_elements] bias tensor.
    """
    rcut_safe = torch.clamp(rcut_matrix, min=0.5)
    repulsion_term = repulsion_c * torch.pow(rcut_safe, -12) * 0.5
    bias_matrix = mace_total_energy - repulsion_term

    logger.info("Computed bias matrix: min=%.6f, max=%.6f",
                bias_matrix.min().item(), bias_matrix.max().item())

    return bias_matrix


def fit_lj_repulsion_bias(
    model,
    data_loader,
    z_table: AtomicNumberTable,
    repulsion_c: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Main post-training bias fitting function.

    Finds the minimum inter-atomic distance per element pair, runs MACE inference
    on the corresponding batches to obtain configuration total energy, and computes bias values.

    Args:
        model: Trained MACE or ScaleShiftMACE model.
        data_loader: Training data loader.
        z_table: AtomicNumberTable defining element types.
        repulsion_c: Repulsion coefficient (default: 1.0).
        device: Device for inference (default: "cpu").

    Returns:
        rcut_matrix: [num_elements, num_elements] minimum distance tensor.
        bias_matrix: [num_elements, num_elements] bias tensor.
        diagnostics: Dictionary with fitting diagnostics.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    num_elements = len(z_table)

    # Step 1: Find minimum distance per element pair and track which batch has it
    rcut_matrix = torch.full(
        (num_elements, num_elements), 999.0, dtype=dtype
    )
    # Track the batch data and edge index for the shortest distance per pair
    best_batch_for_pair: Dict[Tuple[int, int], object] = {}
    best_edge_for_pair: Dict[Tuple[int, int], int] = {}

    logger.info("Step 1: Scanning data for minimum inter-atomic distances...")

    for batch in data_loader:
        positions = batch.positions
        edge_index = batch.edge_index
        node_attrs = batch.node_attrs

        sender = edge_index[0]
        receiver = edge_index[1]
        edge_vec = positions[receiver] - positions[sender]

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
                best_batch_for_pair[(i, j)] = batch
                best_edge_for_pair[(i, j)] = e

    # Step 2 & 3: Run MACE inference on batches containing shortest distances
    logger.info("Step 2-3: Running MACE inference on relevant batches...")

    mace_total_energy = torch.zeros(
        (num_elements, num_elements), dtype=dtype
    )

    model_device = next(model.parameters()).device
    model.eval()

    # Collect unique batches to avoid redundant inference
    batch_to_pairs: Dict[int, list] = {}
    for pair, batch_obj in best_batch_for_pair.items():
        batch_id = id(batch_obj)
        if batch_id not in batch_to_pairs:
            batch_to_pairs[batch_id] = []
        batch_to_pairs[batch_id].append(pair)

    # Map batch id to batch object
    batch_id_to_obj = {id(b): b for b in best_batch_for_pair.values()}

    with torch.no_grad():
        for batch_id, pairs in batch_to_pairs.items():
            batch_data = batch_id_to_obj[batch_id]
            batch_on_device = batch_data.to(model_device)

            output = model(batch_on_device, training=False, compute_force=False)
            energy = output["energy"].detach().cpu().to(dtype)

            for pair in pairs:
                edge_idx = best_edge_for_pair[pair]
                # Find which graph in the batch this edge belongs to
                receiver_atom = batch_data.edge_index[1, edge_idx].item()
                graph_idx = batch_data.batch[receiver_atom].item()
                mace_total_energy[pair[0], pair[1]] = energy[graph_idx]

    logger.info("MACE total energies at rcut configs: min=%.6f, max=%.6f",
                mace_total_energy.min().item(), mace_total_energy.max().item())

    # Step 5: Compute bias matrix
    bias_matrix = compute_bias_from_rcut(rcut_matrix, mace_total_energy, repulsion_c)

    # Step 6: Zero out bias for pairs with no training data
    no_data_mask = rcut_matrix >= 999.0
    bias_matrix[no_data_mask] = 0.0

    num_pairs_found = int((~no_data_mask).sum().item())
    logger.info("Found training data for %d / %d element pairs",
                num_pairs_found, num_elements * num_elements)

    diagnostics = {
        "rcut_matrix": rcut_matrix,
        "mace_total_energy": mace_total_energy,
        "num_pairs_with_data": num_pairs_found,
        "num_pairs_total": num_elements * num_elements,
        "repulsion_c": repulsion_c,
    }

    return rcut_matrix, bias_matrix, diagnostics
