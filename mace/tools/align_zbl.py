###########################################################################################
# ZBL Alignment Utility for Exclusive Mode
# Aligns ZBL energy offsets to ensure continuity at r_max boundaries.
###########################################################################################

import logging
from typing import List

import torch

from mace.tools.utils import AtomicNumberTable


def align_exclusive_zbl(
    model: torch.nn.Module,
    z_table: AtomicNumberTable,
) -> torch.nn.Module:
    """
    Align ZBL energy offsets so energy is continuous at each pair's r_max.

    Uses isolated dimers (no periodic boundary conditions) to evaluate
    MACE and ZBL contributions at the r_max boundary for each element pair.

    For each element pair (Z_i, Z_j):
    1. Create isolated dimer at r = r_max
    2. Evaluate MACE energy contribution (with ZBL masked out)
    3. Evaluate ZBL energy at this distance (before offset)
    4. Set offset = MACE_energy - ZBL_energy

    Args:
        model: MACE model with exclusive ZBL pair repulsion
        z_table: AtomicNumberTable with atomic numbers

    Returns:
        Model with aligned ZBL energy offsets
    """
    if not hasattr(model, "pair_repulsion_type") or model.pair_repulsion_type != "exclusive":
        logging.warning("Model is not using exclusive ZBL mode, skipping alignment")
        return model

    logging.info("Aligning ZBL energy offsets for continuity at r_max boundaries...")

    device = next(model.parameters()).device
    dtype = torch.get_default_dtype()
    model.eval()

    pair_r_max_matrix = model.pair_repulsion_fn.pair_r_max_matrix
    energy_offset_matrix = torch.zeros_like(pair_r_max_matrix)

    # Get all element pairs with defined r_max
    atomic_numbers = z_table.zs

    for i, z1 in enumerate(atomic_numbers):
        for j, z2 in enumerate(atomic_numbers):
            if i > j:
                continue  # Only process upper triangle

            r_max = pair_r_max_matrix[z1, z2].item()
            if r_max <= 0:
                continue  # No r_max defined for this pair

            # Create isolated dimer at r_max
            offset = _compute_energy_offset(
                model, z1, z2, r_max, z_table, device, dtype
            )

            energy_offset_matrix[z1, z2] = offset
            energy_offset_matrix[z2, z1] = offset  # Symmetric

            logging.info(
                f"  Pair ({z1}, {z2}) at r_max={r_max:.4f} Å: offset={offset:.6f} eV"
            )

    # Update the model's energy offset matrix
    model.pair_repulsion_fn.energy_offset_matrix.copy_(energy_offset_matrix)

    model.train()
    return model


def _compute_energy_offset(
    model: torch.nn.Module,
    z1: int,
    z2: int,
    r_max: float,
    z_table: AtomicNumberTable,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Compute the energy offset needed for continuity at r_max.

    At r = r_max boundary:
    - ZBL envelope = 0, so raw ZBL contribution = 0
    - But MACE energy at r_max is NOT zero
    - We need: E_zbl(r_max) + offset * (1 - envelope) = E_mace(r_max)
    - Since envelope(r_max) = 0: offset = E_mace(r_max)

    This function evaluates the MACE model at r = r_max (with ZBL disabled)
    to get the target energy for alignment.
    """
    # Create dimer configuration at r_max distance
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [r_max, 0.0, 0.0]], dtype=dtype, device=device
    )

    # Create node attributes (one-hot encoding)
    num_elements = len(z_table)
    node_attrs = torch.zeros((2, num_elements), dtype=dtype, device=device)
    z1_idx = z_table.z_to_index(z1)
    z2_idx = z_table.z_to_index(z2)
    node_attrs[0, z1_idx] = 1.0
    node_attrs[1, z2_idx] = 1.0

    # Create edge index (bidirectional)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)

    # Create batch tensor
    batch = torch.tensor([0, 0], dtype=torch.long, device=device)

    # Create shifts (no periodic boundaries)
    shifts = torch.zeros((2, 3), dtype=dtype, device=device)

    # Prepare data dict for MACE model
    data_dict = {
        "positions": positions,
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "batch": batch,
        "shifts": shifts,
        "ptr": torch.tensor([0, 2], dtype=torch.long, device=device),
        "cell": torch.zeros((1, 3, 3), dtype=dtype, device=device),
        "head": torch.tensor([0], dtype=torch.long, device=device),
        "pbc": torch.tensor([[False, False, False]], dtype=torch.bool, device=device),
    }

    with torch.no_grad():
        # Compute MACE energy at r_max (without ZBL contribution)
        # Temporarily disable pair_repulsion to get pure MACE energy
        has_pair_repulsion = hasattr(model, "pair_repulsion")
        
        if has_pair_repulsion:
            # Save and temporarily disable pair_repulsion
            original_pair_repulsion = model.pair_repulsion
            model.pair_repulsion = False
        
        try:
            # Check if model has a proper forward method
            if hasattr(model, "forward") and callable(getattr(model, "forward", None)):
                try:
                    output = model(
                        data_dict,
                        training=False,
                        compute_force=False,
                        compute_virials=False,
                        compute_stress=False,
                    )
                    # Get total energy and subtract E0 to get interaction energy
                    # The offset should be the per-pair interaction energy at r_max
                    total_energy = output["energy"].item()
                    
                    # Get E0 (atomic energies) to isolate the interaction contribution
                    if hasattr(model, "atomic_energies_fn"):
                        node_e0 = model.atomic_energies_fn(node_attrs)
                        # Sum over atoms, taking head index 0
                        e0 = node_e0[:, 0].sum().item()
                        mace_interaction_energy = total_energy - e0
                    else:
                        # If no atomic_energies_fn, use total energy as approximation
                        mace_interaction_energy = total_energy
                    
                    # The offset should equal the MACE interaction energy at r_max
                    # The ZBL forward already applies 0.5 factor for bidirectional edges,
                    # so we don't need to divide by 2 here
                    offset = mace_interaction_energy
                    
                except (TypeError, KeyError, RuntimeError) as e:
                    # Model forward failed, fall back to zero offset
                    logging.warning(
                        f"Could not compute MACE energy for pair ({z1}, {z2}): {e}. "
                        "Using zero offset."
                    )
                    offset = 0.0
            else:
                # Model doesn't have proper forward, use zero offset
                logging.warning(
                    f"Model doesn't support forward pass for pair ({z1}, {z2}). "
                    "Using zero offset."
                )
                offset = 0.0
                
        finally:
            # Restore pair_repulsion
            if has_pair_repulsion:
                model.pair_repulsion = original_pair_repulsion

    return offset
