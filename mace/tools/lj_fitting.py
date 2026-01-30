###########################################################################################
# LJ Repulsion Fitting Module
# Ridge regression based automatic fitting for LJ repulsion coefficients
###########################################################################################

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mace.tools.scatter import scatter_sum
from mace.tools.utils import AtomicNumberTable


class LJRidgeFitter:
    """Ridge regression based LJ repulsion coefficient fitter.

    Mathematical model:
    - Energy: E = Σ_{edges} c_ij * r^{-12} * 0.5
    - Force: F = Σ_{edges} 6 * c_ij * r^{-13} * (r_vec/r)  (6 = 12 * 0.5)
    - Ridge regression: (X^T W X + αI) c = X^T W y

    Args:
        z_table: AtomicNumberTable defining element types (e.g., zs=[1,6,8])
        alpha: Ridge regression regularization parameter
        energy_weight: Weight for energy in the loss function
        forces_weight: Weight for forces in the loss function
        use_inverse_frequency_weighting: Whether to use inverse frequency weighting
            for rare atom pairs (default: True)
        r_max: Maximum distance for distance binning (default: 6.0)
        num_distance_bins: Number of distance bins for inverse frequency weighting (default: 10)
    """

    def __init__(
        self,
        z_table: AtomicNumberTable,
        alpha: float = 1.0,
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
        use_inverse_frequency_weighting: bool = True,
        r_max: float = 6.0,
        num_distance_bins: int = 10,
    ):
        self.z_table = z_table
        self.num_elements = len(z_table)
        self.alpha = alpha
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.use_inverse_frequency_weighting = use_inverse_frequency_weighting
        self.r_max = r_max
        self.num_distance_bins = num_distance_bins

        # Standardization parameters
        self.X_mean = None
        self.X_std = None
        self.is_fitted = False

        # Element pair to flat index mapping
        self.num_pairs = self.num_elements ** 2
        self.pair_to_idx: Dict[Tuple[int, int], int] = {}
        self.idx_to_pair: Dict[int, Tuple[int, int]] = {}

        for i in range(self.num_elements):
            for j in range(self.num_elements):
                idx = i * self.num_elements + j
                self.pair_to_idx[(i, j)] = idx
                self.idx_to_pair[idx] = (i, j)

    def _get_distance_bin(
        self, r: float, r_min: float, r_max: float
    ) -> int:
        """Map distance to logarithmic bin index.

        Uses logarithmic binning so that smaller distances have finer resolution.

        Args:
            r: Distance value
            r_min: Minimum distance (lower bound for this pair type)
            r_max: Maximum distance (upper bound)

        Returns:
            bin_idx: Bin index (0 to num_distance_bins - 1)
        """
        if r <= r_min:
            return 0
        if r >= r_max:
            return self.num_distance_bins - 1

        # Logarithmic binning
        log_r = np.log(r)
        log_min = np.log(max(r_min, 1e-6))  # Avoid log(0)
        log_max = np.log(r_max)

        if log_max <= log_min:
            return 0

        bin_idx = int(
            (log_r - log_min) / (log_max - log_min) * self.num_distance_bins
        )
        return min(bin_idx, self.num_distance_bins - 1)

    def preprocess_data(
        self,
        data_loader,
        atomic_energies: np.ndarray,
        compute_forces: bool = True,
    ) -> Dict:
        """Preprocess training data to extract features and targets.

        Uses two-pass scanning:
        - Pass 1: Collect r_min for each pair type
        - Pass 2: Build features and compute (pair, bin) statistics

        Args:
            data_loader: Training data loader
            atomic_energies: np.ndarray [num_heads, num_elements]
                Atomic energies from configure_model(), indexed by z_table order
            compute_forces: Whether to include force features

        Returns:
            Dictionary containing:
            - X_energy: [num_configs, num_pairs] energy features
            - y_energy: [num_configs] energy targets
            - X_forces: [num_force_samples, num_pairs] force features (if compute_forces)
            - y_forces: [num_force_samples] force targets (if compute_forces)
            - r_min_dict: Dict[(i,j), float] minimum distances per element pair
            - pair_bin_counts: Dict[(pair_idx, bin_idx), int] counts per (pair, bin)
            - config_weights: [num_configs] configuration-level inverse frequency weights
            - energy_weights: [num_configs] same as config_weights (for energy samples)
            - force_weights: [num_force_samples] expanded from config_weights
            - force_samples_per_config: [num_configs] number of force samples per config
        """
        # Get atomic energies for first head (single-head assumed)
        if atomic_energies.ndim == 2:
            atomic_e = torch.tensor(atomic_energies[0], dtype=torch.float64)
        else:
            atomic_e = torch.tensor(atomic_energies, dtype=torch.float64)

        # ============= Pass 1: Collect r_min for each pair type =============
        r_min_dict: Dict[Tuple[int, int], float] = {}

        for batch in data_loader:
            positions = batch.positions
            edge_index = batch.edge_index
            node_attrs = batch.node_attrs

            sender = edge_index[0]
            receiver = edge_index[1]
            edge_vec = positions[receiver] - positions[sender]
            edge_length = torch.norm(edge_vec, dim=1)

            element_indices = torch.argmax(node_attrs, dim=1)
            sender_z = element_indices[sender]
            receiver_z = element_indices[receiver]

            for e in range(len(edge_length)):
                pair = (sender_z[e].item(), receiver_z[e].item())
                r = edge_length[e].item()
                if pair not in r_min_dict:
                    r_min_dict[pair] = r
                else:
                    r_min_dict[pair] = min(r_min_dict[pair], r)

        # ============= Pass 2: Build features and statistics =============
        X_energy_list = []
        y_energy_list = []
        X_forces_list = []
        y_forces_list = []

        # Track (pair_idx, bin_idx) counts for inverse frequency
        pair_bin_counts: Dict[Tuple[int, int], int] = {}

        # Track edge info per configuration for weight calculation
        config_edge_info: List[List[Tuple[int, int]]] = []  # [(pair_idx, bin_idx), ...]
        force_samples_per_config: List[int] = []  # Number of force samples per config

        for batch in data_loader:
            positions = batch.positions
            edge_index = batch.edge_index
            node_attrs = batch.node_attrs

            sender = edge_index[0]
            receiver = edge_index[1]
            edge_vec = positions[receiver] - positions[sender]
            edge_length = torch.norm(edge_vec, dim=1)

            element_indices = torch.argmax(node_attrs, dim=1)
            sender_z = element_indices[sender]
            receiver_z = element_indices[receiver]

            # Compute r^{-12} features (with safety clamp)
            r_safe = torch.clamp(edge_length, min=0.5)
            r_inv_12 = torch.pow(r_safe, -12) * 0.5

            # Get pair indices for each edge
            pair_idx = sender_z * self.num_elements + receiver_z

            batch_idx = batch.batch
            edge_batch = batch_idx[receiver]
            num_graphs = batch_idx.max().item() + 1

            # Compute bin indices for each edge
            edge_bins = []
            for e in range(len(edge_length)):
                pair = (sender_z[e].item(), receiver_z[e].item())
                r = edge_length[e].item()
                r_min = r_min_dict.get(pair, 0.5)
                bin_idx = self._get_distance_bin(r, r_min, self.r_max)
                edge_bins.append(bin_idx)

                # Update (pair, bin) counts
                p_idx = pair[0] * self.num_elements + pair[1]
                key = (p_idx, bin_idx)
                pair_bin_counts[key] = pair_bin_counts.get(key, 0) + 1

            edge_bins = torch.tensor(edge_bins, dtype=torch.long)

            # Create feature matrix for this batch
            X_batch = torch.zeros(num_graphs, self.num_pairs, dtype=torch.float64)
            for g in range(num_graphs):
                graph_mask = edge_batch == g
                for p in range(self.num_pairs):
                    pair_mask = pair_idx == p
                    combined_mask = graph_mask & pair_mask
                    X_batch[g, p] = r_inv_12[combined_mask].sum()

                # Collect edge info for this configuration
                graph_pair_idx = pair_idx[graph_mask].tolist()
                graph_bins = edge_bins[graph_mask].tolist()
                config_edge_info.append(list(zip(graph_pair_idx, graph_bins)))

            # Compute target energies (subtract atomic energies)
            y_batch = torch.zeros(num_graphs, dtype=torch.float64)
            for g in range(num_graphs):
                graph_atom_mask = batch_idx == g
                graph_element_idx = element_indices[graph_atom_mask]
                atomic_energy_sum = atomic_e[graph_element_idx].sum()

                if hasattr(batch, "energy"):
                    graph_energy = batch.energy
                    if graph_energy.dim() == 0:
                        y_batch[g] = graph_energy - atomic_energy_sum
                    else:
                        y_batch[g] = graph_energy[g] - atomic_energy_sum

            X_energy_list.append(X_batch)
            y_energy_list.append(y_batch)

            # Build force features if requested
            if compute_forces and hasattr(batch, "forces") and batch.forces is not None:
                forces = batch.forces
                r_inv_13 = torch.pow(r_safe, -13)
                r_unit = edge_vec / r_safe.unsqueeze(1)

                # Count force samples per graph in this batch
                for g in range(num_graphs):
                    graph_atom_mask = batch_idx == g
                    num_atoms_in_graph = graph_atom_mask.sum().item()
                    # Each atom contributes 3 force components (x, y, z)
                    # But we only count atoms that have edges
                    force_count = 0

                    for atom_idx in range(positions.shape[0]):
                        if batch_idx[atom_idx] != g:
                            continue
                        atom_mask = receiver == atom_idx
                        if not atom_mask.any():
                            continue

                        for dim in range(3):
                            feat_values = 6 * r_inv_13[atom_mask] * r_unit[atom_mask, dim]
                            pair_idx_atom = pair_idx[atom_mask]

                            X_force = torch.zeros(self.num_pairs, dtype=torch.float64)
                            for p in range(self.num_pairs):
                                p_mask = pair_idx_atom == p
                                X_force[p] = feat_values[p_mask].sum()

                            X_forces_list.append(X_force)
                            y_forces_list.append(forces[atom_idx, dim].item())
                            force_count += 1

                    force_samples_per_config.append(force_count)
            else:
                # No forces, but still need to track for config alignment
                for g in range(num_graphs):
                    force_samples_per_config.append(0)

        # Stack all features and targets
        result = {
            "X_energy": torch.cat(X_energy_list, dim=0),
            "y_energy": torch.cat(y_energy_list, dim=0),
            "r_min_dict": r_min_dict,
            "pair_bin_counts": pair_bin_counts,
            "force_samples_per_config": force_samples_per_config,
        }

        if compute_forces and X_forces_list:
            result["X_forces"] = torch.stack(X_forces_list, dim=0)
            result["y_forces"] = torch.tensor(y_forces_list, dtype=torch.float64)

        # ============= Compute inverse frequency weights (configuration-level) =============
        if self.use_inverse_frequency_weighting and pair_bin_counts:
            # Compute inverse frequency for each (pair, bin) combination
            inv_freq_table: Dict[Tuple[int, int], float] = {}
            for key, count in pair_bin_counts.items():
                inv_freq_table[key] = 1.0 / count if count > 0 else 0.0

            # Compute configuration weights (sum of edge weights)
            config_weights = []
            for edges in config_edge_info:
                if edges:
                    weight = sum(inv_freq_table.get(key, 0.0) for key in edges)
                else:
                    weight = 1.0
                config_weights.append(weight)

            # Global normalization at configuration level
            total_weight = sum(config_weights)
            if total_weight > 0:
                num_configs = len(config_weights)
                config_weights = [
                    w / total_weight * num_configs for w in config_weights
                ]

            result["config_weights"] = torch.tensor(config_weights, dtype=torch.float64)
            result["energy_weights"] = result["config_weights"].clone()

            # Expand force weights from config weights
            # Same config's energy and all force samples share the same weight
            if compute_forces and force_samples_per_config:
                force_weights = []
                for config_idx, num_force in enumerate(force_samples_per_config):
                    force_weights.extend([config_weights[config_idx]] * num_force)
                result["force_weights"] = torch.tensor(force_weights, dtype=torch.float64)

        return result

    def _standardize_X(
        self, X: torch.Tensor, fit: bool = True
    ) -> torch.Tensor:
        """Apply Z-score standardization to feature matrix.

        Args:
            X: [num_samples, num_pairs] feature matrix
            fit: Whether to compute and save mean and std (True for training)

        Returns:
            X_scaled: Standardized feature matrix
        """
        if fit:
            # Compute mean and std for each column
            self.X_mean = torch.mean(X, dim=0, keepdim=True)  # [1, num_pairs]
            # Use unbiased=False to avoid NaN for single sample
            self.X_std = torch.std(X, dim=0, keepdim=True, unbiased=False)    # [1, num_pairs]

            # Prevent division by zero, use 1 for columns with zero std
            self.X_std = torch.where(
                self.X_std < 1e-10,
                torch.ones_like(self.X_std),
                self.X_std
            )

        # Apply standardization
        X_scaled = (X - self.X_mean) / self.X_std

        return X_scaled

    def _solve_ridge(
        self, X: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve ridge regression: (X^T W X + αI) c = X^T W y

        Args:
            X: [num_samples, num_pairs] feature matrix
            y: [num_samples] target vector
            weights: [num_samples] sample weights (optional)

        Returns:
            c: [num_pairs] coefficient vector
        """
        if weights is None:
            weights = torch.ones(X.shape[0], dtype=X.dtype)

        # Weighted normal equation
        W_sqrt = torch.sqrt(weights).unsqueeze(1)
        X_weighted = X * W_sqrt
        y_weighted = y * torch.sqrt(weights)

        # Solve (X^T W X + αI) c = X^T W y
        XtX = X_weighted.T @ X_weighted
        ridge_term = self.alpha * torch.eye(X.shape[1], dtype=X.dtype)
        A = XtX + ridge_term

        Xty = X_weighted.T @ y_weighted

        # Solve linear system
        c = torch.linalg.solve(A, Xty)

        return c

    def fit(
        self,
        data_loader,
        atomic_energies: np.ndarray,
        compute_forces: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Main fitting routine.

        Weight application order:
        1. Compute configuration-level inverse frequency weights (based on pair-bin distribution)
        2. Multiply by energy_weight / forces_weight

        Args:
            data_loader: Training data loader
            atomic_energies: np.ndarray [num_heads, num_elements]
            compute_forces: Whether to use forces in fitting

        Returns:
            coeff_matrix: [num_elements, num_elements] coefficient matrix
            diagnostics: Dictionary with fitting diagnostics, including:
                - mse: Mean squared error
                - r2: R-squared score
                - r_min_dict: Minimum distances per element pair
                - num_energy_samples: Number of energy samples
                - num_force_samples: Number of force samples
                - X_mean: Feature means for standardization
                - X_std: Feature standard deviations for standardization
                - pair_bin_counts: Counts of each (pair, bin) combination
                - use_inverse_frequency_weighting: Whether inverse frequency weighting was used
        """
        # Preprocess data
        data = self.preprocess_data(data_loader, atomic_energies, compute_forces)

        X_energy = data["X_energy"]
        y_energy = data["y_energy"]
        num_energy_samples = X_energy.shape[0]

        # Build combined feature matrix and target vector
        # Note: Do NOT multiply by sqrt(weight) here - weights are applied through sample_weights
        if compute_forces and "X_forces" in data:
            X_forces = data["X_forces"]
            y_forces = data["y_forces"]
            num_force_samples = X_forces.shape[0]

            X_combined = torch.cat([X_energy, X_forces], dim=0)
            y_combined = torch.cat([y_energy, y_forces], dim=0)
        else:
            X_combined = X_energy
            y_combined = y_energy
            num_force_samples = 0

        # Apply standardization to X_combined
        X_combined_scaled = self._standardize_X(X_combined, fit=True)

        # Build sample weights: first inverse frequency, then multiply by energy/force weight
        if self.use_inverse_frequency_weighting and "energy_weights" in data:
            inv_freq_energy_weights = data["energy_weights"]
            if compute_forces and "force_weights" in data:
                inv_freq_force_weights = data["force_weights"]
            else:
                inv_freq_force_weights = torch.tensor([], dtype=torch.float64)
        else:
            inv_freq_energy_weights = torch.ones(num_energy_samples, dtype=torch.float64)
            inv_freq_force_weights = torch.ones(num_force_samples, dtype=torch.float64)

        # Apply energy_weight and forces_weight after inverse frequency weighting
        final_energy_weights = inv_freq_energy_weights * self.energy_weight
        final_force_weights = inv_freq_force_weights * self.forces_weight

        # Combine weights
        if compute_forces and num_force_samples > 0:
            sample_weights = torch.cat([final_energy_weights, final_force_weights], dim=0)
        else:
            sample_weights = final_energy_weights

        # Solve ridge regression
        c_vector = self._solve_ridge(X_combined_scaled, y_combined, weights=sample_weights)

        # Mark as fitted
        self.is_fitted = True

        # Apply non-negative constraint
        c_vector = torch.clamp(c_vector, min=0.0)

        # Convert to matrix form
        coeff_matrix = c_vector.reshape(self.num_elements, self.num_elements)

        # Ensure symmetry
        coeff_matrix = (coeff_matrix + coeff_matrix.T) / 2

        # Compute diagnostics
        y_pred = X_combined @ c_vector
        residuals = y_combined - y_pred
        mse = torch.mean(residuals**2).item()
        ss_tot = torch.sum((y_combined - y_combined.mean()) ** 2).item()
        ss_res = torch.sum(residuals**2).item()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        diagnostics = {
            "mse": mse,
            "r2": r2,
            "r_min_dict": data["r_min_dict"],
            "num_energy_samples": num_energy_samples,
            "num_force_samples": num_force_samples,
            "X_mean": self.X_mean.numpy() if self.X_mean is not None else None,
            "X_std": self.X_std.numpy() if self.X_std is not None else None,
            "pair_bin_counts": data.get("pair_bin_counts", {}),
            "use_inverse_frequency_weighting": self.use_inverse_frequency_weighting,
            "r_max": self.r_max,
            "num_distance_bins": self.num_distance_bins,
        }

        return coeff_matrix.to(torch.get_default_dtype()), diagnostics

    @staticmethod
    def fit_lj_coefficients(
        data_loader,
        z_table: AtomicNumberTable,
        atomic_energies: np.ndarray,
        alpha: float = 1.0,
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
        compute_forces: bool = True,
        use_inverse_frequency_weighting: bool = True,
        r_max: float = 6.0,
        num_distance_bins: int = 10,
    ) -> Tuple[torch.Tensor, Dict]:
        """Convenience function for LJ coefficient fitting.

        Args:
            data_loader: Training data loader
            z_table: AtomicNumberTable defining element types
            atomic_energies: np.ndarray [num_heads, num_elements]
            alpha: Ridge regularization parameter
            energy_weight: Weight for energy loss
            forces_weight: Weight for forces loss
            compute_forces: Whether to use forces in fitting
            use_inverse_frequency_weighting: Whether to use inverse frequency weighting
                for rare atom pairs (default: True)
            r_max: Maximum distance for distance binning (default: 6.0)
            num_distance_bins: Number of distance bins (default: 10)

        Returns:
            coeff_matrix: [num_elements, num_elements] coefficient matrix
            diagnostics: Dictionary with fitting diagnostics
        """
        fitter = LJRidgeFitter(
            z_table=z_table,
            alpha=alpha,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            use_inverse_frequency_weighting=use_inverse_frequency_weighting,
            r_max=r_max,
            num_distance_bins=num_distance_bins,
        )
        return fitter.fit(data_loader, atomic_energies, compute_forces)
