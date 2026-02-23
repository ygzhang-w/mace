###########################################################################################
# LJ Repulsion Post-Training Bias Fitting Module
# Computes per-element-pair bias by analyzing minimum inter-atomic distances
# and MACE total energies at those configurations.
###########################################################################################

import logging
from typing import Dict, List, Tuple

import numpy as np
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


class LJRidgeFitter:
    """Ridge regression fitter for LJ repulsion coefficients.

    Mathematical model:
    - Energy: E = Σ_{edges} (c_ij * r^{-12} * 0.5 + bias_ij)
    - Ridge regression: (X^T W X + αI) c = X^T W y

    Args:
        z_table: AtomicNumberTable defining element types
        alpha: Ridge regression regularization parameter
        energy_weight: Weight for energy in the loss function
        use_inverse_frequency_weighting: Whether to weight by inverse (pair, bin) frequency
        r_max: Maximum distance for distance binning
        num_distance_bins: Number of logarithmic distance bins
    """

    def __init__(
        self,
        z_table: AtomicNumberTable,
        alpha: float = 1.0,
        energy_weight: float = 1.0,
        use_inverse_frequency_weighting: bool = True,
        r_max: float = 6.0,
        num_distance_bins: int = 10,
    ):
        self.z_table = z_table
        self.num_elements = len(z_table)
        self.alpha = alpha
        self.energy_weight = energy_weight
        self.use_inverse_frequency_weighting = use_inverse_frequency_weighting
        self.r_max = r_max
        self.num_distance_bins = num_distance_bins
        self.num_pairs = self.num_elements ** 2

        self.pair_to_idx: Dict[Tuple[int, int], int] = {}
        for i in range(self.num_elements):
            for j in range(self.num_elements):
                self.pair_to_idx[(i, j)] = i * self.num_elements + j

    def _get_distance_bin(self, r: float, r_min: float, r_max: float) -> int:
        """Map distance to logarithmic bin index."""
        if r <= r_min:
            return 0
        if r >= r_max:
            return self.num_distance_bins - 1
        log_r = np.log(r)
        log_min = np.log(max(r_min, 1e-6))
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
    ) -> Dict:
        """Build feature matrix and targets from all training configs.

        Feature matrix has 2 * num_pairs columns:
        - First num_pairs: Σ r^{-12} * 0.5 per element pair (c_ij features)
        - Last num_pairs: N_{ij} edge count per element pair (bias features)

        Returns dict with X_energy, y_energy, energy_weights, r_min_dict, pair_bin_counts.
        """
        if atomic_energies.ndim == 2:
            atomic_e = torch.tensor(atomic_energies[0], dtype=torch.float64)
        else:
            atomic_e = torch.tensor(atomic_energies, dtype=torch.float64)

        # Pass 1: collect r_min per pair
        r_min_dict: Dict[Tuple[int, int], float] = {}
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
                pair = (sender_z[e].item(), receiver_z[e].item())
                r = edge_length[e].item()
                if pair not in r_min_dict or r < r_min_dict[pair]:
                    r_min_dict[pair] = r

        # Pass 2: build features
        X_energy_list = []
        y_energy_list = []
        pair_bin_counts: Dict[Tuple[int, int], int] = {}
        config_edge_info: List[List[Tuple[int, int]]] = []

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

            r_safe = torch.clamp(edge_length, min=0.5)
            r_inv_12 = torch.pow(r_safe, -12) * 0.5
            pair_idx = sender_z * self.num_elements + receiver_z

            batch_idx = batch.batch
            edge_batch = batch_idx[receiver]
            num_graphs = batch_idx.max().item() + 1

            # Bin indices for weighting
            edge_bins = []
            for e in range(len(edge_length)):
                pair = (sender_z[e].item(), receiver_z[e].item())
                r = edge_length[e].item()
                r_min = r_min_dict.get(pair, 0.5)
                bin_idx = self._get_distance_bin(r, r_min, self.r_max)
                edge_bins.append(bin_idx)
                p_idx = pair[0] * self.num_elements + pair[1]
                key = (p_idx, bin_idx)
                pair_bin_counts[key] = pair_bin_counts.get(key, 0) + 1

            # Feature matrix: [num_graphs, 2 * num_pairs]
            X_batch = torch.zeros(num_graphs, 2 * self.num_pairs, dtype=torch.float64)
            for g in range(num_graphs):
                graph_mask = edge_batch == g
                for p in range(self.num_pairs):
                    pair_mask = pair_idx == p
                    combined_mask = graph_mask & pair_mask
                    # c_ij feature: sum of r^{-12} * 0.5
                    X_batch[g, p] = r_inv_12[combined_mask].sum()
                    # bias feature: edge count
                    X_batch[g, self.num_pairs + p] = combined_mask.sum().float()

                graph_pair_indices = pair_idx[graph_mask].tolist()
                graph_bins = [edge_bins[idx] for idx in range(len(edge_bins)) if graph_mask[idx]]
                config_edge_info.append(list(zip(graph_pair_indices, graph_bins)))

            # Target: E_train - E0
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

        result = {
            "X_energy": torch.cat(X_energy_list, dim=0),
            "y_energy": torch.cat(y_energy_list, dim=0),
            "r_min_dict": r_min_dict,
            "pair_bin_counts": pair_bin_counts,
        }

        # Inverse frequency weights
        if self.use_inverse_frequency_weighting and pair_bin_counts:
            inv_freq_table = {k: 1.0 / v for k, v in pair_bin_counts.items() if v > 0}
            config_weights = []
            for edges in config_edge_info:
                weight = sum(inv_freq_table.get(key, 0.0) for key in edges) if edges else 1.0
                config_weights.append(weight)
            total_weight = sum(config_weights)
            if total_weight > 0:
                num_configs = len(config_weights)
                config_weights = [w / total_weight * num_configs for w in config_weights]
            result["energy_weights"] = torch.tensor(config_weights, dtype=torch.float64)

        return result

    def fit(
        self,
        data_loader,
        atomic_energies: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Fit c_matrix and bias_matrix via ridge regression.

        Returns:
            c_matrix: [num_elements, num_elements] repulsion coefficients (clamped >= 0, symmetrized)
            bias_matrix: [num_elements, num_elements] bias values (symmetrized)
            diagnostics: dict with mse, r2, r_min_dict, num_energy_samples
        """
        data = self.preprocess_data(data_loader, atomic_energies)
        X = data["X_energy"]
        y = data["y_energy"]

        # Standardize
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, keepdim=True, unbiased=False)
        X_std = torch.where(X_std < 1e-10, torch.ones_like(X_std), X_std)
        X_scaled = (X - X_mean) / X_std

        # Weights
        if self.use_inverse_frequency_weighting and "energy_weights" in data:
            weights = data["energy_weights"] * self.energy_weight
        else:
            weights = torch.ones(X.shape[0], dtype=torch.float64) * self.energy_weight

        # Solve ridge: (X^T W X + αI) c = X^T W y
        W_sqrt = torch.sqrt(weights).unsqueeze(1)
        Xw = X_scaled * W_sqrt
        yw = y * torch.sqrt(weights)
        A = Xw.T @ Xw + self.alpha * torch.eye(X.shape[1], dtype=torch.float64)
        b = Xw.T @ yw
        c_vector = torch.linalg.solve(A, b)

        # Split into c and bias coefficients
        c_flat = c_vector[:self.num_pairs]
        bias_flat = c_vector[self.num_pairs:]

        # Clamp c >= 0, reshape to matrices
        c_flat = torch.clamp(c_flat, min=0.0)
        c_matrix = c_flat.reshape(self.num_elements, self.num_elements)
        bias_matrix = bias_flat.reshape(self.num_elements, self.num_elements)

        # Symmetrize
        c_matrix = (c_matrix + c_matrix.T) / 2
        bias_matrix = (bias_matrix + bias_matrix.T) / 2

        # Diagnostics
        y_pred = X @ c_vector
        residuals = y - y_pred
        mse = torch.mean(residuals**2).item()
        ss_tot = torch.sum((y - y.mean()) ** 2).item()
        ss_res = torch.sum(residuals**2).item()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        diagnostics = {
            "mse": mse,
            "r2": r2,
            "r_min_dict": data["r_min_dict"],
            "num_energy_samples": X.shape[0],
        }

        logger.info("Ridge regression fit: R²=%.4f, MSE=%.6f, samples=%d",
                     r2, mse, X.shape[0])

        return (
            c_matrix.to(torch.get_default_dtype()),
            bias_matrix.to(torch.get_default_dtype()),
            diagnostics,
        )

    @staticmethod
    def fit_lj_coefficients(
        data_loader,
        z_table: AtomicNumberTable,
        atomic_energies: np.ndarray,
        alpha: float = 1.0,
        energy_weight: float = 1.0,
        use_inverse_frequency_weighting: bool = True,
        r_max: float = 6.0,
        num_distance_bins: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Convenience static method.

        Returns: (c_matrix, bias_matrix, diagnostics)
        """
        fitter = LJRidgeFitter(
            z_table=z_table,
            alpha=alpha,
            energy_weight=energy_weight,
            use_inverse_frequency_weighting=use_inverse_frequency_weighting,
            r_max=r_max,
            num_distance_bins=num_distance_bins,
        )
        return fitter.fit(data_loader, atomic_energies)
