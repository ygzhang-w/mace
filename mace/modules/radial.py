###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Optional

import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class ChebychevBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.register_buffer(
            "n",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()).unsqueeze(
                0
            ),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x.repeat(1, self.num_basis)
        n = self.n.repeat(len(x), 1)
        return torch.special.chebyshev_polynomial_t(x, n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis},"
        )


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


@compile_mode("script")
class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(
        self,
        p=6,
        trainable=False,
        pair_r_max: torch.Tensor = None,
        zbl_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated. r_max is determined from the covalent radii or pair_r_max."
            )

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        # ZBL scaling factor to adjust energy magnitude
        self.register_buffer(
            "zbl_scale", torch.tensor(zbl_scale, dtype=torch.get_default_dtype())
        )
        # pair_r_max: 2D tensor of shape (max_z, max_z) where pair_r_max[Z_u, Z_v] = r_max
        # Values of -1.0 indicate "use covalent radii fallback"
        if pair_r_max is not None:
            self.register_buffer("pair_r_max", pair_r_max)
        else:
            # Register None as a buffer to ensure attribute exists
            self.register_buffer("pair_r_max", None)

        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi

        # Compute r_max for each edge
        # Use getattr for backward compatibility with old models that don't have pair_r_max
        pair_r_max = getattr(self, "pair_r_max", None)
        if pair_r_max is not None:
            # Use pair_r_max tensor if available
            # Flatten Z_u and Z_v for indexing
            Z_u_flat = Z_u.squeeze(-1)
            Z_v_flat = Z_v.squeeze(-1)
            r_max = pair_r_max[Z_u_flat, Z_v_flat].unsqueeze(-1)
            # Fallback to covalent radii where pair_r_max is -1.0
            covalent_r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
            r_max = torch.where(r_max < 0, covalent_r_max, r_max)
        else:
            # Default: use covalent radii sum
            r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]

        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        # Use getattr for backward compatibility with old models that don't have zbl_scale
        zbl_scale = getattr(self, "zbl_scale", torch.tensor(1.0))
        v_edges = 0.5 * v_edges * envelope * zbl_scale
        V_ZBL = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        zbl_scale = getattr(self, "zbl_scale", torch.tensor(1.0))
        pair_r_max = getattr(self, "pair_r_max", None)
        return f"{self.__class__.__name__}(c={self.c}, zbl_scale={zbl_scale.item()}, pair_r_max={'custom' if pair_r_max is not None else 'covalent_radii'})"


@compile_mode("script")
class LJBasis(torch.nn.Module):
    """Implementation of Lennard-Jones 12-6 potential with polynomial cutoff envelope.

    V_LJ(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6] * envelope(r)

    Args:
        p: Polynomial cutoff order (default: 6)
        trainable: Whether epsilon and sigma are trainable parameters (default: False)
        pair_r_max: Custom r_max tensor of shape (max_z, max_z), values of -1.0 indicate
                    "use covalent radii fallback"
        lj_epsilon: Energy depth parameter in eV (default: 0.01)
        lj_sigma: Zero potential distance in Angstrom (default: 3.0)
        lj_scale: Global scaling factor for the potential (default: 1.0)
    """

    p: torch.Tensor

    def __init__(
        self,
        p: int = 6,
        trainable: bool = False,
        pair_r_max: torch.Tensor = None,
        lj_epsilon: float = 0.01,
        lj_sigma: float = 3.0,
        lj_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        # LJ scaling factor to adjust energy magnitude
        self.register_buffer(
            "lj_scale", torch.tensor(lj_scale, dtype=torch.get_default_dtype())
        )
        # pair_r_max: 2D tensor of shape (max_z, max_z) where pair_r_max[Z_u, Z_v] = r_max
        # Values of -1.0 indicate "use covalent radii fallback"
        if pair_r_max is not None:
            self.register_buffer("pair_r_max", pair_r_max)
        else:
            # Register None as a buffer to ensure attribute exists
            self.register_buffer("pair_r_max", None)

        # LJ parameters: epsilon (energy depth) and sigma (zero potential distance)
        if trainable:
            self.epsilon = torch.nn.Parameter(
                torch.tensor(lj_epsilon, dtype=torch.get_default_dtype()),
                requires_grad=True,
            )
            self.sigma = torch.nn.Parameter(
                torch.tensor(lj_sigma, dtype=torch.get_default_dtype()),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "epsilon", torch.tensor(lj_epsilon, dtype=torch.get_default_dtype())
            )
            self.register_buffer(
                "sigma", torch.tensor(lj_sigma, dtype=torch.get_default_dtype())
            )

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)

        # Prevent numerical overflow by clamping minimum distance
        x_safe = torch.clamp(x, min=0.5)

        # Compute LJ potential: V = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
        sigma_over_r = self.sigma / x_safe
        sigma_over_r_6 = torch.pow(sigma_over_r, 6)
        sigma_over_r_12 = torch.pow(sigma_over_r_6, 2)
        v_edges = 4.0 * self.epsilon * (sigma_over_r_12 - sigma_over_r_6)

        # Compute r_max for each edge (same logic as ZBLBasis)
        # Use getattr for backward compatibility with old models that don't have pair_r_max
        pair_r_max = getattr(self, "pair_r_max", None)
        if pair_r_max is not None:
            # Use pair_r_max tensor if available
            Z_u_flat = Z_u.squeeze(-1)
            Z_v_flat = Z_v.squeeze(-1)
            r_max = pair_r_max[Z_u_flat, Z_v_flat].unsqueeze(-1)
            # Fallback to covalent radii where pair_r_max is -1.0
            covalent_r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
            r_max = torch.where(r_max < 0, covalent_r_max, r_max)
        else:
            # Default: use covalent radii sum
            r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]

        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        # Use getattr for backward compatibility with old models that don't have lj_scale
        lj_scale = getattr(self, "lj_scale", torch.tensor(1.0))
        v_edges = 0.5 * v_edges * envelope * lj_scale
        V_LJ = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_LJ.squeeze(-1)

    def __repr__(self):
        lj_scale = getattr(self, "lj_scale", torch.tensor(1.0))
        pair_r_max = getattr(self, "pair_r_max", None)
        return (
            f"{self.__class__.__name__}(epsilon={self.epsilon.item():.4f}, "
            f"sigma={self.sigma.item():.3f}, lj_scale={lj_scale.item()}, "
            f"pair_r_max={'custom' if pair_r_max is not None else 'covalent_radii'})"
        )


@compile_mode("script")
class LJRepulsionBasis(torch.nn.Module):
    """Region-split Lennard-Jones repulsion potential with unified constant and per-pair bias.

    V_edge = repulsion_c * r^{-12} * 0.5 + bias_ij

    Args:
        num_elements: Number of element types in the model
        repulsion_c: Unified repulsion constant for all pairs (default: 1.0)
        bias_matrix: Optional [num_elements, num_elements] bias tensor, defaults to zeros
        trainable: If True, bias_matrix is a trainable Parameter; otherwise a buffer
    """

    def __init__(
        self,
        num_elements: int,
        repulsion_c: float = 1.0,
        bias_matrix: Optional[torch.Tensor] = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.num_elements = num_elements
        self.repulsion_c = repulsion_c

        if bias_matrix is None:
            bias_matrix = torch.zeros(
                num_elements, num_elements, dtype=torch.get_default_dtype()
            )

        if trainable:
            self.bias_matrix = torch.nn.Parameter(bias_matrix, requires_grad=True)
        else:
            self.register_buffer("bias_matrix", bias_matrix)

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        # Get element type indices from one-hot encoding
        sender_z = torch.argmax(node_attrs[sender], dim=1)
        receiver_z = torch.argmax(node_attrs[receiver], dim=1)

        # Look up bias for each edge
        bias_ij = self.bias_matrix[sender_z, receiver_z]

        # Ensure x is 1D (squeeze if needed)
        x_flat = x.squeeze(-1) if x.dim() > 1 else x

        # Prevent numerical overflow by clamping minimum distance
        x_safe = torch.clamp(x_flat, min=0.5)

        # Compute edge potential: V_edge = c * r^{-12} * 0.5 + bias_ij
        v_edges = self.repulsion_c * torch.pow(x_safe, -12) * 0.5 + bias_ij

        # Aggregate to nodes
        V_LJ = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_LJ.squeeze(-1)

    def __repr__(self):
        trainable_str = "trainable" if self.bias_matrix.requires_grad else "frozen"
        return (
            f"{self.__class__.__name__}(num_elements={self.num_elements}, "
            f"repulsion_c={self.repulsion_c}, {trainable_str})"
        )


@compile_mode("script")
class AgnesiTransform(torch.nn.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """
    Tanh-based smooth transformation:
        T(x) = p1 + (x - p1)*0.5*[1 + tanh(alpha*(x - m))],
    which smoothly transitions from ~p1 for x << p1 to ~x for x >> r0.
    """

    def __init__(self, alpha: float = 4.0, trainable=False):
        """
        Args:
            p1 (float): Lower "clamp" point.
            alpha (float): Steepness; if None, defaults to ~6/(r0-p1).
            trainable (bool): Whether to make parameters trainable.
        """
        super().__init__()
        # Initialize parameters
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_0 based on atomic information.

        Args:
            node_attrs (torch.Tensor): Node attributes (one-hot encoding of atomic numbers).
            edge_index (torch.Tensor): Edge index indicating connections.
            atomic_numbers (torch.Tensor): Atomic numbers.

        Returns:
            torch.Tensor: r_0 values for each edge.
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        return r_0

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:

        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers)
        p_0 = (3 / 4) * r_0
        p_1 = (4 / 3) * r_0
        m = 0.5 * (p_0 + p_1)
        alpha = self.alpha / (p_1 - p_0)
        s_x = 0.5 * (1.0 + torch.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha.item():.4f})"


class RadialMLP(torch.nn.Module):
    """
    Construct a radial MLP (Linear → LayerNorm → SiLU) stack
    given a list of channel sizes, following ESEN / FairChem.
    """

    def __init__(self, channels_list) -> None:
        super().__init__()

        modules = []
        in_channels = channels_list[0]

        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if idx < len(channels_list) - 1:
                modules.append(torch.nn.LayerNorm(out_channels))
                modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)
        self.hs = channels_list

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
