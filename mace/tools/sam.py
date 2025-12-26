"""
SAM (Sharpness-Aware Minimization) optimizer implementation.

This module provides a wrapper for any base optimizer to implement
Sharpness-Aware Minimization, which seeks parameters that lie in
neighborhoods having uniformly low loss.

References:
    Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021).
    Sharpness-aware minimization for efficiently improving generalization.
    ICLR 2021.
"""

from typing import Optional

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    SAM simultaneously minimizes loss value and loss sharpness by seeking
    parameters in neighborhoods with uniformly low loss. This wrapper can
    be applied to any base optimizer (e.g., Adam, AdamW, SGD).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        base_optimizer: Base optimizer class (e.g., torch.optim.Adam).
        rho: Size of the neighborhood for computing max loss. Larger values
            increase regularization strength. Typical values: 0.05 for
            standard SAM, 2.0 for adaptive SAM.
        adaptive: Whether to use Adaptive SAM (ASAM). ASAM scales the
            perturbation by parameter magnitudes, making it scale-invariant.
        **kwargs: Arguments passed to the base optimizer constructor
            (e.g., lr, weight_decay, betas).

    Example:
        >>> model = MyModel()
        >>> base_optimizer = torch.optim.Adam
        >>> optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, rho=0.05)
        >>> # Training loop
        >>> for batch in dataloader:
        >>>     loss = loss_function(model(batch))
        >>>     loss.backward()
        >>>     optimizer.first_step(zero_grad=True)
        >>>     
        >>>     # Second forward-backward pass
        >>>     loss_function(model(batch)).backward()
        >>>     optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}, must be non-negative")

        defaults = {"rho": rho, "adaptive": adaptive}
        super().__init__(params, defaults)

        # Create the base optimizer with the same parameter groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Perform the first optimization step: compute gradient and perturb
        parameters in the direction of steepest ascent.

        This method should be called after the first backward pass. It
        computes the perturbation e_w = rho * grad / ||grad|| and adds
        it to the parameters.

        Args:
            zero_grad: If True, set gradients to zero after this step.
        """
        # Compute gradient norm for each parameter group
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Store original parameter value
                self.state[p]["old_p"] = p.data.clone()

                # Compute perturbation: e_w = rho * grad / ||grad||
                # For adaptive SAM, scale by parameter magnitude
                if group["adaptive"]:
                    e_w = (torch.pow(p, 2) if p.dim() > 0 else torch.abs(p)) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)

                # Perturb parameters: w' = w + e_w
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Perform the second optimization step: restore original parameters
        and apply the base optimizer update.

        This method should be called after the second backward pass. It
        restores the original parameters, then applies the gradient update
        from the perturbed location.

        Args:
            zero_grad: If True, set gradients to zero after this step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Apply base optimizer step with gradients from perturbed location
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a complete SAM optimization step using a closure.

        This is an alternative API similar to LBFGS. The closure should
        perform a forward pass, compute the loss, and call backward().

        Args:
            closure: A callable that reevaluates the model and returns the loss.

        Returns:
            The loss value returned by the closure (if provided).

        Example:
            >>> def closure():
            >>>     loss = loss_function(model(batch))
            >>>     loss.backward()
            >>>     return loss
            >>> optimizer.step(closure)
        """
        if closure is None:
            raise ValueError(
                "SAM.step() requires a closure argument. "
                "Use first_step() and second_step() for manual control."
            )

        # First step: compute gradient and perturb
        loss = closure()
        loss.backward()
        self.first_step(zero_grad=True)

        # Second step: compute gradient at perturbed location
        with torch.enable_grad():
            closure().backward()

        self.second_step(zero_grad=True)

        return loss

    def _grad_norm(self):
        """
        Compute the gradient norm across all parameters.

        Returns:
            Gradient norm as a scalar tensor.
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients of all optimized parameters."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
