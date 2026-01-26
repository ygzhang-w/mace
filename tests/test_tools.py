import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim

from mace.tools import (
    AtomicNumberTable,
    CheckpointHandler,
    CheckpointState,
    atomic_numbers_to_indices,
)


def test_atomic_number_table():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_indices(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]["lr"], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]["lr"], initial_lr)


def test_patience_early_stopping_single_gpu():
    """Test that patience triggers early stopping in non-distributed mode."""
    from mace.tools.train import train
    from mace.tools import MetricsLogger

    # Create a simple mock model and loss function
    model = MyModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10)

    # Create mock data loaders
    mock_train_loader = MagicMock()
    mock_valid_loader = MagicMock()
    mock_valid_loaders = {"valid": mock_valid_loader}

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        logger = MetricsLogger(directory=directory, tag="test")

        # Mock the evaluate function to return increasing loss (trigger patience)
        with patch("mace.tools.train.evaluate") as mock_evaluate:
            # Return increasing loss values to trigger patience
            mock_evaluate.return_value = (1.0, {"rmse_e_per_atom": 0.1, "rmse_f": 0.1})

            # Mock train_one_epoch to do nothing
            with patch("mace.tools.train.train_one_epoch"):
                # Track actual number of epochs
                epoch_count = {"count": 0}

                def count_epochs(*args, **kwargs):
                    epoch_count["count"] += 1

                with patch("mace.tools.train.train_one_epoch", side_effect=count_epochs):
                    train(
                        model=model,
                        loss_fn=loss_fn,
                        train_loader=mock_train_loader,
                        valid_loaders=mock_valid_loaders,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        start_epoch=0,
                        max_num_epochs=100,  # Set high to ensure patience triggers first
                        patience=3,  # Should stop after 3 epochs of no improvement
                        checkpoint_handler=checkpoint_handler,
                        logger=logger,
                        eval_interval=1,
                        output_args={"energy": True, "forces": True},
                        device=torch.device("cpu"),
                        log_errors="PerAtomRMSE",
                        swa=None,
                        ema=None,
                        max_grad_norm=None,
                        log_wandb=False,
                        distributed=False,  # Test non-distributed mode
                        save_all_checkpoints=False,
                        plotter=None,
                        distributed_model=None,
                        train_sampler=None,
                        rank=0,
                    )

                    # Check that training stopped early due to patience
                    # Should run: epoch 0 (initial), then 3 epochs without improvement, total = 4
                    assert epoch_count["count"] <= 5, f"Expected early stopping but got {epoch_count['count']} epochs"
                    assert epoch_count["count"] >= 3, f"Expected at least 3 epochs but got {epoch_count['count']}"

