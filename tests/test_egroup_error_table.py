"""
Tests for PerAtomRMSE_egroup and PerAtomMAE_egroup error table types.

This module tests:
1. Argument parser accepts new error_table types
2. MACELoss computes group energy RMSE and MAE
3. valid_err_log correctly outputs new error types
4. visualise_train error_type dictionary includes new types
"""

import numpy as np
import pytest
import torch

from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.utils import compute_mae, compute_rmse


class TestArgParser:
    """Tests for argument parser with new error_table types."""

    def test_error_table_choices_include_egroup(self):
        """Test that error_table choices include egroup types."""
        parser = build_default_arg_parser()
        
        # Find the error_table argument
        error_table_action = None
        for action in parser._actions:
            if hasattr(action, 'dest') and action.dest == 'error_table':
                error_table_action = action
                break
        
        assert error_table_action is not None, "error_table argument not found"
        assert "PerAtomRMSE_egroup" in error_table_action.choices
        assert "PerAtomMAE_egroup" in error_table_action.choices

    def test_parse_error_table_rmse_egroup(self):
        """Test parsing PerAtomRMSE_egroup error_table option."""
        parser = build_default_arg_parser()
        args = parser.parse_args([
            "--name", "test",
            "--error_table", "PerAtomRMSE_egroup"
        ])
        assert args.error_table == "PerAtomRMSE_egroup"

    def test_parse_error_table_mae_egroup(self):
        """Test parsing PerAtomMAE_egroup error_table option."""
        parser = build_default_arg_parser()
        args = parser.parse_args([
            "--name", "test",
            "--error_table", "PerAtomMAE_egroup"
        ])
        assert args.error_table == "PerAtomMAE_egroup"


class TestVisualiseTrain:
    """Tests for visualise_train error_type dictionary."""

    def test_error_type_includes_egroup(self):
        """Test that error_type dictionary includes egroup types."""
        from mace.cli.visualise_train import error_type
        
        assert "PerAtomRMSE_egroup" in error_type
        assert "PerAtomMAE_egroup" in error_type

    def test_error_type_rmse_egroup_structure(self):
        """Test structure of PerAtomRMSE_egroup error_type entry."""
        from mace.cli.visualise_train import error_type
        
        labels, quantities = error_type["PerAtomRMSE_egroup"]
        
        # Check labels
        label_keys = [label[0] for label in labels]
        assert "rmse_e_per_atom" in label_keys
        assert "rmse_f" in label_keys
        assert "rmse_egroup" in label_keys

    def test_error_type_mae_egroup_structure(self):
        """Test structure of PerAtomMAE_egroup error_type entry."""
        from mace.cli.visualise_train import error_type
        
        labels, quantities = error_type["PerAtomMAE_egroup"]
        
        # Check labels
        label_keys = [label[0] for label in labels]
        assert "mae_e_per_atom" in label_keys
        assert "mae_f" in label_keys
        assert "mae_egroup" in label_keys


class TestMACELossEgroup:
    """Tests for MACELoss group energy metrics computation."""

    @pytest.fixture
    def mock_batch(self):
        """Create a mock batch with group energies."""
        class MockBatch:
            def __init__(self):
                self.num_graphs = 2
                self.ptr = torch.tensor([0, 3, 6])  # 2 graphs with 3 atoms each
                self.energy = torch.tensor([1.0, 2.0])
                self.forces = torch.randn(6, 3)
                self.weight = torch.ones(2)
                self.energy_weight = torch.ones(2)
                self.forces_weight = torch.ones(2)
                self.group_energies = torch.tensor([[0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])
                self.group_energies_weight = torch.ones(2)
                self.atomic_energies = None
                self.atomic_energies_weight = torch.ones(2)
                self.stress = None
                self.stress_weight = torch.ones(2)
                self.virials = None
                self.virials_weight = torch.ones(2)
                self.dipole = None
                self.dipole_weight = torch.ones(2)
                self.polarizability = None
                self.polarizability_weight = torch.ones(2)
        return MockBatch()

    @pytest.fixture
    def mock_output(self):
        """Create mock model output with group energy."""
        return {
            "energy": torch.tensor([1.1, 1.9]),
            "forces": torch.randn(6, 3),
            "group_energy": torch.tensor([0.55, 0.62, 0.68, 0.82, 0.88, 0.98]),
        }

    def test_egroup_rmse_computation(self):
        """Test RMSE computation for group energies."""
        ref = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        pred = np.array([0.55, 0.62, 0.68, 0.82, 0.88, 0.98])
        delta = ref - pred
        
        rmse = compute_rmse(delta)
        expected_rmse = np.sqrt(np.mean(delta ** 2))
        
        assert np.isclose(rmse, expected_rmse)

    def test_egroup_mae_computation(self):
        """Test MAE computation for group energies."""
        ref = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        pred = np.array([0.55, 0.62, 0.68, 0.82, 0.88, 0.98])
        delta = ref - pred
        
        mae = compute_mae(delta)
        expected_mae = np.mean(np.abs(delta))
        
        assert np.isclose(mae, expected_mae)


class TestValidErrLog:
    """Tests for valid_err_log function with egroup error types."""

    def test_valid_err_log_rmse_egroup(self, caplog):
        """Test valid_err_log outputs RMSE_Egroup correctly."""
        import logging
        from mace.tools.train import valid_err_log
        from mace.tools import MetricsLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(directory=directory, tag="test")
            
            eval_metrics = {
                "rmse_e_per_atom": 0.001,
                "rmse_f": 0.01,
                "rmse_egroup": 0.005,
            }
            
            with caplog.at_level(logging.INFO):
                valid_err_log(
                    valid_loss=0.1,
                    eval_metrics=eval_metrics,
                    logger=logger,
                    log_errors="PerAtomRMSE_egroup",
                    epoch=1,
                    valid_loader_name="valid",
                )
            
            # Check that egroup was logged
            assert "RMSE_Egroup" in caplog.text

    def test_valid_err_log_mae_egroup(self, caplog):
        """Test valid_err_log outputs MAE_Egroup correctly."""
        import logging
        from mace.tools.train import valid_err_log
        from mace.tools import MetricsLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as directory:
            logger = MetricsLogger(directory=directory, tag="test")
            
            eval_metrics = {
                "mae_e_per_atom": 0.001,
                "mae_f": 0.01,
                "mae_egroup": 0.005,
            }
            
            with caplog.at_level(logging.INFO):
                valid_err_log(
                    valid_loss=0.1,
                    eval_metrics=eval_metrics,
                    logger=logger,
                    log_errors="PerAtomMAE_egroup",
                    epoch=1,
                    valid_loader_name="valid",
                )
            
            # Check that egroup was logged
            assert "MAE_Egroup" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
