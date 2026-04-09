import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


@pytest.fixture(name="qdiv_configs")
def fixture_qdiv_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = []
    np.random.seed(42)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.new_array("REF_qdiv", np.random.normal(0.0, 0.5, size=len(c)))
        fit_configs.append(c)
    return fit_configs


def test_run_train_qdiv(tmp_path, qdiv_configs):
    ase.io.write(tmp_path / "fit.xyz", qdiv_configs)

    mace_params = {
        "name": "MACE_qdiv",
        "valid_fraction": 0.1,
        "qdiv_weight": 1.0,
        "model": "AtomicQdivMACE",
        "hidden_irreps": "32x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 42,
        "loss": "qdiv",
        "error_table": "QdivRMSE",
        "qdiv_key": "REF_qdiv",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit.xyz"),
        "E0s": "{8: 0.0, 1: 0.0}",
        "eval_interval": 2,
        "use_reduced_cg": False,
    }

    cmd = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        if v is not None:
            cmd.append(f"--{k}={v}")
        else:
            cmd.append(f"--{k}")

    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # Check that a model was saved
    model_files = list(tmp_path.glob("MACE_qdiv*.model"))
    assert len(model_files) > 0, f"No model files found in {tmp_path}"


@pytest.fixture(name="qdiv_configs_with_bias")
def fixture_qdiv_configs_with_bias():
    """Configs with distinct per-element qdiv baselines (like real data)."""
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    # Per-element baselines: O~6.27, H~0.96
    element_bias = {8: 6.27, 1: 0.96}
    fit_configs = []
    np.random.seed(42)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        qdiv = np.array([element_bias[z] + np.random.normal(0, 0.05) for z in c.numbers])
        c.new_array("REF_qdiv", qdiv)
        fit_configs.append(c)
    return fit_configs


def test_run_train_qdiv_with_q0s(tmp_path, qdiv_configs_with_bias):
    ase.io.write(tmp_path / "fit.xyz", qdiv_configs_with_bias)

    mace_params = {
        "name": "MACE_qdiv_q0s",
        "valid_fraction": 0.1,
        "qdiv_weight": 1.0,
        "model": "AtomicQdivMACE",
        "hidden_irreps": "32x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 10,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 42,
        "loss": "qdiv",
        "error_table": "QdivRMSE",
        "qdiv_key": "REF_qdiv",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit.xyz"),
        "E0s": "{8: 0.0, 1: 0.0}",
        "Q0s": "average",
        "eval_interval": 2,
        "use_reduced_cg": False,
    }

    cmd = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        if v is not None:
            cmd.append(f"--{k}={v}")
        else:
            cmd.append(f"--{k}")

    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # Check that a model was saved
    model_files = list(tmp_path.glob("MACE_qdiv_q0s*.model"))
    assert len(model_files) > 0, f"No model files found in {tmp_path}"


def test_get_qdiv(tmp_path, qdiv_configs):
    """Test MACECalculator.get_qdiv() returns correct shape."""
    ase.io.write(tmp_path / "fit.xyz", qdiv_configs)

    mace_params = {
        "name": "MACE_qdiv_calc",
        "valid_fraction": 0.1,
        "qdiv_weight": 1.0,
        "model": "AtomicQdivMACE",
        "hidden_irreps": "32x0e",
        "r_max": 3.5,
        "batch_size": 5,
        "max_num_epochs": 2,
        "swa": None,
        "start_swa": 5,
        "ema": None,
        "ema_decay": 0.99,
        "amsgrad": None,
        "restart_latest": None,
        "device": "cpu",
        "seed": 42,
        "loss": "qdiv",
        "error_table": "QdivRMSE",
        "qdiv_key": "REF_qdiv",
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "train_file": str(tmp_path / "fit.xyz"),
        "E0s": "{8: 0.0, 1: 0.0}",
        "eval_interval": 2,
        "use_reduced_cg": False,
    }

    cmd = [sys.executable, str(run_train)]
    for k, v in mace_params.items():
        if v is not None:
            cmd.append(f"--{k}={v}")
        else:
            cmd.append(f"--{k}")

    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    model_path = tmp_path / "MACE_qdiv_calc.model"
    assert model_path.exists()

    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model_path), model_type="AtomicQdivMACE")
    atoms = qdiv_configs[0].copy()
    qdiv = calc.get_qdiv(atoms)

    assert isinstance(qdiv, np.ndarray)
    assert qdiv.shape == (len(atoms),)
