"""Example: predict per-atom qdiv using a trained AtomicQdivMACE model."""

from ase.io import read

from mace.calculators import MACECalculator

calc = MACECalculator(model_paths="qdiv.model", model_type="AtomicQdivMACE")
atoms = read("structure.xyz")
qdiv = calc.get_qdiv(atoms)
print(qdiv)
