"""SparseLab: inspect LASSO optimization without hiding coefficient ambiguity."""
from .core import fit, path, witness
from .oracle import enumerate_optima
from .demo import demo
__all__ = ['fit', 'path', 'witness', 'enumerate_optima', 'demo']
__version__ = '0.1.0'
