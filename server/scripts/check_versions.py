import importlib
import importlib.metadata as importlib_metadata

beir = importlib.import_module('beir')
import sentence_transformers
import torch

try:
	beir_version = importlib_metadata.version('beir')
except Exception:
	beir_version = getattr(beir, '__version__', 'unknown')

print('beir_version=', beir_version)
print('sentence_transformers=', sentence_transformers.__version__)
print('torch=', torch.__version__)
