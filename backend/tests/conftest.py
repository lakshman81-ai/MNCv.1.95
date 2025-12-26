import sys
import types
import warnings
from pathlib import Path

import numpy as np

# Ensure repository root is importable for `backend` package resolution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _StubTensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __getitem__(self, idx):
        return _StubTensor(self.data[idx])

    def mean(self, dim=None, axis=None):
        axis = 0 if dim is None and axis is None else (dim if dim is not None else axis)
        return _StubTensor(np.mean(self.data, axis=axis))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.data)

    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _ensure_torch_stub():
    if "torch" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")

    def tensor(arr, dtype=None):
        return _StubTensor(np.array(arr, dtype=dtype or np.float32))

    def zeros(shape, dtype=np.float32):
        return _StubTensor(np.zeros(shape, dtype=dtype))

    def no_grad():
        return _NoGrad()

    torch_stub.tensor = tensor
    torch_stub.zeros = zeros
    torch_stub.no_grad = no_grad
    torch_stub.float32 = np.float32
    torch_stub.Tensor = _StubTensor

    sys.modules["torch"] = torch_stub


def pytest_configure(config):
    _ensure_torch_stub()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*nperseg =.*greater than input length.*")
    warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
    warnings.filterwarnings("ignore", message=".*ParseWarning.*")
    warnings.filterwarnings("ignore", message=".*n_fft=.*too large for input signal.*")
