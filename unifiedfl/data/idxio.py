import gzip, struct
import numpy as np

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buf_len = int(np.prod(shape))
    data = np.frombuffer(f.read(buf_len), dtype=np.uint8).reshape(shape)
    return data

def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())

def save_idx(data: np.ndarray, path: str):
    open_f = gzip.open if path.endswith('.gz') else open
    with open_f(path, 'wb') as f: _save_uint8(data, f)

def load_idx(path: str) -> np.ndarray:
    open_f = gzip.open if path.endswith('.gz') else open
    with open_f(path, 'rb') as f: return _load_uint8(f)
