import numpy as np
from pandas._libs.parsers import _concatenate_chunks

class ConcatChunks(object):

    def setup(self):
        data = {}
        count = 10000
        data[0] = np.array([100] * count, dtype=np.int64)
        data[1] = np.array([1.6] * count, dtype=np.float64)
        data[2] = np.array(['3/25/2019'] * count, dtype=np.object)
        data[3] = np.array(['3:16'] * count, dtype=np.object)
        data[4] = np.array([np.NaN] * count, dtype=np.object)
        self.chunks = [data]

    def time_concatenate_chunks(self):
        _concatenate_chunks(self.chunks)
