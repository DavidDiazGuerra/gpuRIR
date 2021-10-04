from filter import FilterStrategy
from scipy.signal import lfilter

class LinearFilter(FilterStrategy):
    def __init__(self, b, a, source_signal):
        self.b = b
        self.a = a
        self.source_signal = source_signal
        self.NAME = "Linear"

    
    '''
    Applies a butterworth bandpass filter.
    '''
    @staticmethod
    def apply_linear_filter(data, a, b):
        y = lfilter(b, a, data)
        return y

    def apply(self, IR):
        return self.apply_linear_filter(IR, self.a, self.b)


