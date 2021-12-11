class Filter(object):
    ''' Encapsulates FilterStrategy and provides interface for applying filters on audio. 
    '''
    def __init__(self, filter_strategy):
        ''' Initializes a filter.

        Parameters
        ----------
        filter_strategy : FilterStrategy
            Instance of a filter.
        '''
        self._filter_strategy = filter_strategy

    def apply(self, IR):
        ''' Applies a filter to given impulse response (IR).

        Parameters
        ----------
        IR : 2D ndarray
            Raw impulse response (IR) data.

        Returns
        -------
        2D ndarray
            Filtered impulse response (IR) data.
        '''
        return self._filter_strategy.apply(IR)

class FilterStrategy(object):
    ''' Superclass of all filters.
    '''
    def apply(self):
        ''' Required method
        '''