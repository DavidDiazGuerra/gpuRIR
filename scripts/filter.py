class Filter(object):
    def __init__(self, filter_strategy):
        self._filter_strategy = filter_strategy

    def apply(self, IR):
        return self._filter_strategy.apply(IR)

class FilterStrategy(object):
    def apply(self):
        '''
        Required method
        '''