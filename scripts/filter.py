class Filter(object):
    def __init__(self, filter_strategy):
        self._filter_strategy = filter_strategy

    def apply(self, IR, params):
        self._filter_strategy.apply(IR, params)

class FilterStrategy(object):
    def apply(self):
        '''
        Required method
        '''