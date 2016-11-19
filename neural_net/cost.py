from theano import tensor as T

class ClassificationCost(object):
    def __init__(self, output, y, model=None):
        self.output = output
        self.y = y

    def get_function(self):
        return T.mean(T.nnet.categorical_crossentropy(self.output, self.y))
