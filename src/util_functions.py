import numpy
from scipy.stats import entropy

def cross_entropy_loss(predict_prob: numpy.ndarray, y_label: numpy.ndarray, take_average: bool = False) -> float:
    loss = 0.0
    batch_size = predict_prob.shape[0]
    for i in range(y_label.size):
        try:
            loss += -numpy.log(predict_prob[i, y_label[i]])
        except:
            raise ValueError('Cross Entropy Loss Overflowing')
    if take_average:
        loss /= batch_size
    return loss

def pick_class(softprob: numpy.ndarray) -> numpy.ndarray:
    classes = numpy.argmax(softprob, axis=1)
    return classes


def shuffle(x, y) -> (numpy.ndarray, numpy.ndarray):
    shuffle_idx = numpy.arange(y.size)
    numpy.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    return x, y


def predict_score(predict_label: numpy.ndarray, y_label: numpy.ndarray, take_average: bool = False) -> float:
    tmp: float = numpy.sum(predict_label == y_label).astype(float)
    if take_average:
        tmp /= y_label.size
    return tmp

'''
def perplexity(prob_distribution: numpy.ndarray) -> float:
    tmp = entropy(prob_distribution.T)
    tmp = numpy.power(2, tmp)
    return numpy.sum(tmp)
'''


def perplexity(predict_prob: numpy.ndarray, y_label: numpy.ndarray, take_average: bool = False) -> float:
    loss = 0.0
    batch_size = predict_prob.shape[0]
    for i in range(y_label.size):
        try:
            loss += numpy.exp2(-numpy.log(predict_prob[i, y_label[i]]))
        except:
            raise ValueError('Perplexity Overflowing')
    if take_average:
        loss /= batch_size
    return loss
















