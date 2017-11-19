import numpy



# the sum of cross_entropy_loss over a batch
def cross_entropy_loss(predict_prob: numpy.ndarray, y_label: numpy.ndarray) -> float:
    loss = 0.0
    assert y_label.ndim is 1
    assert predict_prob.ndim is 2
    assert y_label.shape[0] == predict_prob.shape[0]
    for i in range(y_label.size):
        loss += -numpy.log(predict_prob[i, y_label[i]])
    return loss


def perplexity(predict_prob: numpy.ndarray, y_label: numpy.ndarray) -> (float, int):
    loss = 0.0
    assert y_label.ndim is 1
    assert predict_prob.ndim is 2
    assert y_label.shape[0] == predict_prob.shape[0]
    batch_size = predict_prob.shape[0]
    weight = batch_size
    for i in range(y_label.size):
        loss += -numpy.log(predict_prob[i, y_label[i]])
    avg_loss = loss / batch_size
    perp = numpy.exp2(avg_loss)
    return perp, weight
