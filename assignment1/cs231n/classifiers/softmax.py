import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Initialize num_class and num_train
  num_class = W.shape[1]
  num_train = X.shape[0]
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    scores_normalize = scores/np.sum(scores)
    loss -= np.log(scores_normalize[y[i]]) 
    dW[:,y[i]] -= X[i].T
    for j in xrange(num_class):
        dW[:,j] += (scores[j]/np.sum(scores))*X[i].T
   
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  score_max = np.max(scores,axis=1)
  score_max = np.reshape(score_max,(num_train,1))
  scores = scores - score_max
  scores = np.exp(scores)
  sum_row = np.sum(scores,axis=1)
  sum_row = np.reshape(sum_row,(num_train,1))
  scores = scores/sum_row
  loss -= np.sum(np.log(scores[xrange(num_train),y]))
  loss = loss/num_train
  dW = X.T .dot(scores)
  binary = np.zeros([num_train,num_class])
  binary[xrange(num_train),y] = 1
  dW -= (X.T).dot(binary)
  dW = dW/num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

