import numpy as np
from random import shuffle
from past.builtins import xrange

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
  loss_cand = 0.0
  dW = np.zeros_like(W)
  deno = 0
  numer = 0
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = X[i].dot(W[:,y[i]])
    
    for j in xrange(num_classes):
      deno += np.exp(scores[j])
    numer = np.exp(scores[y[i]])

    dW[:,y[i]] += (-1 * X[i] * (deno-numer)) / deno

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      dW[:,j] += (np.exp(X[i].dot(W[:,j])) * X[i]) / deno 

    loss_cand -= np.log(numer/deno) 
    deno = 0

  dW = dW / num_train + reg * 2 * W
  loss = loss_cand / num_train + reg * np.sum(W*W)
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
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  
  deno = np.sum(exp_scores, axis=1)[:,np.newaxis]
  numer = exp_scores[np.arange(num_train), y][:,np.newaxis]
  
  loss = np.sum(-np.log(numer/deno)) / num_train + reg * np.sum(W*W)


  dscores = np.zeros(scores.shape)
  dscores = exp_scores / deno
  dscores[np.arange(num_train),y] = (-(deno - numer)/deno)[:,0]

  for i in xrange(num_train):
    dW += X[i][:,np.newaxis] * dscores[i][np.newaxis,:]

  dW = dW / num_train + reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################



  return loss, dW

