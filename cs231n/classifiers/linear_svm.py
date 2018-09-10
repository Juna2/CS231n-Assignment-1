import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape   (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimension = X.shape[1]
  loss = 0.0
  count = 0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    
    # Choose one of scores as correct class score
    correct_class_score = scores[y[i]]

    # Now we choose another score and compute margin
    for j in xrange(num_classes):
      # if we choose correct class score again, just pass it
      if j == y[i]:
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      # We consider margin as loss only if it's >0
      if margin > 0:
        # This count is for dW of correct class score weights
        count += 1
        loss += margin
        
        # dW is proportional to X. Accumulate all same dW
        for k in xrange(num_dimension):
          dW[k,j] += X[i,k]
    # dW for correct class score is always below 0
    for k in xrange(num_dimension):
      dW[k,y[i]] -= count * X[i,k]
    
    count = 0
  dW = dW / num_train + reg * 2 * W
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather t  hat first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################



  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]
  count = 0
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  for i in xrange(num_train):
    scores[i,:] -= X[i,:].dot(W[:,y[i]])
    scores[i,:] += 1  
    scores[i,y[i]] = 0
    for j in xrange(num_classes):
      if scores[i,j] > 0:
        count += 1
        dW[:,j] += X[i,:]
    dW[:,y[i]] -= count * X[i,:]
    count = 0

  loss_matrix = np.maximum(scores, 0)
  loss = np.sum(loss_matrix) / num_train + reg * np.sum(W*W)
  dW = dW / num_train + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
