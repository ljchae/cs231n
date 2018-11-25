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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[1]
  num_data = X.shape[0]
  
  for i in range(num_data):
      scoref = np.dot(X[i], W) #score vector
      scoref -= np.max(scoref) #numerical instability
    
      def prob(k):
          return np.exp(scoref[k]) / np.sum(np.exp(scoref))

      L_i = -np.log(prob(y[i]))
      loss += L_i #loss
  
      for k in range(num_class):
          p_i = prob(k)
          dW[:,k] += (p_i-(k==y[i]))*X[i]

  #get average w/ regularization      
  loss /= num_data
  loss += np.sum(W*W) * reg * 0.5
  dW /= num_data
  dW += reg * W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_data = X.shape[0]
  
  scoref = np.dot(X, W)
  scoref -= np.max(scoref)

  prob = np.exp(scoref) / np.sum(np.exp(scoref), axis=1, keepdims=True)

  #loss
  loss = np.sum(-np.log(prob[range(num_data),y]))
  
  #score gradient
  ds = prob
  ds[range(num_data),y] -= 1
  dW = np.dot(X.T, ds)

  #get average w/ regularization
  loss /= num_data
  loss += np.sum(W*W) * reg * 0.5
  dW /= num_data
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


