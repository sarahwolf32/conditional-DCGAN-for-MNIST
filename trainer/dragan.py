import tensorflow as tf
import numpy as np

class Dragan:

    '''
    Computes the discriminator gradient penalty for Deep Regret Analytic 
    Generative Adversarial Networks (DRAGAN).

    Should help stabilize training and prevent mode collapse.
    '''

    def gradient_penalty(self, X, D, y_expanded, initializer):
        X_p = self._perturbed_images(X)
        M = 128
        a_shape = [M] + [1] * (len(X.shape) - 1)
        a = tf.random_uniform(a_shape, 0, 1) 
        interpolates = X + a * (X_p - X)
        grad = tf.gradients(D(interpolates, y_expanded, initializer, True), [interpolates])[0]
        slopes = self._norm(grad)
        p = tf.reduce_mean((slopes - 1.)**2)
        return p

    def _std(self, X):
        '''Computes standard deviation of tensor X.'''
        m = tf.reduce_mean(X)
        diff_squared = tf.square(X - m)
        s = tf.reduce_mean(diff_squared)
        std = tf.sqrt(s)
        return std

    def _perturbed_images(self, X):
        '''Add noise to images.'''
        rand = tf.random_uniform([128, 32, 32, 1], 0., 1.)
        diff = self._std(X) * rand
        return X + diff

    def _norm(self, gradients):
        '''Computes an L2-norm that collapses all but the first dimension.'''
        reduction_indeces = [i + 1 for i in range(len(gradients.shape) - 1)]
        n = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=reduction_indeces))
        return n

    

