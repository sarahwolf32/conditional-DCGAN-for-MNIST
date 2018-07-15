import tensorflow as tf
import numpy as np

class Dragan:

    '''
    Computes the discriminator gradient penalty for Deep Regret Analytic 
    Generative Adversarial Networks (DRAGAN).

    Should help stabilize training and prevent mode collapse.
    '''

    def gradient_penalty(self, X, D):
        X_p = self._perturbed_images(X)
        M = X.shape[0]
        a_shape = [M] + [1] * (len(X.shape) - 1)
        a = tf.random_uniform(a_shape, 0, 1) 
        interpolates = X + a * (X_p - X)
        grad = tf.gradients(D(interpolates), [interpolates])[0]
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
        return X + 0.5 * self._std(X) * np.random.random(X.shape)

    def _norm(self, gradients):
        '''Computes an L2-norm that collapses all but the first dimension.'''
        reduction_indeces = [i + 1 for i in range(len(gradients.shape) - 1)]
        n = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=reduction_indeces))
        return n

    

        
        





# test
g = GradientPenalty()

x = np.array([[1.,2.,3.],[4.,5.,6.],[8., 4., 5.]])
X = tf.constant(x)
sess = tf.Session()
p = g.perturbed_images(X)
print(sess.run(p))