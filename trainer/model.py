import numpy as np
import tensorflow as tf
from architecture import Architecture as Arch

class Model:

    def generator(self, z, y, initializer):

        # z: a random input tensor of size [M, 1, 1, z_size]
        # y: a one-hot label of size [M, 1, 1, num_cat]

        with tf.variable_scope('generator'):

            # concatenate -> [M, 1, 1, z_size + num_cat]
            layer = tf.concat([z, y], axis=3)

            depth = len(Arch.layers_g)
            for i in range(depth):

                layer_config = Arch.layers_g[i]
                is_output = ((i + 1) == depth)

                conv2d = tf.layers.conv2d_transpose(
                    layer, 
                    filters = layer_config['filters'], 
                    kernel_size = layer_config['kernel_size'], 
                    strides = layer_config['strides'], 
                    padding = layer_config['padding'],
                    activation = tf.nn.tanh if is_output else None,
                    kernel_initializer = initializer, 
                    name = 'layer_' + str(i))

                if is_output:
                    layer = conv2d
                else:
                    norm = tf.layers.batch_normalization(conv2d)
                    lrelu = tf.nn.leaky_relu(norm)
                    layer = lrelu
                
            # [M, img_size, img_size, img_channels]
            output = tf.identity(layer, name='generated_images')
            return output 

    def discriminator(self, x, y_expanded, initializer, reuse=False):

        # x: an image tensor of shape [M, img_size, img_size, img_channels]
        # y: a one-hot tensor expanded to size [M, img_size, img_size, num_cat]

        with tf.variable_scope('discriminator', reuse=reuse):

            # concatenate -> [M, img_size, img_size, 11]
            layer = tf.concat([x, y_expanded], axis=3)

            depth = len(Arch.layers_d)
            for i in range(depth):

                layer_config = Arch.layers_d[i]
                is_input = (i == 0)
                is_output = ((i + 1) == depth)

                conv = tf.layers.conv2d(
                    layer, 
                    filters = layer_config['filters'], 
                    kernel_size = layer_config['kernel_size'], 
                    strides = layer_config['strides'], 
                    padding = layer_config['padding'], 
                    activation=tf.nn.leaky_relu if is_input else None,
                    kernel_initializer=initializer, 
                    name='layer_' + str(i))
                
                if is_input:
                    layer = conv
                elif is_output:
                    layer = tf.nn.sigmoid(conv)
                else:
                    norm = tf.layers.batch_normalization(conv)
                    layer = tf.nn.leaky_relu(norm)

            output = tf.reshape(layer, [-1, 1])
            return output # [M, 1]

    # loss
    def loss(self, Dx, Dg):
        '''
        Dx = Probabilities assigned by D to the real images, [M, 1]
        Dg = Probabilities assigned by D to the generated images, [M, 1]
        '''
        with tf.variable_scope('loss'):
            loss_d = tf.identity(-tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg)), name='loss_d')
            loss_g = tf.identity(-tf.reduce_mean(tf.log(Dg)), name='loss_g')
            return loss_d, loss_g

    # Train
    def trainers(self):

        img_size = Arch.img_size
        num_cat = Arch.num_cat

        # placeholders for training data
        images_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 1], name='images_holder')
        labels_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_cat], name='labels_holder')

        # placeholders for random generator input
        z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, Arch.z_size], name='z_holder')
        y_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, num_cat], name='y_holder')
        y_expanded_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_cat], name='y_expanded_holder')

        # forward pass
        weights_init = tf.truncated_normal_initializer(stddev=0.02)
        generated_images = self.generator(z_holder, y_holder, weights_init)
        Dx = self.discriminator(images_holder, labels_holder, weights_init, False)
        Dx = tf.identity(Dx, name="Dx")
        Dg = self.discriminator(generated_images, y_expanded_holder, weights_init, True)
        Dg = tf.identity(Dg, name="Dg")

        # compute losses
        loss_d, loss_g = self.loss(Dx, Dg)

        # optimizers
        optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

        # backprop
        g_vars = tf.trainable_variables(scope='generator')
        d_vars = tf.trainable_variables(scope='discriminator')
        train_g = optimizer_g.minimize(loss_g, var_list=g_vars, name='train_g')
        train_d = optimizer_d.minimize(loss_d, var_list = d_vars, name='train_d')

        return train_d, train_g, loss_d, loss_g, generated_images, Dx, Dg





