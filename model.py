import numpy as np
import tensorflow as tf
from architecture import Architecture

class Model:

    GENERATOR_SCOPE = 'generator'
    DISCRIMINATOR_SCOPE = 'discriminator'

    def generator(z, y, initializer):

        # z: a random input tensor of size [M, 1, 1, 100]
        # y: a one-hot label of size [M, 1, 1, 10]

        with tf.variable_scope(GENERATOR_SCOPE):

            # concatenate -> [M, 1, 1, 110]
            layer = tf.concat([z, y], axis=3)

            depth = len(Architecture.layers_g)
            for i in range(depth):

                layer_config = Architecture.layers_g[i]
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

    def discriminator(x, y, initializer, reuse=False):

        # x: an image tensor of shape [M, img_size, img_size, img_channels]
        # y: a one-hot label of size [M, 1, 1, 10]
        # architecture: a list of dictionaries that specify the config for each layer

        with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=reuse):

            # concatenate -> [M, img_size, img_size, 11]
            y_expand = y * np.ones([x.shape[0], x.shape[1], x.shape[2], 10])
            layer = tf.concat([x, y_expand], axis=3)

            depth = len(Architecture.layers_d)
            for i in range(depth):

                layer_config = Architecture.layers_d[i]
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
    def loss(Dx, Dg):
        '''
        Dx = Probabilities assigned by D to the real images, [M, 1]
        Dg = Probabilities assigned by D to the generated images, [M, 1]
        '''
        with tf.variable_scope('loss'):
            loss_d = tf.identity(-tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg)), name='loss_d')
            loss_g = tf.identity(-tf.reduce_mean(tf.log(Dg)), name='loss_g')
            return loss_d, loss_g

    # Train
    def trainers():

        # placeholders for training data
        images_holder = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='images_holder')
        labels_holder = y_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 10], name='labels_holder')

        # placeholders for random generator input
        z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 100], name='z_holder')
        y_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 10], name='y_holder')

        # forward pass
        weights_init = tf.truncated_normal_initializer(stddev=0.02)
        generated_images = generator(z_holder, y_holder, weights_init)
        Dx = discriminator(images_holder, labels_holder, weights_init, False)
        Dg = discriminator(generated_images, y_holder, weights_init, True)

        # compute losses
        loss_d, loss_g = loss(Dx, Dg)

        # optimizers
        optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

        # backprop
        g_vars = tf.trainable_variables(scope=GENERATOR_SCOPE)
        d_vars = tf.trainable_variables(scope=DISCRIMINATOR_SCOPE)
        train_g = optimizer_g.minimize(loss_g, var_list=g_vars, name='train_g')
        train_d = optimizer_d.minimize(loss_d, var_list = d_vars, name='train_d')

        return train_d, train_g, loss_d, loss_g, generated_images





