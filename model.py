import numpy as np
import tensorflow as tf
from architecture import Architecture

def generator(z, y, architecture, initializer):

    # z: a random input tensor of size [M, 1, 1, 100]
    # y: a one-hot label of size [M, 1, 1, 10]
    # architecture: a list of dictionaries that specify the config for each layer

    # concatenate -> [M, 1, 1, 110]
    layer = tf.concat([z, y], axis=3)

    depth = len(architecture)
    for i in range(depth):

        layer_config = architecture[i]
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
        
    output = tf.identity(layer, name='generated_images')
    return output # [M, 32, 32, 1]

def discriminator(x, y, architecture, initializer):

    # x: an image tensor of shape [M, 32, 32, 1]
    # y: a one-hot label of size [M, 1, 1, 10]
    # architecture: a list of dictionaries that specify the config for each layer

    # concatenate -> [M, 32, 32, 11]
    y_expand = y * np.ones([x.shape[0], x.shape[1], x.shape[2], 10])
    layer = tf.concat([x, y_expand], axis=3)

    depth = len(architecture)
    for i in range(depth):

        layer_config = architecture[i]
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








# placeholders
z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 100])
y_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 10])

# TEST
m = 5
z = np.random.normal(0.0, 1.0, size=[m, 1, 1, 100])
y_labels = [0, 5, 2, 1, 4]
y = tf.one_hot(y_labels, depth=10)
y = tf.reshape(y, [-1, 1, 1, 10])

g = generator(z, y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    g_val = sess.run(g)
    print(g_val.shape)
    



