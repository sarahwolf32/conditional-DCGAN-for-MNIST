import numpy as np
import tensorflow as tf
from model import Model
from train_ops import TrainOps
from train_config import TrainConfig
from dataset_loader import DatasetLoader

def create_training_ops():

    # get trainers
    model = Model()
    train_d, train_g, loss_d, loss_g, generated_images = model.trainers()

    # initialize variables
    global_step_var = tf.Variable(0, name='global_step')
    epoch_var = tf.Variable(0, name='epoch')
    batch_var = tf.Variable(0, name='batch')

    # prepare summaries
    loss_d_summary_op = tf.summary.scalar('Discriminator_Loss', loss_d)
    loss_g_summary_op = tf.summary.scalar('Generator_Loss', loss_g)
    images_summary_op = tf.summary.image('Generated_Image', generated_images, max_outputs=1)
    summary_op = tf.summary.merge_all()

def begin_training():
    create_training_ops()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ops = TrainOps()
    ops.populate(sess)
    #train(sess, ops, config)

# Test
config = TrainConfig(local=True)
dataset, num_batches = DatasetLoader().load_dataset(config)
print("dataset")
print(dataset)
print("num batches")
print(num_batches)
