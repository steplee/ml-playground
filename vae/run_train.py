import tensorflow as tf
import numpy as np
import sys,random,os
import matplotlib.pyplot as plt

import text_data


import vae


# Train params
debug_print = False
batch_size = 32
default_batches = 10000
show_every = -1

# Data params
classes = 100
char_size = 60

# Model params
the_latent_size = 49*4
the_conv_channels = [16,16]

tf.app.flags.DEFINE_integer("batches", default_batches, "Number of batches to train")
tf.app.flags.DEFINE_string("name", "vae", "Name of model. Used for saving files, etc.")
flags = tf.app.flags.FLAGS


def test_vae(sess=None):
    print(" - Start.")

    im_size = [char_size,char_size,1]


    dset = text_data.gen_dataset(classes, char_size, batch_size)
    diter = dset.make_one_shot_iterator()
    dnext = diter.get_next(name='x_in') # NOTE: name is crucial when loading!


    net = vae.vae_cnn(x_in=dnext,
            latent_size=the_latent_size,
            name=flags.name)


    sess = sess if sess else tf.Session()
    sess.run(tf.global_variables_initializer())


    # Train loop
    for batch in range(flags.batches):
        loss,xh = sess.run([net.opt_loss_given_x, net.decode_given_x])
        loss = loss[0]

    print(" - Done, saving to saves/{}".format(flags.name))

    net.save(sess)

if __name__=='__main__' and 'run' in sys.argv:
    test_vae()

