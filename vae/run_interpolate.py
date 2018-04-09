import tensorflow as tf
import numpy as np
import sys,random,os
import matplotlib.pyplot as plt

import text_data

import pdb


import vae


# Train params
debug_print = False
batch_size = 32
default_batches = 9000
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


# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                      RUNNER
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

def test_saved_vae(name=flags.name):
    save_dir = os.path.join('saves',name)

    with tf.Session() as sess:
        ret = tf.saved_model.loader.load(sess, [tag_constants.TRAINING], name)

        print(ret)



def test_vae(sess=None):
    print(" - Start.")

    im_size = [char_size,char_size,1]

    sess = sess if sess else tf.Session()
    sess.run(tf.global_variables_initializer())

    dset = text_data.gen_dataset(classes, char_size, batch_size)
    diter = dset.make_one_shot_iterator()
    dnext = diter.get_next()

    name = flags.name

    save_dir = os.path.join('saves', name)
    net = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],  save_dir)


    x_in = tf.get_default_graph().get_tensor_by_name('x_in:0')
    z_in = tf.get_default_graph().get_tensor_by_name('z_in:0')
    encode_given_x = tf.get_default_graph().get_tensor_by_name('encode_given_x:0')
    decode_given_z = tf.get_default_graph().get_tensor_by_name('decode_given_z:0')
    decode_given_x = tf.get_default_graph().get_tensor_by_name('decode_given_x:0')
    loss_given_x = tf.get_default_graph().get_tensor_by_name('loss_given_x:0')
    opt_loss_given_x = tf.get_default_graph().get_tensor_by_name('opt_loss_given_x:0')

    ''' 
    Create image showing interpolation in z-space
        The top half will be an interpolated line from two actual values.
        The bottom half will be a real point progressively going further away.
    '''

    N = batch_size
    bx = text_data.gen_chars(N, char_size)

    ps = []

    zs = sess.run(encode_given_x, feed_dict={x_in:bx})

    # This could be completely batched, but speed isn't a problem rn
    for j in range(N-1):
        if j<=N//2:
            za,zb = zs[j],zs[j+1]
            if j==N//2:
                ps.append(np.ones_like(ps[-1])*.01) # put a break
        else:
            r = np.random.uniform(size=zs[0].shape) * 3.0
            za,zb = zs[j],zs[j] + r*r

        path = np.linspace(1,0,N)
        cs = [path[i]*za + (1-path[i])*zb for i in range(N)]

        xhs = list(sess.run(decode_given_z, feed_dict={z_in:cs})[...,0])

        p = (np.hstack(xhs))
        p -= np.min(p)
        p /= np.max(p)
        ps.append(p)

    pp = np.vstack(ps)
    plt.figure(figsize=(11,6))
    #plt.imshow(pp,cmap='gray')
    plt.imsave('{}-interpolation.png'.format(flags.name),pp,cmap='gray')
    print(" - Done, created {}-interpolation.png".format(flags.name))

if __name__=='__main__' and 'run' in sys.argv:
    test_vae()

