import tensorflow as tf
import numpy as np
import sys,random
import matplotlib.pyplot as plt

import text_data


CHAR_SIZE = 80
IM_SIZE = [CHAR_SIZE,CHAR_SIZE,1]
LATENT_DIMS = 49

lr = .003
CLASSES = 10 # # of chars

BATCH_SZ = 24
DEFAULT_BATCHES = 5000 # train loops

debug_print = False
show_every = -1

def mk_vae_net():
    x = tf.placeholder(tf.float32, shape=[None,*IM_SIZE])

    # Conv (encoder)
    net = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,2)
    net = tf.layers.conv2d(net, 8, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,2)
    #net = tf.layers.batch_normalization(net)

    # FC 
    sz = net.shape[1]*net.shape[2]*net.shape[3]
    net = tf.reshape(net, [-1, sz])

    # Reparameterization:
    #   z = m + s*epsilon
    mu  = tf.layers.dense(net, LATENT_DIMS, name='mu_op', use_bias=False)
    sig = tf.layers.dense(net, LATENT_DIMS, name='sig_op', activation=tf.nn.softplus) + 1e-4
    eps = tf.random_normal(shape=[LATENT_DIMS], name='eps')
    z = tf.add(mu, sig * eps, name='z_op')

    # Deconv (decoder)
    net = tf.layers.dense(z, sz)
    net = tf.reshape(net, [BATCH_SZ, IM_SIZE[0]//4,IM_SIZE[1]//4,8])
    net = tf.layers.conv2d_transpose(net, 16, 4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(net, 4, 4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 1, 1, padding='same', activation=tf.nn.sigmoid)
    net = tf.clip_by_value(net, 1e-6,1-1e-6)


    # Loss
    # See Kingma thesis for derivation. It is pretty simple so long as sigma is diagonal.
    #
    # log p = E[ logp(x,z) - logq(z|x) ]   -   KL[q(z|x) | p(z|x)] 
    #    ELBO \_________________________/

    recons = tf.reduce_sum(x*tf.log(net) + (1-x)*tf.log(1-net),1)
    recons = tf.reduce_mean(recons)
    likeli = .5 * tf.reduce_mean(1. + tf.log(tf.square(sig)) - tf.square(mu) - tf.square(sig))
    loss = - (likeli + recons)

    if debug_print:
        loss = tf.Print(loss,[loss,sig])

    opt_op = tf.train.AdamOptimizer(lr).minimize(loss)
    opt_loss = tf.tuple([loss], control_inputs=[opt_op]) # loss=opt

    return x,z,net, opt_loss


def run(sess=None, batches=DEFAULT_BATCHES):
    print(" - Start.")

    x_in,z,net,loss_train = mk_vae_net()

    sess = sess if sess else tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    # A function, call to get batch
    char_gen = text_data.gen_char_gen(10, CHAR_SIZE)

    # Train loop
    for batch in range(batches):
        bx,by = char_gen(BATCH_SZ)

        loss,xh = sess.run([loss_train,net], feed_dict={x_in:bx})
        loss = loss[0]

        if batch % 10 == 0:
            print("%5d: %.3f"%(batch,loss))
            if show_every > 0 and batch % show_every == 0:
                p0 = bx[0,:,:,0]
                p1 = xh[0,:,:,0]
                p = np.hstack([p0,p1])
                plt.imshow(p,cmap='gray')
                plt.show()

    print(" - Done, creating interp.png")

    ''' Create image showing interpolation in z-space '''

    N = BATCH_SZ
    ps = []
    for j in range(N-1):
        a,b = bx[j],bx[j+1]
        path = np.linspace(0,1,N)
        cs = [path[i]*a + (1-path[i])*b for i in range(N)]

        xhs = list(sess.run([(net)], feed_dict={x_in:cs})[0][...,0])

        p = (np.hstack(xhs))
        p += np.min(p)
        p /= np.max(p)
        ps.append(p)

    pp = np.vstack(ps)
    plt.figure(figsize=(11,6))
    plt.imshow(pp,cmap='gray')
    plt.imsave('interp.png',pp,cmap='gray')
    plt.show()

if __name__=='__main__' and 'run' in sys.argv:
    try:
        batches = int(sys.argv[-1]) # user can provide training loops as last arg
    except:
        batches = DEFAULT_BATCHES

    run(batches=batches)
