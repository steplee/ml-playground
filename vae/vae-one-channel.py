import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import sys,random
import matplotlib.pyplot as plt

import text_data

# TODO: do latent sampling in training loop, penalize `islands` of pixels


CHAR_SIZE = 140
IM_SIZE = [CHAR_SIZE,CHAR_SIZE,1]
LATENT_DIMS = 49*5

lr = .003
CLASSES = 400 # # of chars

BATCH_SZ = 24
DEFAULT_BATCHES = 9000 # train loops

debug_print = False
show_every = -1

def mk_vae_net():
    x_in = tf.placeholder(tf.float32, shape=[None,*IM_SIZE])
    x = x_in
    box = tf.concat([tf.random_uniform([2],.0,.2), tf.random_uniform([2],.7,1.1)], 0)
    box = tf.expand_dims(box,0)
    box = tf.tile(box, [BATCH_SZ,1])
    binds = tf.constant(list(range(BATCH_SZ)))
    x = tf.image.crop_and_resize(x, box, binds, tf.constant(IM_SIZE[0:2]))
    x = tfc.image.rotate(x, tf.random_uniform([1], -.5,.5))
    x = tfc.image.translate(x, tf.random_uniform([2], -20.,20.))

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
    z_from_x = tf.add(mu, sig * eps, name='z_op')

    # Deconv (decoder)
    def decoder(z,reuse=None):
        with tf.variable_scope("decode",reuse=reuse):
            net = tf.layers.dense(z, sz)
            net = tf.reshape(net, [BATCH_SZ, IM_SIZE[0]//4,IM_SIZE[1]//4,8])
            net = tf.layers.conv2d_transpose(net, 16, 4, strides=2, padding='same', activation=tf.nn.relu)
            net = tf.layers.conv2d_transpose(net, 4, 4, strides=2, padding='same', activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 1, 1, padding='same', activation=tf.nn.sigmoid)
            net = tf.clip_by_value(net, 1e-6,1-1e-6)
            return net

    z_in = tf.placeholder(tf.float32, shape=[None,LATENT_DIMS])

    net = decoder(z_from_x)
    decode_from_z = decoder(z_in, reuse=True)



    # Edge loss.
    edge_kernel = tf.transpose(tf.constant([[[[1., 1.,1.],
                                 [1., -1.,1.],
                                 [1., 1.,1.]]]]), perm=[2,3,0,1])
    '''edge_kernel = tf.transpose(tf.constant([[[[-1., -1.,-1.],
                                 [-1.,  1.,-1.],
                                 [-1., -1.,-1.]]]]), perm=[2,3,0,1])'''
    edges = tf.nn.conv2d(net, edge_kernel, [1,1,1,1], 'SAME')
    edges *= net # only count where we have a pixel on

    edge_loss = tf.reduce_mean(tf.square(edges)) * 60

    # VAE Loss
    # See Kingma thesis for derivation. It is pretty simple so long as sigma is diagonal.
    #
    # log p = E[ logp(x,z) - logq(z|x) ]   -   KL[q(z|x) | p(z|x)] 
    #    ELBO \_________________________/


    recons = tf.reduce_sum(x*tf.log(net) + (1-x)*tf.log(1-net),1)
    #recons = -tf.losses.mean_squared_error(net,x) * 1
    recons = tf.reduce_mean(recons)
    likeli = .5 * tf.reduce_mean(1. + tf.log(tf.square(sig)) - tf.square(mu) - tf.square(sig))
    loss = - (likeli + recons*2)
    #loss = loss + edge_loss

    if debug_print:
        loss = tf.Print(loss,[loss,sig])

    opt_op = tf.train.AdamOptimizer(lr).minimize(loss)
    opt_loss = tf.tuple([loss], control_inputs=[opt_op]) # loss=opt

    return x_in,z_from_x,net, opt_loss, decode_from_z,z_in


def run(sess=None, batches=DEFAULT_BATCHES):
    print(" - Start.")

    x_in,z_from_x,net,loss_train, decode_from_z,z_in = mk_vae_net()

    # This line shows whether using GPU or not
    #sess = sess if sess else tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = sess if sess else tf.Session()
    sess.run(tf.global_variables_initializer())

    # A function, call to get batch
    char_gen = text_data.gen_char_gen(CLASSES, CHAR_SIZE)

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
        zs = sess.run(z_from_x, feed_dict={x_in:bx})
        za,zb = zs[0],zs[1]
        path = np.linspace(0,1,N)
        cs = [path[i]*za + (1-path[i])*zb for i in range(N)]

        xhs = list(sess.run(decode_from_z, feed_dict={z_in:cs})[...,0])

        p = (np.hstack(xhs))
        p -= np.min(p)
        p /= np.max(p)
        ps.append(p)

    pp = np.vstack(ps)
    plt.figure(figsize=(11,6))
    #plt.imshow(pp,cmap='gray')
    plt.imsave('interp.png',pp,cmap='gray')

if __name__=='__main__' and 'run' in sys.argv:
    try:
        batches = int(sys.argv[-1]) # user can provide training loops as last arg
    except:
        batches = DEFAULT_BATCHES

    run(batches=batches)
