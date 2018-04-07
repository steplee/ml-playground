import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import sys,random
import matplotlib.pyplot as plt

import text_data

'''
This is a regular VAE augmented with an additional make boundaries
between classes in latent space (hopefully) tighter.
It does this by taking lines in between pairs of z-points and encouraging them to
be decode into the endpoints.
Actually this is done with 3 z-points, interpolating a -> b -> c with improper weighting.
I want to see what effect this has on drawing samples from random z points.
'''


CHAR_SIZE = 140
IM_SIZE = [CHAR_SIZE,CHAR_SIZE,1]
LATENT_DIMS = 49*5

lr = .003
CLASSES = 400 # # of chars

BATCH_SZ = 24
DEFAULT_BATCHES = 5400 # train loops

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
    x = tfc.image.rotate(x, tf.random_uniform([BATCH_SZ], -.5,.5))
    x = tfc.image.translate(x, tf.random_uniform([BATCH_SZ,2], -20.,20.))

    # Conv (encoder)
    def mk_encoder(x, reuse=False):
        with tf.variable_scope('encoder',reuse=reuse):
            net = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(net, 2,2)
            net = tf.layers.conv2d(net, 8, 3, padding='same', activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(net, 2,2)
            #net = tf.layers.batch_normalization(net)

            # FC 
            conv0_sz = net.shape[1]*net.shape[2]*net.shape[3]
            net = tf.reshape(net, [-1, conv0_sz])

            # Reparameterization:
            #   z = m + s*epsilon
            mu  = tf.layers.dense(net, LATENT_DIMS, name='mu_op', use_bias=False)
            sig = tf.layers.dense(net, LATENT_DIMS, name='sig_op', activation=tf.nn.softplus) + 1e-4
            eps = tf.random_normal(shape=[LATENT_DIMS], name='eps')
            z_from_x = tf.add(mu, sig * eps, name='z_op')
            return mu,sig,z_from_x, conv0_sz

    # Deconv (decoder)
    def mk_decoder(z,conv0_sz, batch_size, reuse=None):
        with tf.variable_scope("decoder",reuse=reuse):
            net = tf.layers.dense(z, conv0_sz)
            net = tf.reshape(net, [batch_size, IM_SIZE[0]//4,IM_SIZE[1]//4,8])
            net = tf.layers.conv2d_transpose(net, 16, 4, strides=2, padding='same', activation=tf.nn.relu)
            net = tf.layers.conv2d_transpose(net, 4, 4, strides=2, padding='same', activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 1, 1, padding='same', activation=tf.nn.sigmoid)
            net = tf.clip_by_value(net, 1e-6,1-1e-6)
            return net


    mu,sig,z_from_x,conv0_sz = mk_encoder(x)
    decode_from_x = mk_decoder(z_from_x, conv0_sz, BATCH_SZ)


    # If we want to feed in z for exploring the latent space.
    z_in = tf.placeholder(tf.float32, shape=[None,LATENT_DIMS])
    decode_from_given_z = mk_decoder(z_in, conv0_sz, BATCH_SZ, reuse=True)


    # VAE Loss
    # See Kingma thesis for derivation. It is pretty simple so long as sigma is diagonal.
    #
    # log p = E[ logp(x,z) - logq(z|x) ]   -   KL[q(z|x) | p(z|x)] 
    #    ELBO \_________________________/

    def mk_loss(xh, mu, sig):
        recons = tf.reduce_sum(x*tf.log(xh) + (1-x)*tf.log(1-xh),1)
        #recons = -tf.losses.mean_squared_error(net,x) * 1
        recons = tf.reduce_mean(recons)
        likeli = .5 * tf.reduce_mean(1. + tf.log(tf.square(sig)) - tf.square(mu) - tf.square(sig))
        loss = - (likeli + recons*2)
        #loss = loss + edge_loss
        return loss



    ''' Force interpolated z points values to resemble their endpoints.
        It will reuse `x`, `mu`, and `sig`, computed for the primary net. '''
    def mk_interp_subnet(x, z, conv0_sz):
        K, k, kk = 24, 24//3, 24//2

        '''
        tb = np.concatenate([np.linspace(0,1,kk,dtype=np.float32) , np.linspace(1,0,kk,dtype=np.float32)])
        tb = tf.transpose(tf.constant(np.array([[[tb]]])))
        tc = np.concatenate([np.zeros(kk,dtype=np.float32), np.linspace(0,1,kk,dtype=np.float32)])
        tc = tf.transpose(tf.constant(np.array([[[tc]]])))'''
        tb = np.concatenate([np.linspace(0,1,kk,dtype=np.float32) , np.linspace(1,0,kk,dtype=np.float32)])
        tb = tf.transpose(tf.constant(np.array([[[tb]]])))
        tc = np.linspace(0,1,K,dtype=np.float32)
        tc = tf.transpose(tf.constant(np.array([[[tc]]])))

        _loss = None

        for i in range(10):
            x_a = tf.tile(x[i:i+1], [k,1,1,1])
            x_b = tf.tile(x[i+1:i+2], [k,1,1,1])
            x_c = tf.tile(x[i+2:i+3], [k,1,1,1])
            x_target = tf.concat([x_a,x_b,x_c], 0)

            zs = (1-tb-tc)*z[i] + (tb)*z[i+1] + (tc)*z[i+2]

            xh = mk_decoder(zs, conv0_sz, K, reuse=True)

            if _loss is None:
                _loss = -tf.reduce_mean(tf.reduce_sum(x_target*tf.log(xh) + (1-x_target)*tf.log(1-xh),1))
            else:
                _loss += -tf.reduce_mean(tf.reduce_sum(x_target*tf.log(xh) + (1-x_target)*tf.log(1-xh),1))

        return _loss
        #return mk_loss(x_target, mu,sig)



    vae_loss = mk_loss(decode_from_x, mu, sig)
    interp_loss = mk_interp_subnet(x, z_from_x, conv0_sz)

    loss = vae_loss + interp_loss
    #loss = vae_loss 

    if debug_print:
        loss = tf.Print(vae_,[vae_loss,sig])

    opt_op = tf.train.AdamOptimizer(lr).minimize(vae_loss)
    opt_loss = tf.tuple([loss], control_inputs=[opt_op]) # loss=opt

    return x_in,z_from_x,decode_from_x, opt_loss, decode_from_given_z,z_in


def run(sess=None, batches=DEFAULT_BATCHES):
    print(" - Start.")

    x_in,z_from_x,net,loss_train, decode_from_given_z,z_in = mk_vae_net()

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
        if j>N//2:
            za,zb = zs[0],zs[1]
        else:
            za,zb = zs[0],zs[0] + np.random.uniform(size=zs[0].shape)*2.2
        path = np.linspace(0,1,N)
        cs = [path[i]*za + (1-path[i])*zb for i in range(N)]

        xhs = list(sess.run(decode_from_given_z, feed_dict={z_in:cs})[...,0])

        p = (np.hstack(xhs))
        p -= np.min(p)
        p /= np.max(p)
        ps.append(p)

    pp = np.vstack(ps)
    plt.figure(figsize=(11,6))
    #plt.imshow(pp,cmap='gray')
    plt.imsave('interp-avoid.png',pp,cmap='gray')

if __name__=='__main__' and 'run' in sys.argv:
    try:
        batches = int(sys.argv[-1]) # user can provide training loops as last arg
    except:
        batches = DEFAULT_BATCHES

    run(batches=batches)
