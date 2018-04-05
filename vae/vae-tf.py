import tensorflow as tf
import numpy as np
import sys,random
import matplotlib.pyplot as plt

# Data
#from tensorflow.examples.tutorials.mnist import input_data

CHAR_SIZE = 56
#IM_SIZE = [28,28,1]
IM_SIZE = [CHAR_SIZE,CHAR_SIZE,1]
lr = .0003
CLASSES = 10
BATCH_SZ = 24
LATENT_DIMS = 49

import matplotlib.font_manager as mfm

font_path = "/data/SourceHanSerif-Regular.otf"
prop = mfm.FontProperties(fname=font_path)

# Returns a batch sampler function
def gen_char_gen(n=50, size=CHAR_SIZE):

    start = 20272
    chars = [str(chr(q)) for q in range(start,start+n)]

    ims = np.zeros([n,size,size,1])
    inset = .1
    #font_range = (9,13+1)
    #font_range = (2,3)
    font_range = (120,200)
    char_w = (.2,.3)

    fig = plt.figure(figsize=(4,4), dpi=size/4)

    for i in range(n):
        s = np.random.randint(*font_range)
        zz = s/font_range[1]*char_w[1] + (1-s/font_range[1])*char_w[0]
        x = np.random.uniform(0,1-zz)
        y = np.random.uniform(0,1-zz)
        plt.text(x,y,chars[i], fontsize=s, fontproperties=prop)
        fig.canvas.draw()
        #print(np.array(fig.canvas.renderer._renderer).shape)
        im = np.array(fig.canvas.renderer._renderer)[...,0:1]
        ims[i] = (im.max() - im) / im.max()
        fig.clear()

    plt.close(fig)
    del fig

    def gene(batch_size):
        inds = np.random.choice((range(n)), batch_size)
        x = ims[inds]
        #f,a=plt.subplots(figsize=(5,5))
        #a.imshow(x[0,...,0])
        #plt.show()
        return x,inds

    return gene

# --------------------------------------------------------------------------

def mk_ae_net():
    x = tf.placeholder(tf.float32, shape=[None,*IM_SIZE])

    # Conv (encoder)
    net = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,2)
    net = tf.layers.conv2d(net, 8, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,2)
    net = tf.layers.batch_normalization(net)

    # FC 
    sz = net.shape[1]*net.shape[2]*net.shape[3]
    net = tf.reshape(net, [-1, sz])

    # Reparameterization:
    #   z = m + s*epsilon
    mu  = tf.layers.dense(net, LATENT_DIMS, name='mu_op', use_bias=False)
    sig = tf.layers.dense(net, LATENT_DIMS, name='sig_op', use_bias=False, activation=tf.nn.relu)
    sig += tf.ones(LATENT_DIMS) * .1
    mu  = tf.identity(mu, name='mu')
    sig = tf.identity(sig, name='sig')
    eps = tf.random_normal(shape=[LATENT_DIMS], name='eps')
    z = tf.add(mu, sig * eps, name='z_op')
    z = tf.identity(z,name='z')

    # Deconv (decoder)
    net = tf.layers.dense(z, sz)
    net = tf.reshape(net, [BATCH_SZ, IM_SIZE[0]//4,IM_SIZE[1]//4,8])
    net = tf.layers.conv2d_transpose(net, 16, 4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d_transpose(net, 4, 4, strides=2, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 1, 1, padding='same', activation=tf.nn.relu)

    return x,z,net

def mk_loss(xh, xt):
    # log p = E[ logp(x,z) - logq(z|x) ]   -   KL[q(z|x) | p(z|x)] 
    #    ELBO __________________________

    z = tf.get_default_graph().get_tensor_by_name('z:0')
    eps = tf.get_default_graph().get_tensor_by_name('eps:0')
    sig = tf.get_default_graph().get_tensor_by_name('sig:0')

    d = LATENT_DIMS

    # z ~ N(mu,sig)
    log_pz = -1/(2*np.pi) - d - tf.reduce_sum(tf.square(eps))/2
    log_qzx = log_pz - tf.reduce_sum(tf.log(sig))

    # Recons score
    #log_pzx = tf.losses.sigmoid_cross_entropy(xt>.5,xh)
    recons_loss  = tf.losses.mean_squared_error(xt,xh)

    # Negative loss because we are minizming w/ sgd (want to maxim)
    loss0 = tf.reduce_sum(recons_loss)
    loss1 = tf.reduce_mean(log_qzx) * .0000001
    loss = loss0 + loss1

    loss = tf.Print(loss,[loss0,loss1])

    opt_op = tf.train.AdamOptimizer(lr).minimize(loss)
    ret = tf.tuple([loss], control_inputs=[opt_op])
    return ret

def run(sess=None):
    print(" - Start.")
    sess = sess if sess else tf.InteractiveSession()

    #mnist = input_data.read_data_sets("/data/mnist/", one_hot=True)

    x_in,z,net = mk_ae_net()
    loss_train = mk_loss(net, x_in)

    sess.run(tf.global_variables_initializer())

    char_gen = gen_char_gen(10)

    # loop
    for batch in range(660):
        #bx,by = mnist.train.next_batch(BATCH_SZ)
        #bx = bx.reshape([BATCH_SZ, 28,28, 1])
        bx,by = char_gen(BATCH_SZ)

        loss,xh = sess.run([loss_train,net], feed_dict={x_in:bx})
        loss = loss[0]

        if batch % 10 == 0:
            print("%5d: %.3f"%(batch,loss))
            '''
            if batch % 200 == 0:
                p0 = bx[0,:,:,0]
                p1 = xh[0,:,:,0]
                p = np.hstack([p0,p1])
                plt.imshow(p,cmap='gray')
                plt.show()
                #print("\t{} -> {}".format(by[0].argmax(), yh[0].argmax()))
            '''

    print(" - Done.")

    N = BATCH_SZ
    ps = []
    for j in range(N-1):
        a,b = bx[j],bx[j+1]
        path = np.linspace(0,1,N)
        cs = [path[i]*a + (1-path[i])*b for i in range(N)]

        xhs = list(sess.run([net], feed_dict={x_in:cs})[0][...,0])

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
    run()
