import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

# Data
from tensorflow.examples.tutorials.mnist import input_data

IM_SIZE = [28,28,1]
lr = .003
CLASSES = 10
BATCH_SZ = 24
LATENT_DIMS = 49

'''
import matplotlib.font_manager as mfm

font_path = "/data/SourceHansSerif-Regular.otf"
prop = mfm.FontProperties(fname=font_path)
#plt.text(0.5, 0.5, s=u'测试', fontproperties=prop)

def gen_char_gen(n=50)
    #plt.text(0.5, 0.5, s=u'测试', fontproperties=prop)
    start = 20272
    chars = ['\u'+hex(q) for q in range(start,start+n)]
    ims = 

    def gene():
        random.sample(chars)
'''


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
    mu  = tf.layers.dense(net, LATENT_DIMS)
    sig = tf.layers.dense(net, LATENT_DIMS)
    z = mu + sig * tf.random_normal(shape=[LATENT_DIMS])

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

    loss = tf.losses.mean_squared_error(xh, xt)
    #opt_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    opt_op = tf.train.AdamOptimizer(lr).minimize(loss)
    ret = tf.tuple([loss], control_inputs=[opt_op])
    return ret

def run(sess=None):
    print(" - Start.")
    sess = sess if sess else tf.InteractiveSession()

    mnist = input_data.read_data_sets("/data/mnist/", one_hot=True)

    x_in,z,net = mk_ae_net()
    loss_train = mk_loss(net, x_in)

    sess.run(tf.global_variables_initializer())

    # loop
    for batch in range(1200):
        bx,by = mnist.train.next_batch(BATCH_SZ)
        bx = bx.reshape([BATCH_SZ, 28,28, 1])

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
    a,b = bx[0],bx[1]
    path = np.linspace(0,1,N)
    cs = [path[i]*a + (1-path[i])*b for i in range(N)]

    xhs = list(sess.run([net], feed_dict={x_in:cs})[0][...,0])

    p = (np.hstack(xhs))
    plt.figure(figsize=(11,6))
    plt.imshow(p,cmap='gray')
    plt.imsave('interp.png',p,cmap='gray')
    plt.show()


if __name__=='__main__' and 'run' in sys.argv:
    run()
