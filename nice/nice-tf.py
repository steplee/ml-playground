import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import tensorflow.contrib.eager as tfe

lr = .006
batch_size = 30

BATCHES = 5000

D = DIMS = 784
mask1 = set(np.random.choice(range(D), D//2))
mask1 = np.array([1. if i in mask1 else 0 for i in range(D)])
mask2 = set(np.random.choice(range(D), D//2))
mask2 = np.array([1. if i in mask2 else 0 for i in range(D)])

log2pi = np.log(2.0*np.pi)

def additive_couple(x, m,Sii, mask, mode='forward'):
    if mode == 'forward':
        y1 = x * (1-mask)
        #y2 = (x + m(y1)**2.) * mask
        y2 = (x + m(y1)) * mask
        y = y1 + (y2)
    else:
        y1 = x * (1-mask)
        #y2 = (x - tf.sqrt(m(y1))) * mask
        y2 = (x - (m(y1))) * mask
        y = y1 + (y2)

    Sii = (tf.ones([D]))
    y = y * Sii # TODO

    det = tf.reduce_sum(Sii)
    #det = tf.constant(1.0)

    return y, det


def mk_nice():
    x = tf.placeholder(tf.float32, shape=[None,DIMS])

    d = DIMS//2

    with tf.variable_scope('net'):
        w1 = tf.get_variable('w1', shape=[D,D], dtype=tf.float32)
        w2 = tf.get_variable('w2', shape=[D,D], dtype=tf.float32)
        b1 = tf.get_variable('b1', shape=[D], dtype=tf.float32)
        b2 = tf.get_variable('b2', shape=[D], dtype=tf.float32)
        # TODO not used, instead eye is used
        Sii1 = tf.Variable(tf.random_uniform([int(DIMS)],.8,1.2), dtype=tf.float32)
        Sii2 = tf.Variable(tf.random_uniform([int(DIMS)],.8,1.2), dtype=tf.float32)

        #act = tf.nn.sigmoid
        act = tf.nn.softplus
        #act = tf.nn.relu

        def m1(z):
            #return act(tf.matmul(z,w1) + b1 )
            return act(tf.matmul(z,w1)  )
            #return z
        def m2(z):
            return act(tf.matmul(z,w2) )

        enc,det1 = additive_couple(x, m1,Sii1,   mask1, 'forward')
        enc,det2 = additive_couple(enc, m2,Sii2, mask2, 'forward')

        def decode(z,reuse=None):
            with tf.variable_scope('decode', reuse=reuse):
                dec,det3 = additive_couple(z, m2,Sii2,   mask2, 'backward')
                dec,det4 = additive_couple(dec, m1,Sii1, mask1, 'backward')
                return dec, [det3,det4]

        def lossy(z, dets, reuse=False):
            with tf.variable_scope('loss', reuse=reuse):
                log_likeli = tf.log(tf.abs(tf.reduce_prod(dets)))

                log_prior = -tf.reduce_mean((.5) * tf.reduce_sum(tf.square(enc), axis=1) + log2pi*(DIMS/2))

                logpx = log_prior + log_likeli
                loss = -logpx # we want to maximize logpx === minimize -logpx
                return loss


        net,det_dec = decode(enc)

        # We can input a z without encoding
        z_in = tf.placeholder(tf.float32, shape=[None,DIMS])
        dec_from_z,_ = decode(z_in, reuse=True)


        loss = lossy(net, dets=[det1,det2,*det_dec])
        
        opt = tf.train.AdamOptimizer(lr).minimize(loss)

        return x,net,loss,opt, enc, dec_from_z,z_in

def run():
    x,net,loss_op,opt_op,  enc_to_z,dec_from_z,z_in = mk_nice()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/data/mnist/", one_hot=True)

    xhats = []

    for batch in range(BATCHES):
        bx,by = mnist.train.next_batch(batch_size)

        xhat,loss,opt = sess.run([net,loss_op,opt_op], feed_dict={x:bx})
        loss = loss

        if batch % 10 == 0:
            with tf.variable_scope('net',reuse=True):
                norm = sess.run(tf.norm(tf.get_variable('w1')))
            print ("\t {: <5}: \tloss {:.2f} ({})".format(batch,loss,norm))

        if batch % (BATCHES//10) == 0:
            '''local = []
            #for i in range(min(batch_size,6)):
                #im = xhat[i].reshape([28,28])
                #im = im - np.min(im)
                #local.append(im/im.max())
            local = np.vstack(local)
            xhats.append(local)'''
            N = 10
            for j in range(N):
                a,b = bx[j],bx[j+1]
                path = np.linspace(0,1,N)
                za,zb = sess.run(enc_to_z, feed_dict={x:[a,b]})

                cs = [path[i]*za + (1-path[i])*zb for i in range(N)]
                xhs = (sess.run(dec_from_z, feed_dict={z_in:cs}))

                xhs = list(xhs.reshape([10,28,28]))
                p = (np.vstack(xhs))
                p -= np.min(p)
                p /= np.max(p)
                xhats.append(p)

    im = np.hstack(xhats)
    plt.imshow(im,cmap='gray')
    plt.show()



if __name__=='__main__' and 'run' in sys.argv:
    run()
