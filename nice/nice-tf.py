import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import tensorflow.contrib.eager as tfe

lr = .00001
batch_size = 30

BATCHES = 1000

layers = 6
D = DIMS = 784



log2pi = np.log(2.0*np.pi)

def additive_couple(x, m,Sii, mask, mode='forward'):
    if mode == 'forward':
        y1 = x * (1-mask)
        y2 = (x + m(y1)) * mask
        y = y1 + y2
    else:
        y1 = x * (1-mask)
        y2 = (x - m(y1)) * mask
        y = y1 + y2

    y = y * Sii

    det = tf.reduce_sum(Sii) # determinant is sum of diagonals

    return y, det


def mk_nice():
    x = tf.placeholder(tf.float32, shape=[None,DIMS])

    d = DIMS//2

    mask=[]
    for i in range(layers):
        m = set(np.random.choice(range(D), D//2))
        m = np.array([1. if i in m else 0 for i in range(D)])
        mask.append(tf.constant(m,dtype=tf.float32))

    with tf.variable_scope('net'):

        #act = tf.nn.sigmoid
        #act = tf.nn.softplus
        act = tf.nn.relu

        def m1(w,b):
            def m2(z):
                #return act(tf.matmul(z,w1) + b1 )
                return act(tf.matmul(z,w))
            return m2

        # Params
        w,b,Sii = [],[],[]
        for i in range(layers):
            wi = tf.get_variable('w'+str(i), shape=[D,D], dtype=tf.float32)
            bi = tf.get_variable('b'+str(i), shape=[D], dtype=tf.float32)
            si = tf.Variable(tf.random_uniform([int(DIMS)],.8,1.2), dtype=tf.float32)
            w.append(wi); b.append(bi); Sii.append(si)


        def encode(x,reuse=None):
            with tf.variable_scope('encode', reuse=reuse):
                enc = x
                dets = []
                for i in range(layers):
                    enc,det = additive_couple(enc, m1(w[i],b[i]), Sii[i], mask[i], 'forward')
                    dets.append(det)
                return enc,dets

        def decode(z,reuse=None):
            with tf.variable_scope('decode', reuse=reuse):
                dets = []
                dec = z
                for i in range(layers-1,-1,-1):
                #for i in range(layers):
                    dec,det = additive_couple(dec, m1(w[i],b[i]), Sii[i], mask[i], 'backward')
                    dets.append(det)
                return dec, dets

        def lossy(z, enc,dets, reuse=False):
            with tf.variable_scope('loss', reuse=reuse):
                log_likeli = tf.log(tf.abs(tf.reduce_prod(dets)))

                #log_prior = tf.reduce_sum(-tf.log(1+tf.exp(enc)) - tf.log(1+tf.exp(-enc)))
                log_prior = -tf.reduce_mean((.5) * tf.reduce_sum(tf.square(enc), axis=1) + log2pi*(DIMS/2))

                logpx = log_prior + log_likeli
                loss = -logpx # we want to maximize logpx === minimize -logpx
                return loss


        enc,det_enc = encode(x)
        dec,det_dec = decode(enc)
        dets = det_enc+det_dec

        # We can input a z without encoding
        z_in = tf.placeholder(tf.float32, shape=[None,DIMS])
        dec_from_z,_ = decode(z_in, reuse=True)


        loss = lossy(dec, enc,dets=dets)
        
        opt = tf.train.AdamOptimizer(lr).minimize(loss)

        return x,dec,loss,opt, enc, dec_from_z,z_in

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
