import tensorflow as tf
import numpy as np
import sys

# Data
from tensorflow.examples.tutorials.mnist import input_data

IM_SIZE = [28,28,1]
lr = .01
CLASSES = 10
BATCH_SZ = 24

def mk_net():
    x = tf.placeholder(tf.float32, shape=[None,*IM_SIZE])
    yt = tf.placeholder(tf.int32, shape=[None,CLASSES])

    net = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,1)
    net = tf.layers.conv2d(net, 16, 3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2,1)

    sz = net.shape[1]*net.shape[2]*net.shape[3]
    net = tf.reshape(net, [-1, sz])
    net = tf.layers.dense(net, CLASSES)

    return x,yt,net

def mk_loss(yh, ytrue):
    loss = tf.losses.softmax_cross_entropy(ytrue, yh)
    #opt_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    opt_op = tf.train.AdamOptimizer(lr).minimize(loss)
    ret = tf.tuple([loss], control_inputs=[opt_op])
    return ret


def show_two_digits(a,b):
    pass

def run(sess=None):
    print(" - Start.")
    sess = sess if sess else tf.InteractiveSession()

    mnist = input_data.read_data_sets("/data/mnist/", one_hot=True)

    x_in,yt_in,net = mk_net()
    loss_train = mk_loss(net, yt_in)

    sess.run(tf.global_variables_initializer())

    # loop
    for batch in range(1000):
        bx,by = mnist.train.next_batch(BATCH_SZ)
        bx = bx.reshape([BATCH_SZ, 28,28, 1])
        loss,yh = sess.run([loss_train,net], feed_dict={x_in:bx, yt_in:by});
        loss = loss[0]

        if batch % 10 == 0:
            print("%5d: %.3f"%(batch,loss))
            if batch % 30 == 0:
                print("\t{} -> {}".format(by[0].argmax(), yh[0].argmax()))

    print(" - Done.")



if __name__=='__main__' and 'run' in sys.argv:
    run()
