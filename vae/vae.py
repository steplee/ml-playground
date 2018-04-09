import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import sys,random,os

'''
VAE
'''



# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
#                      MODEL 
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 



'''
Endpoints:
    encode_given_x
    decode_given_z
    decode_given_x
    loss_given_x
    opt_loss_given_x (which returns loss)

'''
class vae_cnn:
    def __init__(self,
            x_in,
            latent_size,
            conv_channels=[16,16],
            lr=.003,
            debug_print=False,
            name='vae'):

        self.x_in = x_in
        self.in_size = list(map(int,x_in.shape[1:]))
        self.lr = tf.Variable(lr, trainable=False, name='lr')
        self.conv_channels = conv_channels
        self.latent_size = latent_size
        self.debug_print = debug_print
        self.name = name

        #self.x_in = tf.placeholder(tf.float32, shape=[None,*in_size], name='x_in')

        # Use these to train and reconstruct, evaluates full model.
        self.encode_given_x, self.mu_x, self.sig_x, \
                             self.conv_0_size         = self.make_encoder(self.x_in, name='encode_given_x')
        self.decode_given_x                           = self.make_decoder(self.encode_given_x,name='decode_given_x')
        self.loss_given_x,self.opt_loss_given_x  = self.make_loss(self.x_in, self.decode_given_x,
                                                                   self.mu_x, self.sig_x)

        # Use this to sample from  p(z | x), it evaluates the latter half of the model.
        self.z_in = tf.placeholder(tf.float32, shape=[None,latent_size], name='z_in')
        self.decode_given_z = self.make_decoder(self.z_in, reuse=True, name='decode_given_z')



    def make_encoder(self, x, name, reuse=False):
        layers = self.conv_channels
        with tf.variable_scope('encoder',reuse=reuse):
            net = x 
            for i,chans in enumerate(layers):
                net = tf.layers.conv2d(net, chans, 3, padding='same', activation=tf.nn.relu)
                net = tf.layers.max_pooling2d(net, 2,2)

            conv_0_size = net.shape[1:]
            flat_size = np.prod(conv_0_size)
            net = tf.reshape(net, [-1, flat_size])
            net = tf.layers.dense(net, self.latent_size)

            mu  = tf.layers.dense(net, self.latent_size, name='mu_op', use_bias=False)
            sig = tf.layers.dense(net, self.latent_size, name='sig_op', activation=tf.nn.softplus) + 1e-4
            eps = tf.random_normal(shape=[self.latent_size], name='eps')
            z   = tf.add(mu, sig * eps, name='z_op')

        return tf.identity(z,name=name), mu,sig, conv_0_size

    def make_decoder(self, z, name, reuse=False):
        layers = self.conv_channels
        with tf.variable_scope("decoder",reuse=reuse):
            # latent -> conv
            flat_size = np.prod(self.conv_0_size)
            net = tf.layers.dense(z, flat_size)
            net = tf.reshape(net, ([-1, *self.conv_0_size]))

            for i,chans in enumerate(layers[::-1]):
                net = tf.layers.conv2d_transpose(net, chans, 4, strides=2, padding='same', activation=tf.nn.relu)

            net = tf.layers.conv2d(net, 1, self.in_size[2], padding='same', activation=tf.nn.sigmoid)
            net = tf.clip_by_value(net, 1e-6,1-1e-6)

        return tf.identity(net, name=name)

    def make_loss(self, x_true, x_recons, mu, sig):
        recons = tf.reduce_sum(x_true*tf.log(x_recons)
                            + (1-x_true)*tf.log(1-x_recons), 1)
        recons = tf.reduce_mean(recons)

        likeli = .5 * tf.reduce_mean(1. + tf.log(tf.square(sig)) - tf.square(mu) - tf.square(sig))

        loss = - (likeli + recons*2)


        if self.debug_print:
            loss = tf.Print(vae_,[likeli,recons])

        opt_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        opt_and_loss = tf.tuple([loss], control_inputs=[opt_op]) # Evaluating loss => evaluating optimization

        return tf.identity(loss,'loss_given_x'), tf.identity(opt_and_loss, 'opt_loss_given_x')


    def save(self, sess):
        save_dir = os.path.join('saves',self.name)
        if os.path.exists(save_dir):
            print("\nDeleting {}!!\n".format(save_dir))
            import shutil
            shutil.rmtree(save_dir)
        tf.saved_model.simple_save(sess, save_dir,
                inputs  = {"x_in": self.x_in, "z_in": self.z_in },
                outputs = {"encode_given_x":self.encode_given_x,
                           "loss_given_x":self.loss_given_x,
                           "decode_given_x":self.decode_given_x,
                           "decode_given_z":self.decode_given_z })


