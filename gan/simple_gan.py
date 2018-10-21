import tensorflow as tf
import numpy as np

img_size = (48,)*2

__chans = 8
'''
For now assume we want same G and D to have layer sizes mirrored.
Then whenever we pool in D, we take a stride step 2 in a conv2d_tranpose in G.

However if we batchnorm at layer 3 in D, we should not do it in the first layer of G,
so batchnorm order is not mirrored
# TODO compare performance

'''
def_d_layers = [
        {"chans": __chans, "pool": True},
        {"chans": __chans, "pool": True},
        {"chans": __chans, "pool": True},
        {"chans": __chans, "pool": False},
        #{"chans": __chans},
        #{"chans": __chans},
        #{"chans": __chans, "batchnorm": True}
]
def_g_lr = .00004
def_d_lr = .00004


# calculate z_size we need for a perfectly sized generator
conv_0_size = img_size[0]
conv_0_size /= np.prod([2 for d in def_d_layers if 'pool' in d and d['pool']])
conv_0_size = int(conv_0_size)
# We will reshape z_scale to this
conv_0_size = (conv_0_size, conv_0_size, def_d_layers[-1]['chans'])
random_z_size = 16
target_z_size = int(np.prod(conv_0_size))

print("Network sizes:")
print(" conv_0   : ",conv_0_size)
print(" target_z : ",target_z_size)


class Harness:
    def __init__(self,
            x_data,
            is_training,
            noise_batch_size,
            g_layers = def_d_layers[::-1],
            d_layers = def_d_layers,
            g_lr = def_g_lr,
            d_lr = def_d_lr,
            activation = tf.nn.relu
            ):
        self.is_training = is_training
        self.x_data = x_data

        z_noise = self.z_noise = tf.random_normal(shape=[noise_batch_size,random_z_size], dtype=tf.float32)
        self.z_given = tf.placeholder(shape=[noise_batch_size,random_z_size], dtype=tf.float32, name='z_given')


        self.d = D(x_data,                d_layers, is_training, lr=d_lr, activation=activation)
        self.g = G(z_noise, self.z_given, g_layers, is_training, lr=g_lr, activation=activation)

        self.d.complete_init( self.g.decode_from_z_noise )
        self.g.complete_init( self.d.g_noise_score )

    def train(self, sess, batches):
        for batch in range(batches):
            # Update D
            for k in range(2):
                d_loss = sess.run(self.d.opt_loss, feed_dict={self.is_training:True})[0]

            # Update G
            g_loss = sess.run(self.g.opt_loss, feed_dict={self.is_training:True})[0]

            if batch % 20 == 0:
                #print("Batch {:<6}: d_loss: {:<8.5f}, g_loss: {:<8.5f}".format(batch,d_loss,g_loss))
                pass

    def save(self, sess):
        save_dir = os.path.join('saves',self.name)
        if os.path.exists(save_dir):
            print("\nDeleting {}!!\n".format(save_dir))
            import shutil
            shutil.rmtree(save_dir)
        tf.saved_model.simple_save(sess, save_dir,
                inputs  = {"x_data": self.x_data, "z_given": self.z_given },
                outputs = {"encode_given_x":self.encode_given_x,
                           "loss_given_x":self.loss_given_x,
                           "decode_given_x":self.decode_given_x,
                           "decode_given_z":self.decode_given_z })

    # Draw from either a given numpy z, or random z's
    def draw_samples(self, sess, num, z=None):
        if z is None:
            decode_op = self.g.decode_from_z_noise
            feed_dict = {}
        else:
            decode_op = self.g.decode_from_z_given 
            feed_dict = {decode_op: z}

        # Note we fix z_noise size at model creation, so we may return less samples then num

        feed_dict.update({self.is_training:False})
        xs = sess.run(decode_op, feed_dict=feed_dict)
        return xs[:num]


class G:
    def __init__(self,
            z_noise, z_given,
            layers, # list of dict {chans: int, pool: bool, batchnorm: bool}
            is_training, # placeholder of tf.bool
            activation=tf.nn.relu,
            lr = .003,
            ):
        self.z_given = z_given
        self.z_noise = z_noise
        self.layers = layers
        self.activation = activation
        self.is_training = is_training
        self.lr = lr

        self.decode_from_z_noise = self.make_decoder(z_noise)
        self.decode_from_z_given = self.make_decoder(z_given, reuse=True)

    def complete_init(self, d_x_g_score):
        # D( G(z) )
        judged = d_x_g_score

        # Maximize score = Minimize -score
        self.loss_from_decode_noise = -tf.reduce_mean(tf.log(d_x_g_score))
                                    #+ tf.square(.5-tf.reduce_mean(self.decode_from_z_noise))*.3
        self.opt_loss = self.make_opt(self.loss_from_decode_noise)
        #self.opt_loss = tf.tuple([self.loss_from_decode_noise], control_inputs=[self.opt_loss])



    def make_decoder(self, z, reuse=None):
        with tf.variable_scope('G/decode', reuse=reuse):
            '''
            net = tf.layers.dense(z, target_z_size)
            net = tf.reshape(net,[-1,*conv_0_size])

            print("decodei n shape ",net.shape)
            
            for (i, layer) in enumerate(self.layers):
                strides = 2 if ('pool' in layer and layer['pool'] == True) else 1
                net = tf.layers.conv2d_transpose(net, layer['chans'], 4, strides, padding='same', activation=self.activation)
                if 'batchnorm' in self.layers[-i-1] and self.layers[-i-1]['batchnorm'] == True:
                    net = tf.layers.batch_normalization(net, training=self.is_training)

            print("decoded shape ",net.shape)
            #net = tf.layers.conv2d_transpose(net, 1, 1, padding='same', activation=None)
            #net = tf.reduce_mean(net, axis=3)
            #net = tf.expand_dims(net, axis=3)
            net = tf.layers.conv2d(net, 1, 1, padding='same', activation=tf.nn.leaky_relu, use_bias=False)
            return net
            '''
            net = tf.nn.relu(tf.layers.dense(z, img_size[0]*img_size[1] // 2))
            net = tf.layers.dense(net, img_size[0]*img_size[1])
            net = tf.reshape(net, [-1, *img_size, 1])
            return net


    def make_opt(self, loss):
        #return tf.train.AdamOptimizer(self.lr).minimize(loss)
        return tf.train.GradientDescentOptimizer(self.lr).minimize(loss)






'''
:x_in should be a [2, B, H, W, C], where the first axis contains, in the first
dimension the true data, and in the second the generated data.
I.e. D should learn to always predict 0 as true.
'''
class D:
    def __init__(self,
            x_data,
            layers,
            is_training,
            lr=.003,
            activation=tf.nn.relu):
        self.x_data = x_data
        self.layers = layers
        self.activation = activation
        self.is_training = is_training
        self.lr = lr

        self.data_score = self.make_judge(x_data, reuse=None)

    def complete_init(self, x_g_from_noise):
        self.g_noise_score = self.make_judge(x_g_from_noise, reuse=True)

        loss_d = - tf.reduce_mean(tf.log(self.data_score + .1))
        loss_g = - tf.reduce_mean(tf.log(1.000001 - self.g_noise_score))
        self.loss = loss_d + loss_g

        self.loss = tf.Print(self.loss,[loss_d,loss_g, tf.reduce_mean(self.data_score),tf.reduce_mean(self.g_noise_score)])

        self.opt_loss = self.make_opt(self.loss)
        self.opt_loss = tf.tuple([self.loss], control_inputs=[self.opt_loss])


    '''
    This should be called __2__ times.
    D(G(z)) is needed in two places, but reuse the same op by
    accessing d.g_logits_from_noise
    '''
    def make_judge(self, x, reuse=None):
        with tf.variable_scope('D/judge', reuse=reuse):
            '''
            net = x
            print('judge in shape ',net.shape)

            for (layer) in self.layers:
                net = tf.layers.conv2d(net, layer['chans'], 3, padding='same', activation=self.activation)
                if 'pool' in layer and layer['pool'] == True:
                    net = tf.layers.max_pooling2d(net, 2,2)
                if 'batchnorm' in layer and layer['batchnorm'] == True:
                    net = tf.layers.batch_normalization(net, training=self.is_training)

            net = tf.layers.dense(net, 1, use_bias = False, activation=None)
            net = tf.nn.sigmoid(net)
            print('judge  shape ',net.shape)
            return net
            '''
            net = tf.reshape(x, [x.shape[0], -1])
            net = tf.nn.relu(tf.layers.dense(net, 30))
            net = tf.nn.sigmoid(tf.layers.dense(net, 1))
            return net

    def make_opt(self, loss):
        #return tf.train.AdamOptimizer(self.lr).minimize(loss)
        return tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

