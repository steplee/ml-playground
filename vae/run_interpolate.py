import tensorflow as tf
import numpy as np
import sys,random
import matplotlib.pyplot as plt

import text_data


import vae


# Train params
debug_print = False
batch_size = 32
default_batches = 9000
show_every = -1

# Data params
classes = 100
char_size = 60

# Model params
the_latent_size = 49*4
the_conv_channels = [16,16]


tf.app.flags.DEFINE_integer("batches", default_batches, "Number of batches to train")
tf.app.flags.DEFINE_string("name", "vae", "Name of model. Used for saving files, etc.")
flags = tf.app.flags.FLAGS


# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                      RUNNER
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

def test_vae(sess=None):
    print(" - Start.")

    net = vae.vae_cnn(in_size=[char_size,char_size,1], latent_size=the_latent_size, batch_size=batch_size)

    sess = sess if sess else tf.Session()
    sess.run(tf.global_variables_initializer())

    # A function, call to get batch
    char_gen = text_data.gen_char_gen(classes, char_size)

    # Train loop
    for batch in range(flags.batches):
        bx,by = char_gen(batch_size)

        loss,xh = sess.run([net.opt_loss_given_x, net.decode_given_x], feed_dict={net.x_in:bx})
        loss = loss[0]

        if batch % 10 == 0:
            print("%5d: %.3f"%(batch,loss))
            if show_every > 0 and batch % show_every == 0:
                p0 = bx[0,:,:,0]
                p1 = xh[0,:,:,0]
                p = np.hstack([p0,p1])
                plt.imshow(p,cmap='gray')
                plt.show()



    ''' 
    Create image showing interpolation in z-space
        The top half will be an interpolated line from two actual values.
        The bottom half will be a real point progressively going further away.
    '''
    N = batch_size
    ps = []
    for j in range(N-1):
        a,b = bx[j],bx[j+1]
        zs = sess.run(net.encode_given_x, feed_dict={net.x_in:bx})
        if j<=N//2:
            za,zb = zs[0],zs[1]
            if j==N//2:
                ps.append(np.ones_like(ps[-1])*.01) # put a break
        else:
            r = np.random.uniform(size=zs[0].shape) * 5.0
            za,zb = zs[0],zs[0] + r*r

        path = np.linspace(1,0,N)
        cs = [path[i]*za + (1-path[i])*zb for i in range(N)]

        xhs = list(sess.run(net.decode_given_z, feed_dict={net.z_in:cs})[...,0])

        p = (np.hstack(xhs))
        p -= np.min(p)
        p /= np.max(p)
        ps.append(p)

    pp = np.vstack(ps)
    plt.figure(figsize=(11,6))
    #plt.imshow(pp,cmap='gray')
    plt.imsave('{}-interp.png'.format(flags.name),pp,cmap='gray')
    print(" - Done, created {}-interp.png".format(flags.name))

if __name__=='__main__' and 'run' in sys.argv:
    test_vae()

