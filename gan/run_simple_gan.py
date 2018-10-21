import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import text_data
import simple_gan

img_size = (48,)*2

batch_size = 32

def run():

    x_data = text_data.gen_dataset(40, img_size[0], batch_size)
    x_data = x_data.make_one_shot_iterator()
    x_data = x_data.get_next(name='x_data') # NOTE: name is crucial when loading!

    is_training = tf.placeholder(tf.bool)

    harness = simple_gan.Harness(x_data, is_training, noise_batch_size=batch_size)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    try:
        harness.train(sess, batches=70000)
    except KeyboardInterrupt:
        pass

    # Produce viz.
    want_num = 10
    xs = harness.draw_samples(sess, num=batch_size)[:want_num]
    #xs = sess.run(x_data)

    print ("sample mean brightness: {}".format(np.mean(xs)))

    xs = np.vstack(xs)[...,0] # pyplot expects no channel if grayscale
    xs = np.clip(xs,0,1)
    plt.imshow(xs, cmap='gray')
    plt.show()




if __name__=='__main__':
    run()
