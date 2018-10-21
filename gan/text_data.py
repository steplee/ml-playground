import numpy as np
import sys,random
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

''' Renders unicode strings to numpy arrays. '''

# https://github.com/adobe-fonts/source-han-serif/blob/release/OTF/Korean/SourceHanSerifK-Regular.otf
font_path = "/data/SourceHanSerif-Regular.otf"
prop = mfm.FontProperties(fname=font_path)

def gen_chars(n,size,start=20272):
    chars = [str(chr(q)) for q in range(start,start+n)]

    ims = np.zeros([n,size,size,1])
    font_size = 150

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    fig = plt.figure(figsize=(4,4), dpi=size/4)

    for i in range(n):
        plt.axis('off')
        plt.annotate(chars[i], [.5,.5], fontsize=font_size, fontproperties=prop,va='center',ha='center')
        fig.canvas.draw()
        #print(np.array(fig.canvas.renderer._renderer).shape)
        im = np.array(fig.canvas.renderer._renderer)[...,1:2]
        im = 1. - im/255.
        im = np.clip(im,0,1.)
        ims[i] = im
        fig.clear()

    plt.close(fig)
    del fig

    return ims

# Returns a tf.dataset
def gen_dataset(n, size, batch_size):
    import tensorflow as tf

    xs = gen_chars(n, size)
    ys = list(range(len(xs)))

    def xform(x):
        box = tf.concat([tf.random_uniform([2],.0,.2), tf.random_uniform([2],.7,1.0)], 0)
        box = tf.expand_dims(box,0)
        box = tf.tile(box, [batch_size,1])
        binds = tf.constant(list(range(batch_size)))
        x = tf.image.crop_and_resize(x, box, binds, (x.shape[1:3]))
        x = tf.contrib.image.rotate(x, tf.random_uniform([batch_size], -.3,.3))
        x = tf.contrib.image.translate(x, tf.random_uniform([batch_size,2], -10.,10.))
        return x


    ten = tf.data.Dataset.from_tensor_slices(xs)
    ten = ten.shuffle(buffer_size=n)
    ten = ten.batch(batch_size)
    ten = ten.map(map_func=xform) # yes, this is single-threaded
    ten = ten.repeat()

    return ten


def test(n=100):
    gen = gen_char_gen(n, 100)
    bx,by = gen(5)

    p = np.hstack(list(bx[...,0]))

    plt.imshow(p,cmap='gray')
    plt.show()

if __name__=='__main__' and 'test_text' in sys.argv:
    test()
