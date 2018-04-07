import numpy as np
import sys,random
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

''' Renders unicode strings to numpy arrays. '''

# https://github.com/adobe-fonts/source-han-serif/blob/release/OTF/Korean/SourceHanSerifK-Regular.otf
font_path = "/data/SourceHanSerif-Regular.otf"
prop = mfm.FontProperties(fname=font_path)

# Returns a batch sampler function
def gen_char_gen(n, size):

    start = 20272
    chars = [str(chr(q)) for q in range(start,start+n)]

    ims = np.zeros([n,size,size,1])
    font_size = 130

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

    def gene(batch_size):
        inds = np.random.choice((range(n)), batch_size)
        x = ims[inds]
        #f,a=plt.subplots(figsize=(5,5))
        #a.imshow(x[0,...,0])
        #plt.show()
        return x,inds

    return gene

def test(n=100):
    gen = gen_char_gen(n, 100)
    bx,by = gen(5)

    p = np.hstack(list(bx[...,0]))

    plt.imshow(p,cmap='gray')
    plt.show()

if __name__=='__main__' and 'test_text' in sys.argv:
    test()
