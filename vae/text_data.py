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
    inset = .1
    #font_range = (9,13+1)
    #font_range = (2,3)
    font_range = (120,200)
    char_w = (.2,.3)

    fig = plt.figure(figsize=(4,4), dpi=size/4)

    for i in range(n):
        s = np.random.randint(*font_range)
        zz = s/font_range[1]*char_w[1] + (1-s/font_range[1])*char_w[0]
        x = np.random.uniform(0,1-zz)
        y = np.random.uniform(0,1-zz)
        #plt.text(x,y,chars[i], fontsize=s, fontproperties=prop)
        plt.annotate(chars[i], [x,y], fontsize=s, fontproperties=prop)
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
