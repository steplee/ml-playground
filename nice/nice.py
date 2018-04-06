import numpy as np
import sys
import matplotlib.pyplot as plt

# TODO use pytorch or tf for gradients ...

# Data
from tensorflow.examples.tutorials.mnist import input_data


try:
    __mnist
except:
    __mnist = input_data.read_data_sets("/data/mnist/", one_hot=True)



# Additive coupling layers
# x is batched vector
# :m is forward function
# :mode is either `forward` or `reverse`
def additive_couple(x, m, mode='forward'):
    B = x.shape[0] # batch size
    D = x.shape[1] # dimensionality
    d = x.shape[1]//2

    if mode == 'forward':
        y1 = x[:,0:d]
        y2 = x[:,d:] + m(y1)
    else:
        y1 = x[:,0:d]
        y2 = x[:,d:] - m(y1)

    y = np.hstack([y1,y2])

    Sii = np.ones(B) * 1.0
    y = y.T
    y *= Sii
    y = y.T

    return y, Sii


m1 = np.square

log2pi = np.log(np.pi*2.0)

def nice(batch_size=10):

    mnist = __mnist
    batches = 600

    D = 784
    theta = (np.random.normal(size=(D//2,D//2)))  *.01
    theta2 = (np.random.normal(size=(D//2,D//2))) *.01
    def my_m(x):
        return np.dot(x,theta)
    def my_m2(x):
        return np.dot(x,theta2)

    mix_inds = list(range(D//2,D))+list(range(D//2))

    for batch in range(batches):
        bx,by = mnist.train.next_batch(batch_size)
        for epoch in range(1):

            #m = np.square
            m = my_m
            m2 = my_m2

            # Map to z-space
            fx,det = additive_couple(bx, m, mode="forward")
            # swap and redo
            bx2 = fx[:,mix_inds]
            fx2,det = additive_couple(bx2, m2, mode="forward")

            # Eval prior & l
            logdet = np.log(np.abs(det))
            #print('det',logdet)
            px = (-1/2) * np.sum(fx2, axis=1) + log2pi*(-D/2)

            # Use px as loss and back-prop
            lr = .0006
            theta2 -= lr*np.dot(fx2[:,:D//2].T,bx2[:,:D//2])
            theta -= lr*np.dot(fx2[:,:D//2].T, bx[:,D//2:])


            print(np.mean(px))
            print("weights",np.abs(theta).mean())

    # Map back to x-space, then plot
    #fx_corrupt = fx2 + np.random.normal(size=fx.shape) * .01
    fx_corrupt = fx2 
    fx_inv,det2 = additive_couple(fx_corrupt, m2, mode="backward")
    fx_inv = fx_inv[:,mix_inds]
    fx_inv,det2 = additive_couple(fx_inv, m, mode="backward")
    p_bx = bx.clip(0,1).reshape([batch_size, 28,28, 1])
    p_fx_inv = fx_inv.clip(0,1).reshape([batch_size, 28,28, 1])
    #mk_grid(list(p_bx) + list(p_fx_inv))


    a = np.hstack(list(p_bx))
    b = np.hstack(list(p_fx_inv))
    ab = np.vstack([a,b])

    ab = ab[...,0] # pyplot doesn't like 1-dim channel
    plt.imshow(ab, cmap='gray', interpolation='none')
    plt.show()


    # Do we do well on fake data?
    bx_fake = (bx*(.5+np.random.normal(size=bx.shape)) * 2.0).clip(0,1)
    fx,det = additive_couple(bx_fake, m, mode="forward")
    bx_fake2 = fx[:,mix_inds]
    fx2,det = additive_couple(bx_fake2, m2, mode="forward")
    px = (-1/2) * np.sum(fx2, axis=1) + log2pi*(-D/2)
    print("Fake data score: {}".format(px.mean()))



if __name__=='__main__' and 'run' in sys.argv:
    #nice()

    import tangent
    print(tangent.grad(additive_couple))

