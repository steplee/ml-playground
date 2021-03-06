import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Implements bare bones dense Lucas Kanade alignment over SE(2) according to the document:
    https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf

All step numbers come from the listing on page 4

It is single scale, not vectorized, and pretty unstable. LR (the step size) must be pretty low.
'''

SZ = 128

IMG = '/data/CelebA/img_align_celeba/062960.jpg'
#IMG = 'data/pelican.jpg'
BORDER = cv2.BORDER_REPLICATE
BORDER = cv2.BORDER_CONSTANT
KNOWN_SCALE = False
BLUR = False

LR = .0000001

def getData(sz=(SZ,SZ)):
    y = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2GRAY)
    y = cv2.resize(y, sz)
    W__ = np.eye(3); W__[0,2] = 5; W__[0,2] = 5
    y = cv2.warpPerspective(y, W__, sz, cv2.INTER_LINEAR, borderMode=BORDER)

    W = np.eye(3)
    W[0,2] = sz[0]/30
    W[1,2] = sz[1]/50
    x = cv2.warpPerspective(y, W, sz, cv2.INTER_LINEAR, borderMode=BORDER)
    return x.astype(np.float64),y.astype(np.float64)

def warp_image(x, W):
    return cv2.warpPerspective(np.copy(x), W, x.shape[0:2], cv2.INTER_LINEAR, borderMode=BORDER)

def compute_jacc(W,xy,p):
    x,y = xy
    return np.array([[x,0,y,0,1,0],
                     [0,x,0,y,0,1]], dtype=np.float64)

kernDelX = np.array([[1, -1]], dtype=np.float64)
kernDelY = np.array([[1, -1]], dtype=np.float64).T

def lk(x,y, p):
    assert(x.shape[0] == x.shape[1])
    h = x.shape[0]
    xx = np.copy(x)
    yy = np.copy(y)
    if BLUR:
        x = cv2.GaussianBlur(x, (5,5), 2,2)
        y = cv2.GaussianBlur(y, (5,5), 2,2)

    ims = []
    Ws = []

    # In the compositional algorithm, W is reused iter to iter and modified.
    W = np.vstack([p.reshape(3,2).T, [0,0,1]])

    for i in range(40):
        print('step',i,'\n',W)

        # 1.
        xw = warp_image(np.copy(x), W)
        ims += [xw]

        # 2.
        err = (y - xw)

        # 3. grad[I] : (2,h,h)
        di = np.stack([cv2.filter2D(x, -1, kernDelX),
                       cv2.filter2D(x, -1, kernDelY)])
        dwi = np.stack([warp_image(di[0], W),
                        warp_image(di[1], W)])

        xs = ys = np.arange(0,h,1,dtype=int)
        xys = np.stack(np.meshgrid(xs,ys)).reshape([2,-1]).T

        # The paper lists the algorithm as sets of intemediaries & sums over space,
        # a python implementation is more efficient doing accumulating in a single loop over all XY.
        H = np.zeros([6,6], dtype=np.float64)
        sd_err_sum = np.zeros([6], dtype=np.float64)

        for xy in xys:
            # 4.
            #if (dwi[:,xy[1],xy[0]].sum() > 20): # only count high gradient areas
                Jwp = compute_jacc(W,xy,None)

                # 5.
                sd = np.dot(dwi[:, xy[1],xy[0]], Jwp) / (256)

                # 6.
                H += np.outer(sd,sd) / len(xys)

                # 7.
                sd_err_sum += sd.T.dot(err[xy[0],xy[1]])

        #print('H',H)
        step = np.linalg.inv(H).dot(sd_err_sum).astype(np.float64) * LR
        print(' - {} step: {}'.format(i, step))
        #p -= step
        dp = np.array([1,0,0,1,0,0]) + step

        if KNOWN_SCALE:
            dp[0:4] *= 1.414/np.linalg.norm(p[0:4].reshape([2,2])) # rotation matrix constraint

        W = W.dot(np.vstack([dp.reshape(3,2).T, [0,0,1]]))
        Ws += [W]


    # Viz.

    dbg = np.vstack(ims)
    dbg = np.vstack([
                      np.hstack([xx,yy]),
                      np.hstack([warp_image(xx,Ws[0]),warp_image(xx,Ws[-1])]),
                      np.hstack([warp_image(xx,Ws[0])//2+yy//2, warp_image(xx,Ws[-1])//2+yy//2])
    ])
    plt.imshow(dbg,cmap='gray')
    plt.text(0,10, 'corrupted'); plt.text(h+10,10, 'truth')
    plt.text(0,h+10, 'first estimate'); plt.text(h+10,h+10, 'final estimate')
    plt.text(0,2*h+10, 'corrupted overlay'); plt.text(h+10,2*h+10, 'final estimate overlay')
    plt.show()


x,y = getData()
#plt.imshow(y),plt.show()

#iniW = np.eye(3).astype(np.float64)
ini_p = np.array([1,0,0,1,0,0], dtype=np.float64)
lk(x,y, ini_p)
