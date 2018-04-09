import numpy as np
import matplotlib.pyplot as plt




# RBF / squared exp. kernel
# :l controls horizontal scale, :sig controls vertical
def k(a,b, l=1.8, sig=2.0):
    c = np.zeros([a.shape[0],b.shape[0]])
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            c[i,j] = sig * np.exp( (-1./(2.*l**2.)) * (a[i]-b[j])**2.0 )
    return c

# Noise free
def predict_noise_free_gp(xs):
    K    = k(obs_x,obs_x)
    Ks   = k(obs_x, xs)
    Kss  = k(xs,xs)
    Kinv = np.linalg.inv(K)

    mu = np.mean(xs) + Ks.T.dot(Kinv).dot( obs_y - np.mean(obs_x) )
    sig = Kss - Ks.T.dot(Kinv).dot(Ks)

    return mu,sig

# - - - - - - -  Predict w/ noise free gp on some 1d data - - - - - - 

obs_x = np.array([ .1 , .2 , .4 ,  .5 , .7 , .75 , .9])
obs_y = np.array([ .1 , .12 , .3 , .8 , .4 , .35 , .7])

query_x = np.linspace(0,1,100)

mu,sig = predict_noise_free_gp(query_x)
print('mu',mu)
print('sig',sig)

# Plot
def plot_gp(obs_x, obs_y, qx,mu,sig):
    plt.scatter(obs_x,obs_y, marker='x', c='g')

    sigd = np.dot(sig,mu) * 2.0
    plt.plot(qx,mu,'b-', alpha=.6)
    plt.fill_between(qx, mu-sigd, mu+sigd, alpha=.4)

    plt.axis([0,1,-.4,1.4])

    plt.show()

plot_gp(obs_x,obs_y, query_x,mu,sig)

# stopped at pg 519








# 2d old ...

'''
def mahalanobis(x, mu,sig):
    prec = np.linalg.inv(sig)
    det = np.linalg.det(sig)
    return (1./(np.sqrt(2*np.pi)*det)) * np.exp( -(x-mu).dot(prec).dot(x-mu) / (2.*det**2.) )

def f(x):
    mus = np.array([[1,4],[0,-1]])
    sigs = np.stack([ [[1,.2],[.2,1]]  ,np.eye(2)*1.3])
    mixs = [.5,.5]
    return np.sum(mahalanobis(x, mu,sig)*mix for (mu,sig,mix) in zip(mus,sigs,mixs))

x0 = np.array([1,2])
x1 = np.array([5,9])


def plot():
    xs  = np.linspace(-10,10, 100)
    ys  = np.linspace(-10,10, 100)
    xys = np.meshgrid(xs,ys)

    vals = np.apply_along_axis(f, 0, xys)

    #plt.imshow(vals,cmap='gray')
    plt.contourf(vals,cmap='gray')
    plt.show()
'''
