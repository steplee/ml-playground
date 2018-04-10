import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb


# Data distribution
x_size = 2
dispersion = .1
K = 4

# TODO compute points on hypersphere
mus = np.array([
    [1,1], [-1,1], [-1,-1], [1,-1]
])
sigs = np.stack([np.eye(2)*dispersion for _ in range(K)])

# Utilities
def get_batch(bs):
    divs = bs//K
    xs = np.concatenate([np.random.multivariate_normal(mus[i],sigs[i],size=divs) for i in range(K)])
    ys = np.concatenate([[i]*divs for i in range(K)])
    inds = list(range(bs))
    np.random.shuffle(inds)
    return xs[inds], ys[inds]

# Plot 2d vectors by color of class. xs and ys should be in the same order.
def plot_with_true_classes(xs,ys,opacity=1.):
    xss = [np.array([xi for xi,yi in zip(xs,ys) if yi==c]) for c in range(K)]
    colors = cm.get_cmap('jet')(np.linspace(0,1,K))
    for i,xs in enumerate(xss):
        plt.scatter(xs[:,0],xs[:,1], c=colors[i], alpha=opacity)


# Model Params
hid_size = 40
latent_size = 2
act = torch.nn.ReLU

# Train params
batch_size = 32
batches = 5000
lr = .003

def run():

    # Construct.

    encode = torch.nn.Sequential(
            nn.Linear(x_size, hid_size),
            act,
            nn.Linear(hid_size,latent_size*2) )
    decode = torch.nn.Sequential(
            nn.Linear(latent_size, hid_size),
            act,
            nn.Linear(hid_size, x_size))

    opt = torch.optim.Adam(list(encode.parameters())+list(decode.parameters()))

    # Train.

    for batch in range(batches):
        bx,by = get_batch(batch_size)
        bx = torch.autograd.Variable(torch.FloatTensor(bx),requires_grad=False)

        mu_sig = (encode(bx))
        mu,sig = mu_sig[:,:latent_size],mu_sig[:,latent_size:]

        mu_0 = np.zeros([batch_size,latent_size])
        eps = torch.autograd.Variable(torch.normal(means=torch.FloatTensor(mu_0)),
                requires_grad=False)
        z = mu + sig * eps

        xhat = (decode(z))

        # I don't think mse has any probablistic interpretation, but it seems to work.
        prior_fit = .5 * torch.mean(1 - torch.mean(mu*mu) + torch.log(sig*sig) - sig*sig )
        recon_fit = -F.mse_loss(xhat, bx)
        fit = prior_fit + recon_fit
        loss = -fit

        if batch % (batches//20) == 0:
            print(" {}: {}".format(batch,loss.data.numpy()[0]))

        loss.backward()
        opt.step()
        opt.zero_grad()


    # Analyze.

    xhat = xhat.data.numpy()
    plot_with_true_classes(xhat,by)
    plot_with_true_classes(bx.data.numpy(),by, opacity=.4)
    plt.title("Reconstruction")
    plt.show()

    plt.title("z-space")
    plot_with_true_classes(z.data.numpy(),by, opacity=1.0)
    plt.show()

    # TODO compute p(x)
    '''
    # Let's fully see z-space 

    Q = 50
    argmax_at_point = np.zeros([Q,Q])
    xy = np.meshgrid(*[np.linspace(-2,2,50),]*2)
    xy = np.stack([xy.reshape([-1]) for xy in xy])
    xy = torch.autograd.Variable(torch.FloatTensor(xy),requires_grad=False)

    # p(x) ~= KL(q,p) + 
    # p(x|z) = p(z|x) p(x) = p(x,z)p(x) / p(z)
    x_from_z = decode(xy)
    '''

run()
