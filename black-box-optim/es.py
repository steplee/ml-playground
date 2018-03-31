import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal as mvn

D = 2
S = 50
lr = .1

true_mu = np.array([1,2])
true_sig = np.array([2,1])

# lets have a Gaussian at mu=(1,2)
def f(x):
    det = np.sum(true_sig)
    dx = x-true_mu
    return np.sqrt(2*np.pi)*det   *   np.exp(-np.linalg.norm( dx.dot(np.linalg.inv(true_sig)).dot(dx) ) )

def reward(x):
    mu = np.array([1,2])
    return -np.linalg.norm(x-mu)

def loss(x):
    mu = np.array([1,2000])
    return np.linalg.norm(x-mu)



def naive_es(theta):
    (mu,sig) = theta
    print("Naive ES:")
    print("Starting at {}, with reward {}".format(str(mu), reward(mu)))

    steps = 300

    for step in range(steps):
        # Samples
        s = np.random.normal(size=(S,D))
        #s = sig.dot(s)
        #s += mu

        # Gradient of reward
        g = np.zeros(S)

        # Per sample
        for i,si in enumerate(s):
            g[i] += reward(mu + sig.dot(si))

        # center gradient & normalize
        #print("Mean: {}, std: {}".format(np.mean(g), np.std(g)))
        g = (g-np.mean(g)) / np.std(g)

        # update
        mu += (lr/S) * np.dot(s.T, g)

    print("\tStopping at {}, with reward {}".format(str(mu), reward(mu)))
    print("\tTrue:       {}\n".format(str(true_mu)))
    return mu,sig



" See https://arxiv.org/pdf/1604.00772.pdf "
# TODO: step-size is currently fixed.
# TODO: rank mu update
# TODO: more interesting loss
def cma_es_rank_1():
    #(mu,C_i) = theta
    mu_i = np.zeros(D)
    C_i = np.eye(D) * .1

    epochs = 5000

    # We choose only k of K total samples
    K, k = 64, 32

    step = .1
    ps = np.zeros(D)
    pc = np.zeros(D)
    cc = .01
    c1 = .01
    cs = .01
    #ds = 1.0

    cm = .3

    # Our fitness function weighs samples by _rank_ only, ignoring relative magnitude.
    # The measure mu_eff is 'effective selection mass', which is >= 1 and <= k
    # It is derived by the allocation of weights w_i by the fitness function.
    # Below, it will come to be important for trade-off of robustness/speed.

    # Negative weighting is used, but TODO add code in loop to check if neg
    #weight = np.array([-np.log((rank+1)/(k+1)) for rank in range(k)]) # +2 since rank should start at 1
    #weight = np.array([k - rank + 2 for rank in range(k)]) # +2 since rank should start at 1
    weight = np.array([(k/2.) - rank + 2 for rank in range(k)]) # +2 since rank should start at 1
    weight = weight / np.sum(weight)

    mu_eff = np.sum(weight[i]**2.0 for i in range(k))
    print("mu_eff: {}".format(mu_eff))
    print(weight)


    for epoch in range(epochs):
        # Sample points
        # y will be local coords (mu at origin), x will be global
        y = np.random.multivariate_normal(mean=np.zeros(D), cov=C_i, size=K)
        x = mu_i + y
        x_f = list(map(loss,x))
        x_sorted_inds = np.argsort(x_f)[:k]
        y_sorted = (y[x_sorted_inds])
        x_sorted = (x[x_sorted_inds])

        y_i1 = np.sum(weight[i] * y_sorted[i] for i in range(k))
        mu_i1 = (1-cm)*mu_i + cm * np.sum(weight[i] * (x_sorted[i]) for i in range(k))

        # When computing covariance matrix, we always use newly sampled values as one term,
        # but the other we have a choice:
        #  1. Empirical average of samples
        #  2. Actual (old) mean parameter of sampling distribution  => CMA-ES
        #  3. Actual (new) mean parameter of sampling distribution  => XEntropy-Method

        # Here we go with (2) with additional weighting by fitness,
        # and exponential smoothing of old estimates.

        # Having our covariance estimator be reliable _requires mu_eff to be large_
        # Having fast search (opp. robust) requires k small and mu_eff small.
        # This dillemma can be balanced by reusing info from old generations.

        # evo path for step size
        #Chi = np.linalg.inv(np.linalg.cholesky(C_i)) # ^-1/2
        #ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff)*Chi.dot(y_i1)
        #step = step * np.exp( (cs/ds) * ( (np.linalg.norm(ps)/1.7) - 1.) )

        innov = (mu_i1 - mu_i)/np.std(mu_i1-mu_i)

        # pc will be used as the outer-product, rather than mu_i1 - mu_i. It will smooth steps.
        pc = (1-cc)*pc + np.sqrt(cc * (2-cc) * mu_eff) * y_i1

        C_i1 = (1-c1)*C_i + c1*np.outer(pc, pc)
        #C_i1 = (1-c1)*C_i + c1*np.outer(innov,innov)

        C_i = C_i1
        mu_i = mu_i1
        print(epoch,str(mu_i), str(np.mean(x_f)), str(np.linalg.det(C_i)))

        if loss(mu_i) < .01:
            print("Stopping at epoch",epoch)
            break


    print("\tStopping at {}, with loss {}".format(str(mu_i), loss(mu_i)))
    print("\tTrue:       {}\n".format(str(true_mu)))
    return mu_i,C_i


if __name__=='__main__':
    #naive_es((np.zeros(D), np.eye(D)))
    cma_es_rank_1()
