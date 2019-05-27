import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

def sigmoid(z):
    return 1. / ( 1 + np.exp(-z) )

def inv_sigmoid(z):
    return np.log(z/(1.-z))

def softmax(z):
    # apply softmax to each row of z 
    # remove max to avoid under / overflow
    z -= np.max(z, axis=1, keepdims = True)
    sm = np.exp(z) / np.sum(np.exp(z),axis=1, keepdims = True)
    return sm

def Bernoulli(pi, T):
    # Bernoulli samples uing the Gumbel-Max trick.
    # each row of pi defines the class probabilities pi_1, ..., pi_K
    # T is the temperature
    assert( pi.shape[1] == 2 )
    N = pi.shape[0]
    z = -np.log(-np.log(np.random.rand(N,2))) + np.log(pi)
    return softmax(z / T)

def black_box_variational_inference(logprob, X, y, num_samples, batch_size):
    
    rs = npr.RandomState(0)
    
    # the number of weights
    M = X.shape[1]
    
    def unpack_params(params):
        # variational parameters for w: mean and variance
        w_mu, w_log_s2 = params[:M], params[M:2*M]
        # variational parameters for s: Bernoulli variable
        s_pi = sigmoid(params[2*M:3*M])
        # hyperparameters
        log_s2_w, pi_w = params[3*M], sigmoid(params[3*M+1])
        # noise variance
        log_s2 = params[3*M+2]
        return w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2

    def entropy(s_pi, w_log_s2, log_s2_w):
        # entropy of the approx. posterior
        return  np.sum( \
                       - (1-s_pi)*np.log(1-s_pi) - s_pi*np.log(s_pi) \
                       + 0.5*(1-s_pi)*(log_s2_w + np.log(2*np.pi*np.e)) \
                       + 0.5*s_pi*(w_log_s2 + np.log(2*np.pi*np.e)) \
                       )
    
    def variational_objective(params, t):
        # stochastic estimate of the variational lower bound
        
        w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2 = unpack_params(params)
        
        # compute the expectation (the "data fit" term) by Monte Carlo sampling
        datafit = 0.
        for _ in range(num_samples):
            # acquire Bernoulli samples
            s = Bernoulli(pi = np.column_stack( [ 1-s_pi, s_pi ] ), T=0.5)[:,1]
            # acquire Normal distributed samples
            mean = s*w_mu
            var = s*np.exp(w_log_s2) + (1-s)*np.exp(log_s2_w)
            w = mean + np.sqrt(var) * np.random.randn(M)
            # compute the log of the joint probability
            datafit = datafit \
                      + logprob(s, w, log_s2_w, pi_w, log_s2, X, y, batch_size, t)
        datafit = datafit / num_samples
        # compute entropy of the approx. posterior of the weights
        regularizer = entropy(s_pi, w_log_s2, log_s2_w)
        # the lower bound to maximize
        lower_bound = regularizer + datafit
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params
        
if __name__ == '__main__':
    
    np.random.seed(123)
    
    # std of observation noise
    sigma = 1.
    # Number of observations.
    N = 100
    # probability that a parameter is larger than noise
    sig_prob = 0.05
    # number of weights
    M = 200
    # generate parameters following Bettencourt
    # betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#3_experiments
    beta = np.zeros(M+1)
    bernoullis1 = np.random.binomial(n=1, p=sig_prob, size=M)
    bernoullis2 = np.random.binomial(n=1, p=0.5     , size=M)
    for m in range(M):
        if bernoullis1[m]:
            # large parameter
            if bernoullis2[m]:
                beta[m] = 10 + np.random.randn()
            else:
                beta[m] = -10 + np.random.randn()
        else:
            beta[m] = 0.25*np.random.randn()
    print("true weights:\n{}".format(beta[:-1]))
    # offset
    beta[M] = 0.
    # inputs
    Xtrain = np.random.randn(N,M+1)
    Xtrain[:,M] = 1
    # outputs
    ytrain = np.matmul(Xtrain,beta) + sigma*np.random.randn(N)
    # joint probability for a batch
    def logprob(s, w, log_s2_w, pi_w, log_s2, X, y, batch_size, t):
        
        N = X.shape[0]
        M = w.shape[0]
        # we consider only a batch of size b
        batch_size = min(batch_size, N)
        b = float(batch_size)
        indices = np.random.choice(N, batch_size, replace = False)
        Xbatch = X[indices]
        ybatch = y[indices]
        
        def logprior():
           return -M/2.*( np.log(2*np.pi) + log_s2_w) \
                  - 1./(2.*np.exp(log_s2_w))*np.sum(np.square(w)) \
                  + np.sum( s*np.log(pi_w) + (1-s)*np.log(1-pi_w) )
            
        def loglik():
            # the noise model is Gaussian
            y_mean = np.dot(Xbatch,s*w)
            return -b/2.*( np.log(2*np.pi)+log_s2 ) \
                   - 1./(2.*np.exp(log_s2))*np.sum( np.square(ybatch-y_mean) )
        
        return N/b*loglik() + logprior()

    # build variational objective
    objective, gradient, unpack_params = \
        black_box_variational_inference(logprob,
                                        Xtrain, ytrain, \
                                        num_samples=1, batch_size=1000)
    
    # callback during optimization
    def callback(params, t, g):
        if t % 1000 == 0:
            lb = -objective(params, t)
            w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2 = unpack_params(params)
            print("Iteration {:05d} lower bound {:.3e}, noise std {:.3e}" \
                  .format(t, lb, np.exp(0.5*log_s2)))
            #input("Press Enter to continue...")

    # optimization
    print("Optimizing variational parameters...")
    
    # initializing the parameters
    init_w_mu     = np.random.randn(M+1)
    init_w_log_s2 = np.log(np.random.rand(M+1))
    init_s_pi     = inv_sigmoid( np.random.uniform(low=0.4, high=0.6, size=M+1) )
    init_log_s2_w = [ np.log(1.) ]
    init_pi_w    =  [ inv_sigmoid(0.5) ]
    init_log_s2  =  [ np.log(1e-2) ]
    init_var_params = np.concatenate([init_w_mu, \
                                      init_w_log_s2, \
                                      init_s_pi, \
                                      init_log_s2_w, \
                                      init_pi_w, \
                                      init_log_s2])
    
    # optimizing
    variational_params = adam(gradient, \
                              init_var_params, \
                              step_size=0.005, \
                              num_iters=50000, \
                              callback=callback)
    w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2 = unpack_params(variational_params)
    
    # print some results
    print("mean of offset: {}".format(w_mu[M]*s_pi[M]))
    print("optimized hyperparameters:")
    print("sparsity: {}".format(pi_w))
    print("slab variance: {}".format(np.exp(log_s2_w)))
    print("noise std: {}".format(np.exp(0.5*log_s2)))
    
    # plot
    fig = plt.figure(figsize=(16,8), facecolor='white')
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(M), beta[:-1], \
           linewidth = 3, color = "black", label = "ground truth")
    ax.scatter(np.arange(M), beta[:-1], \
           s = 70, marker = '+', color = "black")
    ax.plot(np.arange(M), w_mu[:-1]*s_pi[:-1], \
               linewidth = 3, color = "red", \
               label = "linear model with spike and slab prior")
    ax.set_xlim([0,M-1])
    ax.set_ylabel("Slopes", fontsize=18)
    ax.hlines(0,0,M-1)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.legend(prop={'size':14})
    
    fig.set_tight_layout(True)
    fig.savefig('foo.png')
    plt.show()
