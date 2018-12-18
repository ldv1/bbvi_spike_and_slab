import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

def sigmoid(z):
    return 1. / ( 1 + np.exp(-z) )
    
def softmax(z):
    z -= np.max(z, axis=1, keepdims = True)
    sm = np.exp(z) / np.sum(np.exp(z),axis=1, keepdims = True)
    return sm

def Bernoulli(pi, T):
    """Bernoulli samples uing the Gumbel-Max trick."""
    assert( pi.shape[1] == 2 )
    N = pi.shape[0]
    z = -np.log(-np.log(np.random.rand(N,2))) + np.log(pi)
    return softmax(z / T)

def black_box_variational_inference(logprob, X, y, num_samples, batch_size):
    
    rs = npr.RandomState(0)
    
    M = X.shape[1]
    
    def unpack_params(params):
        # variational parameters for w
        w_mu, w_log_s2 = params[:M], params[M:2*M]
        # variational parameters for s
        s_pi = sigmoid(params[2*M:3*M])
        # hyperparameters
        log_s2_w, pi_w = params[3*M], sigmoid(params[3*M+1])
        # noise variance
        log_s2 = params[3*M+2]
        return w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2

    def entropy(s_pi, w_log_s2, log_s2_w):
        return  np.sum( \
                       - (1-s_pi)*np.log(1-s_pi) - s_pi*np.log(s_pi) \
                       + 0.5*(1-s_pi)*(log_s2_w + np.log(2*np.pi*np.e)) \
                       + 0.5*s_pi*(w_log_s2 + np.log(2*np.pi*np.e)) \
                       )
    
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        
        w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2 = unpack_params(params)
        # samples
        datafit = 0.
        for _ in range(num_samples):
            s = Bernoulli(pi = np.column_stack( [ 1-s_pi, s_pi ] ), T=0.5)[:,1]
            mean = s*w_mu
            var = s*np.exp(w_log_s2) + (1-s)*np.exp(log_s2_w)
            w = mean + np.sqrt(var) * np.random.randn(M)
            datafit = datafit \
                      + logprob(s, w, log_s2_w, pi_w, log_s2, X, y, batch_size, t)
        datafit = datafit / num_samples
        regularizer = entropy(s_pi, w_log_s2, log_s2_w)
        lower_bound = regularizer + datafit
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params
        
if __name__ == '__main__':
    
    np.random.seed(123)
    
    # variance of observation noise
    sigma = 1.
    # Number of observations.
    N = 100
    # probability that a prameter is larger than noise
    sig_prob = 0.05
    # number of parameters
    M = 200
    # generate parameters following Bettencourt
    # https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#3_experiments
    beta = np.zeros(M)
    bernoullis1 = np.random.binomial(n=1, p=sig_prob, size=M)
    bernoullis2 = np.random.binomial(n=1, p=0.5     , size=M)
    for m in range(M):
        if bernoullis1[m]:
            if bernoullis2[m]:
                beta[m] = 10 + np.random.randn()
            else:
                beta[m] = -10 + np.random.randn()
        else:
            beta[m] = 0.25*np.random.randn()
    print("true weights: {}".format(beta))
    # offset (not accounted for at the moment)
    alpha = 0
    # inputs
    Xtrain = np.random.randn(N,M)
    # outputs
    ytrain = np.matmul(Xtrain,beta) + alpha + sigma*np.random.randn(N)
    # joint probabilities
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
        lb = -objective(params, t)
        print("Iteration {:05d} lower bound {:.3e}".format(t, lb))
        #input("Press Enter to continue...")

    # optimization
    print("Optimizing variational parameters...")
    init_w_mu     = np.random.randn(M)
    init_w_log_s2 = np.log(np.random.rand(M))
    init_s_pi     = np.random.rand(M)
    init_log_s2_w = [ np.log(1.) ]
    init_pi_w    =  [ 0.5 ]
    init_log_s2  =  [ np.log(1e-2) ]
    init_var_params = np.concatenate([init_w_mu, \
                                      init_w_log_s2, \
                                      init_s_pi, \
                                      init_log_s2_w, \
                                      init_pi_w, \
                                      init_log_s2])
    variational_params = adam(gradient, \
                              init_var_params, \
                              step_size=0.05, \
                              num_iters=30000, \
                              callback=callback)
    w_mu, w_log_s2, s_pi, log_s2_w, pi_w, log_s2 = unpack_params(variational_params)
    
    # plot
    fig = plt.figure(figsize=(16,8), facecolor='white')
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(M), beta, \
           linewidth = 3, color = "black", label = "ground truth")
    ax.scatter(np.arange(M), beta, \
           s = 70, marker = '+', color = "black")
    ax.plot(np.arange(M), w_mu*s_pi, \
               linewidth = 3, color = "red", label = "linear model with spike and slab prior")
    ax.set_xlim([0,M-1])
    ax.set_ylabel("Slopes")
    ax.hlines(0,0,M-1)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.legend(prop={'size':14})
    
    fig.set_tight_layout(True)
    fig.savefig('foo.png')
    plt.show()
        
        
