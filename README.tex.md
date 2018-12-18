# Black-box spike and slab variational inference, example with linear models

## Motivation

Duvenaud showed in [Black-Box Stochastic Variational Inference in Five Lines of Python](https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf)
how to make use of the Python module [autograd](https://github.com/HIPS/autograd)
to easily code black-box variational inference introduced in [Black Box Variational Inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) by Ranganath et al.

I adapted his code to linear models with spike and slab prior.

## Dependencies
You will need python 3 with [autograd](https://github.com/HIPS/autograd) and [matplotlib](https://matplotlib.org/).

## Model
Let $\mathcal{D} = \{ (\boldsymbol{x}_n, y_n ) \}_{n=1}^N$ be our dataset
with $\boldsymbol{x} \in \mathbb{R}^M$.
We consider a linear regression model with spike and slab prior.
$$
\begin{eqnarray}
w_m & \sim & \mathcal{N}( 0, \sigma_w^2)  \text{, } & m = 1, \cdots, M \\
s_m &\sim & \text{Bernoulli}(\pi_w) \text{, } & m = 1, \cdots, M  \\
y_n & \sim & \mathcal{N}( \sum_{m=1}^M{ w_m s_m x_{nm} }, \sigma^2) \text{, } & n = 1, \cdots, N
\end{eqnarray}
$$
where $x_{nm}$ designates the $m$-component of $\boldsymbol{x}_n$.
$\sigma^2$ is the variance of the i.i.d. Gaussian noise.
$\pi_w$ and $\sigma_w$ are hyperparameters.

We use [black-box variational inference](https://arxiv.org/abs/1401.0118)
to find an approximation to the posterior over all parameters using optimization.
Following Titsias and Lazaro-Gredilla
in [Spike and Slab Variational Inference for Multi-Task and Multiple Kernel Learning](ttps://papers.nips.cc/paper/4305-spike-and-slab-variational-inference-for-multi-task-and-multiple-kernel-learning.pdf),
we propose the following approximation to the posterior:
$$
\begin{eqnarray}
q(\boldsymbol{w}, \boldsymbol{s})
&=&
\prod_{m=1}^M{ q(w_m|s_m) \ q(s_m) } \\
&=&
\prod_{m=1}^M{
  \mathcal{N}( s_m \mu_m, s_m \sigma_m^2 + (1-s_m) \sigma^2_w ) \
  \pi_m^{s_m} (1-\pi_m)^{(1-s_m)}
  }
\end{eqnarray}
$$
where $\mu_m$, $\sigma_m^2$ and $\pi_m$ for $m=1,\cdots,M$ are variational parameters.

The ELBO is given by
$$
\begin{eqnarray}
\mathcal{L}( \phi; \boldsymbol{y},\boldsymbol{X} )
&=&
H( q_\phi(\boldsymbol{w}, \boldsymbol{s} | \boldsymbol{y},\boldsymbol{X}) )
+
\text{E}_{\boldsymbol{w}, \boldsymbol{s} \sim q_\phi(\boldsymbol{w}, \boldsymbol{s} | \boldsymbol{y},\boldsymbol{X})}
[
\log p( \boldsymbol{y}, \boldsymbol{w}, \boldsymbol{s} | \boldsymbol{X} )
]
\end{eqnarray}
$$
with $ \phi = \{ (w_m,s_m) \}_{m=1}^M $.
The entropy of a product of independent variables is the sum of the entropy. Hence we have
$$
H( q_\phi(\boldsymbol{w}, \boldsymbol{s} | \boldsymbol{y},\boldsymbol{X}) )
=
\sum_{m=1}^M { 
  H( q_\phi(\boldsymbol{w}_m, \boldsymbol{s}_m | \boldsymbol{y},\boldsymbol{X}) )
}
$$
Because of our factorization, we get:
$$
\begin{eqnarray}
-H( q_\phi(\boldsymbol{w}_m, \boldsymbol{s}_m | \boldsymbol{y},\boldsymbol{X}) )
&=&
\int{ (1-\pi_m) \mathcal{N}( w_m | 0, \sigma_w^2 ) 
      \log \left[ (1-\pi_m) \mathcal{N}( w_m | 0, \sigma_w^2 ) \right]  \ dw_m } \\
&&+
\int{ \pi_m \mathcal{N}( w_m | \mu_m, \sigma_m^2 )
       \log \left[ \pi_m \mathcal{N}( w_m | \mu_m, \sigma_m^2 ) \right]  \ dw_m } \\
&=&
(1-\pi_m) [ \log (1-\pi_m) - H( \mathcal{N}( w_m | 0, \sigma_w^2 ) ) ] \\
&&+
\pi_m [ \log \pi_m - H( \mathcal{N}( w_m | \mu_m, \sigma_m^2 ) ) ] \\
&=&
(1-\pi_m) [ \log (1-\pi_m) - 0.5 \log( 2 \pi e  \sigma_w^2 ) ) ] \\
&&+
\pi_m [ \log \pi_m - 0.5 \log( 2 \pi e  \sigma_m^2 ) ) ] \\
&=&
(1-\pi_m)  \log (1-\pi_m) + \pi_m  \log \pi_m \\
&&-
\frac{1}{2} (1-\pi_m)  \log(  2 \pi e \sigma_w^2 ) - \frac{1}{2} \pi_m  \log( 2 \pi e  \sigma_m^2 )
\end{eqnarray}
$$

An unbiased Monte Carlo approximation to the expectation  of
$ \log p( \boldsymbol{y}, \boldsymbol{w}, \boldsymbol{s} | \boldsymbol{X} ) $
can be computed by first sampling from the Bernoulli variables $s_m$
using the [Gumbel-Max trick](https://arxiv.org/abs/1611.01144)
and then from the Gaussian variables $w_m$
using the [reparameterization trick](https://arxiv.org/abs/1312.6114).

## Results
For the generation of the dataset, we follow Bettencourt
in [Bayes Sparse Regression](https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html).
The covariates are all independently distributed around zero with unit variance,
and there is a population of both large, relevant slopes and small, irrelevant slopes.
Moreover, the data are collinear, with more covariates than observations,
which implies a non-identified likelihood.

Bettencourt showed that the Finnish horseshoe prior does a pretty good job.
However, it demands a certain level of expertise.

The spike and slab prior finds relevant slopes and set irrelevant slopes to 0.  
![Demo in 2D](https://github.com/ldv1/bbvi_spike_and_slab/blob/master/demo.gif)

## Authors
Laurent de Vito

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
