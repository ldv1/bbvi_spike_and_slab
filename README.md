# Black-box spike and slab variational inference, example with linear models

## Motivation

Duvenaud showed in [Black-Box Stochastic Variational Inference in Five Lines of Python](https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf)
how to make use of the Python module [autograd](https://github.com/HIPS/autograd)
to easily code black-box variational inference introduced in [Black Box Variational Inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) by Ranganath et al.

I adapted his code to linear models with spike and slab prior.

## Dependencies
You will need python 3 with [autograd](https://github.com/HIPS/autograd) and [matplotlib](https://matplotlib.org/).

## Model
Let <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/2ec879977d040ea566976854b814f559.svg?invert_in_darkmode" align=middle width=133.14312pt height=27.656969999999987pt/> be our dataset
with <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/9dee8ae1f722af54f7ad95f39e060941.svg?invert_in_darkmode" align=middle width=56.56612500000001pt height=27.656969999999987pt/>.
We consider a linear regression model with spike and slab prior.
<p align="center"><img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/f3b4508b42f26a35b1412698c20b66c6.svg?invert_in_darkmode" align=middle width=428.78385000000003pt height=88.93351499999999pt/></p>
where <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/fa8e0740db409b3997f0c1680b4d3db8.svg?invert_in_darkmode" align=middle width=29.18586pt height=14.155350000000013pt/> designates the <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433210000000003pt height=14.155350000000013pt/>-component of <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/815ff1ddd3e4950336c24643f757e41d.svg?invert_in_darkmode" align=middle width=18.959490000000006pt height=14.61206999999998pt/>.
<img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/e6718aa5499c31af3ff15c3c594a7854.svg?invert_in_darkmode" align=middle width=16.535475000000005pt height=26.76201000000001pt/> is the variance of the i.i.d. Gaussian noise.
<img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/51b5a929b95bcaa91a728fbc3c4eb154.svg?invert_in_darkmode" align=middle width=19.189665pt height=14.155350000000013pt/> and <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/3167b7a401a3798ca0ab37ea2e4b4991.svg?invert_in_darkmode" align=middle width=19.212435000000003pt height=14.155350000000013pt/> are hyperparameters.

We use [black-box variational inference](https://arxiv.org/abs/1401.0118)
to find an approximation to the posterior over all parameters using optimization.
Following Titsias and Lazaro-Gredilla
in [Spike and Slab Variational Inference for Multi-Task and Multiple Kernel Learning](ttps://papers.nips.cc/paper/4305-spike-and-slab-variational-inference-for-multi-task-and-multiple-kernel-learning.pdf),
we propose the following approximation to the posterior:
<p align="center"><img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/3002f7a92cf04b46ef049817d6e01d66.svg?invert_in_darkmode" align=middle width=608.3236499999999pt height=100.14641999999999pt/></p>
where <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/caa16132408d1ef98631f70f158333ab.svg?invert_in_darkmode" align=middle width=21.56979pt height=14.155350000000013pt/>, <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/7af59a8c57340aea043283dd5bf8af08.svg?invert_in_darkmode" align=middle width=21.057960000000005pt height=26.76201000000001pt/> and <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/be447d665f2aa387ed81a35d066e256b.svg?invert_in_darkmode" align=middle width=21.035190000000004pt height=14.155350000000013pt/> for <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/b553f913b0e35bd1e8abd53012962216.svg?invert_in_darkmode" align=middle width=101.57878500000001pt height=22.46574pt/> are variational parameters.

The ELBO is given by
<p align="center"><img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/4baeef412d6c21d1609d7ff7a94e3a3c.svg?invert_in_darkmode" align=middle width=509.8598999999999pt height=18.639225pt/></p>
with <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/59ded6f4bf84eeccda7ad2b9773c39f6.svg?invert_in_darkmode" align=middle width=140.99827499999998pt height=27.656969999999987pt/>.
The entropy of a product of independent variables is the sum of the entropy. Hence we have
<p align="center"><img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/709dcc114dfeb44b6dbd488a3b72fbda.svg?invert_in_darkmode" align=middle width=329.604pt height=47.60745pt/></p>
Because of our factorization, we get:
<p align="center"><img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/ce49d3d0f9c9d6affa4ef7f81f423ecd.svg?invert_in_darkmode" align=middle width=835.56495pt height=149.020245pt/></p>

An unbiased Monte Carlo approximation to the expectation  of
<img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/1e0080fd72a8c76998c0700b1d0b1543.svg?invert_in_darkmode" align=middle width=114.22537499999999pt height=24.65759999999998pt/>
can be computed by first sampling from the Bernoulli variables <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/8752334087e82b4b407bf6f5467c6799.svg?invert_in_darkmode" align=middle width=19.370340000000002pt height=14.155350000000013pt/>
using the [Gumbel-Max trick](https://arxiv.org/abs/1611.01144)
and then from the Gaussian variables <img src="https://github.com/ldv1/bbvi_spike_and_slab/blob/master/svgs/632bb59e30c41191cd91e8dda5c8add4.svg?invert_in_darkmode" align=middle width=23.433465pt height=14.155350000000013pt/>
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
