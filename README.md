# Black-box spike and slab variational inference, example with linear models

## Motivation

Duvenaud showed in [Black-Box Stochastic Variational Inference in Five Lines of Python](https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf)
how to make use of the Python module [autograd](https://github.com/HIPS/autograd)
to easily code black-box variational inference introduced in [Black Box Variational Inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) by Ranganath et al.

I adapted his code to linear models with spike and slab prior.

## Dependencies
You will need python 3 with [autograd](https://github.com/HIPS/autograd) and [matplotlib](https://matplotlib.org/).

## Model
A succinct description of the model can be found [here](https://github.com/ldv1/bbvi_spike_and_slab/blob/master/paper.pdf).

In short: We use [black-box variational inference](https://arxiv.org/abs/1401.0118)
to find an approximation to the posterior over all parameters using optimization.
The slab and slab prior introduces continuous and discrete random variables.
To sample from the posterior, we use the
[Gumbel-Max trick](https://arxiv.org/abs/1611.01144) for the discrete varibles
and
[reparameterization trick](https://arxiv.org/abs/1312.6114) for the continuous variables.

## Results
For the generation of the dataset, we follow Bettencourt
in [Bayes Sparse Regression](https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html).
The covariates are all independently distributed around zero with unit variance,
and there is a population of both large, relevant slopes and small, irrelevant slopes.
Moreover, the data are collinear, with more covariates than observations,
which implies a non-identified likelihood.

Bettencourt showed that the Finnish horseshoe prior does a pretty good job.
However, it demands a certain level of expertise.

The spike and slab prior finds relevant slopes and set irrelevant slopes to 0,
as depicted on the following figure:

![Demo in 2D](https://github.com/ldv1/bbvi_spike_and_slab/blob/master/demo.png)

## Authors
Laurent de Vito

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
