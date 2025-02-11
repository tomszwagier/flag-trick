# flag-trick

Authors' implementation of [_Nested subspace learning with flags_](https://arxiv.org/abs/2502.06022). Cleaning and documentation of the repository in progress.


## Abstract

Many machine learning methods look for low-dimensional representations of the data. The underlying subspace can be estimated by first choosing a dimension $q$ and then optimizing a certain objective function over the space of $q$-dimensional subspaces (the Grassmannian). Trying different $q$ yields in general non-nested subspaces, which raises an important issue of consistency between the data representations. In this paper, we propose a simple trick to enforce nestedness in subspace learning methods. It consists in lifting Grassmannian optimization problems to flag manifolds (the space of nested subspaces of increasing dimension) via nested projectors. We apply the flag trick to several classical machine learning methods and show that it successfully addresses the nestedness issue.
