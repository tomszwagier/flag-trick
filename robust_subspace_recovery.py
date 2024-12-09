import autograd.numpy as anp
import pymanopt

from flag import Flag

anp.random.seed(42)

p = 10
# q = 2
# manifold = pymanopt.manifolds.Grassmann(p, q)
signature = (1, 2, 5,)
manifold = Flag(p, signature)


def haystack(n_in, n_out, sigma2_in, sigma2_out, sigma2):
    """ Cf. Lerman, G., McCoy, M.B., Tropp, J.A. et al.
    Robust Computation of Linear Models by Convex Relaxation.
    Found Comput Math 15, 363–410 (2015). https://doi.org/10.1007/s10208-014-9221-0
    """
    # p, q = manifold._n, manifold._p
    p, q = manifold._p, manifold._q
    U = pymanopt.manifolds.Grassmann(p, q).random_point()
    X_in = anp.random.multivariate_normal(anp.zeros((p,)), sigma2_in / q * U @ U.T, size=n_in).T
    X_out = anp.random.multivariate_normal(anp.zeros((p,)), sigma2_out / p * anp.eye(p), size=n_out).T
    X = anp.concatenate([X_in, X_out], axis=1)
    X += anp.random.multivariate_normal(anp.zeros((p,)), sigma2 * anp.eye(p), size=n_in+n_out).T
    return X, U


X, U = haystack(150, 50, sigma2_in=1, sigma2_out=1, sigma2=1e-4)


@pymanopt.function.autograd(manifold)
def cost(point):
    # return anp.sum(anp.linalg.norm(X - point @ point.T @ X, ord=2, axis=0))
    flag_trick = anp.zeros((manifold._p, manifold._p))
    for q_k in manifold._signature:
        flag_trick += 1 / len(manifold._signature) * point[:, :q_k] @ point[:, :q_k].T
    return anp.sum(anp.linalg.norm(X - flag_trick @ X, ord=2, axis=0))


problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem, initial_point=anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :manifold._q])


def subspace_error(U_true, U_found, type="angle"):
    if type == "angle":  # Square root of sum of squared principal angles. Cf. review.
        error = anp.linalg.norm(anp.arccos(anp.clip(anp.linalg.svd(U_true.T @ U_found)[1], -1, 1)))
    elif type == "projection":  # Cf. GMS paper.
        error = anp.linalg.norm(U_true @ U_true.T - U_found @ U_found.T)
    else:
        raise NotImplementedError
    return error


print("Subspace Error", subspace_error(U, result.point, type="angle"))
