import autograd.numpy as anp
from autograd.scipy.linalg import expm
from pymanopt.manifolds.manifold import Manifold


class _FlagBase(Manifold):
    @property
    def typical_dist(self):
        raise NotImplementedError()

    def norm(self, point, tangent_vector):
        return anp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        raise NotImplementedError()

    def zero_vector(self, point):
        raise NotImplementedError()

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def to_tangent_space(self, point, vector):
        return self.projection(point, vector)


class Flag(_FlagBase):
    r"""The Grassmann manifold.

    This is the manifold of subspaces of dimension ``p`` of a real vector space
    of dimension ``n``.
    The optional argument ``k`` allows to optimize over the product of ``k``
    Grassmann manifolds.
    Elements are represented as ``n x p`` matrices if ``k == 1``, and as ``k x
    n x p`` arrays if ``k > 1``.

    Args:
        n: Dimension of the ambient space.
        p: Dimension of the subspaces.
        k: The number of elements in the product.

    Note:
        The geometry assumed here is the one obtained by treating the
        Grassmannian as a Riemannian quotient manifold of the Stiefel manifold
        (see also :class:`pymanopt.manifolds.stiefel.Stiefel`)
        with the orthogonal group :math:`\O(p) = \set{\vmQ \in \R^{p \times p}
        : \transp{\vmQ}\vmQ = \vmQ\transp{\vmQ} = \Id_p}`.
    """

    def __init__(self, p: int, signature: tuple):
        self._p = p
        self._q = signature[-1]
        self._signature = signature
        self._signature_full = (0,) + signature + (p,)
        self._type = anp.diff(self._signature_full)

        if not ((0 < signature[0]) and (len(signature) == 1 or anp.all(signature[:-1] < signature[1:])) and (self._q < p)):
            raise ValueError(
                f"Need 0 < signature[0] < ... < signature[-1] < p. Values supplied were p = {p} and signature = {signature}"
            )
        name = f"Flag manifold Fl({p},{signature})"
        dimension = int(p * (p - 1) / 2 - anp.sum(self._type * (self._type - 1) / 2))
        super().__init__(name, dimension)

    def dist(self, point_a, point_b):
        raise NotImplementedError()

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        G = anp.eye(self._p) - 1 / 2 * point @ point.T
        return anp.trace(tangent_vector_a.T @ G @ tangent_vector_b)

    def projection(self, point, vector):
        Z = anp.zeros_like(vector)
        for k in range(1, len(self._signature) + 1):
            U_k = point[:, self._signature_full[k-1]: self._signature_full[k]]
            X_k = vector[:, self._signature_full[k-1]: self._signature_full[k]]
            Z_k = (anp.eye(self._p) - U_k @ U_k.T) @ X_k
            for l in range(1, len(self._signature) + 1):
                if l != k:
                    U_l = point[:, self._signature_full[l-1]: self._signature_full[l]]
                    X_l = vector[:, self._signature_full[l-1]: self._signature_full[l]]
                    Z_k -= U_l @ X_l.T @ U_k
            Z[:, self._signature_full[k-1]: self._signature_full[k]] = anp.copy(Z_k)
        return Z

    def euclidean_to_riemannian_hessian(self, point, euclidean_gradient, euclidean_hessian, tangent_vector):
        raise NotImplementedError()

    def retraction(self, point, tangent_vector):
        u, _, vt = anp.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vt
        # Other possibility: QR retraction
        # return anp.linalg.qr(point + tangent_vector, mode='reduced')[0]
        # return self.exp(point, tangent_vector)

    def random_point(self):
        u, _, vt = anp.linalg.svd(anp.random.normal(size=(self._p, self._q)), full_matrices=False)
        return u @ vt

    def random_tangent_vector(self, point):
        raise NotImplementedError()

    def exp(self, point, tangent_vector):
        point_completion = anp.concatenate([point, anp.linalg.svd(point, full_matrices=True, compute_uv=True)[0][:, self._q:]], axis=1)
        B = anp.zeros_like(point_completion)
        B[:, :self._q] = point_completion.T @ tangent_vector
        B[:, self._q:] = anp.concatenate([- B[self._q:, :self._q].T, anp.zeros((self._p - self._q, self._p - self._q))], axis=0)
        B = (B - B.T) / 2  # for numerical stability
        # return point_completion @ expm(B)[:, :self._q]
        u, _, vt = anp.linalg.svd(point_completion @ expm(B))
        return u @ vt[:, :self._q]  # for numerical stability, we project orthogonally

    def log(self, point_a, point_b):
        raise NotImplementedError()
