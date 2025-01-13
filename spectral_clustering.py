import matplotlib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from time import time

from flag import Flag
from utils import subspace_error, plot_subspace_errors


def normalized_graph_Laplacian(X):
    pairwise_dist = euclidean_distances(X.T, X.T, squared=False)
    W = anp.exp(- pairwise_dist**2 / (2 * anp.median(pairwise_dist)**2))
    D_12 = anp.diag(anp.sum(W, axis=1)**(-1/2))
    L = anp.eye(n) - D_12 @ W @ D_12
    return L


def learn_Gr(n, q, L, beta, init="random"):
    init_point = {"random": pymanopt.manifolds.grassmann.Grassmann(n, q).random_point(), "svd": anp.linalg.svd(L, full_matrices=True, compute_uv=True)[0][:, -q:]}  # initialize with smallest eigenvectors of L, but change to largest eigenvectors of W if this does not work well, to do like in GSC
    grassmann = Flag(n, (q,))

    @pymanopt.function.autograd(grassmann)
    def cost_gr(point):
        return anp.trace(point @ point.T @ L) + beta * anp.sum(anp.abs(point @ point.T))

    problem_gr = pymanopt.Problem(grassmann, cost_gr)
    optimizer_gr = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_gr = optimizer_gr.run(problem_gr, initial_point=init_point[init])
    return result_gr.point


def learn_Fl(n, signature, L, beta, init="random"):
    q = signature[-1]
    init_point = {"random": pymanopt.manifolds.grassmann.Grassmann(n, q).random_point(), "svd": anp.linalg.svd(L, full_matrices=True, compute_uv=True)[0][:, -q:]}  # initialize with smallest eigenvectors of L, but change to largest eigenvectors of W if this does not work well, to do like in GSC
    flag = Flag(n, signature)

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((n, n))
        for q_k in signature:
            flag_trick += 1 / len(signature) * point[:, :q_k] @ point[:, :q_k].T
        return anp.trace(flag_trick @ L) + beta * anp.sum(anp.abs(flag_trick))

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_fl = optimizer_fl.run(problem_fl, initial_point=init_point[init])
    return result_fl.point


def plot_nestedness_scatter_SC(U_Gr_1, U_Gr_2, U_Fl, y=None):
    n = U_Gr_1.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    plt.set_cmap("Accent")
    axes[0, 0].scatter(U_Gr_1[:, 0], anp.zeros(n,), c='tab:red' if y is None else y, alpha=.8)
    axes[0, 0].set_title('Subspace - 1D')
    axes[1, 0].scatter(U_Gr_2[:, 0], U_Gr_2[:, 1], c='tab:red' if y is None else y, alpha=.8)
    axes[1, 0].set_title('Subspace - 2D')
    axes[0, 1].scatter(U_Fl[:, 0], anp.zeros(n,), c='tab:green' if y is None else y, alpha=.8)
    axes[0, 1].set_title('Flag - 1D')
    axes[1, 1].scatter(U_Fl[:, 0], U_Fl[:, 1], c='tab:green' if y is None else y, alpha=.8)
    axes[1, 1].set_title('Flag - 2D')
    plt.show(block=False)


def plot_variance_SC(U_Gr_list, U_Fl, signature):
    n = U_Fl.shape[0]
    U_Fl_list = [anp.zeros((n, n))] + [U_Fl[:, :q_k] for q_k in signature] + [anp.eye(n)]
    U_Gr_list = [anp.zeros((n, n))] + U_Gr_list + [anp.eye(n)]
    plt.figure()
    plt.plot((0,)+signature+(n,), [anp.sum(anp.linalg.norm(U - anp.mean(U, axis=0), axis=1)**2) / anp.sum(anp.linalg.norm(anp.eye(n) - anp.mean(anp.eye(n), axis=0), axis=1)**2) for U in U_Gr_list], label='Gr')
    plt.plot((0,)+signature+(n,), [anp.sum(anp.linalg.norm(U - anp.mean(U, axis=0), axis=1)**2) / anp.sum(anp.linalg.norm(anp.eye(n) - anp.mean(anp.eye(n), axis=0), axis=1)**2) for U in U_Fl_list], label='Fl')
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.plot((0,)+signature, anp.diff([anp.sum(anp.linalg.norm(U - anp.mean(U, axis=0), axis=1)**2) / anp.sum(anp.linalg.norm(anp.eye(n) - anp.mean(anp.eye(n), axis=0), axis=1)**2) for U in U_Gr_list]), label='Gr')
    plt.plot((0,)+signature, anp.diff([anp.sum(anp.linalg.norm(U - anp.mean(U, axis=0), axis=1)**2) / anp.sum(anp.linalg.norm(anp.eye(n) - anp.mean(anp.eye(n), axis=0), axis=1)**2) for U in U_Fl_list]), label='Fl')
    plt.legend()
    plt.show(block=False)


if __name__ == "__main__":
    anp.random.seed(42)

    # Nestedness
    dataset = load_breast_cancer()  # load_digits / load_iris / load_wine / fetch_olivetti_faces / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    X = anp.concatenate([X[:, y==c][:, :100//len(anp.unique(y))] for c in anp.unique(y)], axis=1)  # 100 samples equally distributed between classes
    y = anp.concatenate([y[y==c][:100//len(anp.unique(y))] for c in anp.unique(y)])
    (p, n), C = X.shape, len(anp.unique(y))

    L = normalized_graph_Laplacian(X)

    signature = tuple(anp.arange(1, n))
    q = signature[-1]
    beta = 0.00001

    # start_fl = time()
    # U_Fl = learn_Fl(n, signature, L, beta, init="random")
    # time_fl = time() - start_fl
    # U_Gr_list = []
    # start_gr = time()
    # for dim in signature:
    #     U_Gr = learn_Gr(n, dim, L, beta, init="random")
    #     U_Gr_list.append(U_Gr)
    # time_gr = time() - start_gr
    #
    # print(f"Gr: nestedness_errors = {[subspace_error(U_Gr_list[k], U_Gr_list[k+1], type='angle') for k in range(len(signature) - 1)]}, time = {time_gr}")
    # print(f"Fl: nestedness_errors = {[subspace_error(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]], type='angle') for k in range(len(signature) - 1)]}, time = {time_fl}")
    # plot_nestedness_scatter_SC(U_Gr_list[0], U_Gr_list[1], U_Fl, y=y)
    # plot_subspace_errors(U_Gr_list, U_Fl, signature)
    # plot_variance_SC(U_Gr_list, U_Fl, signature)


    # Classification
    # dataset = load_breast_cancer()  # load_digits / load_iris / load_wine / fetch_olivetti_faces / load_breast_cancer
    # X, y = dataset.data.T, dataset.target
    # X = anp.concatenate([X[:, y==c][:, :100//len(anp.unique(y))] for c in anp.unique(y)], axis=1)  # 100 samples equally distributed between classes
    # y = anp.concatenate([y[y==c][:100//len(anp.unique(y))] for c in anp.unique(y)])
    # (p, n), C = X.shape, len(anp.unique(y))
    #
    # L = normalized_graph_Laplacian(X)
    #
    # signature = (1, 2, 5, 10)
    # q = signature[-1]
    # beta = 0.01
    #
    # U_Gr = learn_Gr(n, q, L, beta, init="svd")
    # U_Fl = learn_Fl(n, signature, L, beta, init="svd")
    #
    # clus_Gr = KMeans(n_clusters=C, random_state=42)
    # y_Gr_pred = clus_Gr.fit_predict(U_Gr / anp.linalg.norm(U_Gr, axis=1)[:, anp.newaxis])
    # print(f"RI Gr({n, q})", rand_score(y, y_Gr_pred))  # check if I should use better evaluation metrics! https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    #
    # y_Fl_pred_w = anp.zeros((n, C))
    # # y_Fl_pred_u = anp.zeros((n, C))
    # weights = anp.zeros((len(signature)))
    # for k, q_k in enumerate(signature):
    #     U_Fl_k = U_Fl[:, :q_k]
    #     clus_Fl = KMeans(n_clusters=C, random_state=42)
    #     y_Fl_pred = clus_Fl.fit_predict(U_Fl_k / anp.linalg.norm(U_Fl_k, axis=1)[:, anp.newaxis])
    #     w = rand_score(y, y_Fl_pred)
    #     weights[k] = w
    #     print(f"RI Fl({n, signature}) (q={q_k})", w)
    #     lb = LabelBinarizer()
    #     lb.fit(y_Fl_pred)
    #     y_bin = lb.transform(y_Fl_pred)
    #     if C == 2:
    #         y_bin = anp.concatenate([1 - y_bin, y_bin], axis=1)  # 0 -> [1, 0]
    #     y_Fl_pred_w = y_Fl_pred_w + w * y_bin
    #     # y_Fl_pred_u = y_Fl_pred_u + 1 / len(signature) * y_bin
    # y_Fl_pred_w_max = anp.argmax(y_Fl_pred_w, axis=1)
    # print(f"RI Fl-W", rand_score(y, y_Fl_pred_w_max))
    # # y_Fl_pred_u_max = anp.argmax(y_Fl_pred_u, axis=1)
    # # print(f"RI Fl-U", rand_score(y, y_Fl_pred_u_max))
    # weights = weights / anp.sum(weights)
