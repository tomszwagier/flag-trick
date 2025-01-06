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
from time import time

from flag import Flag
from utils import plot_explained_variance, plot_nestedness_images, plot_nestedness_scatter, subspace_error, plot_subspace_errors


def LDA_scatter_matrices(X, y):
    X_list = [X[:, y==c] for c in list(anp.unique(y))]
    mu_list = [anp.mean(X_c, axis=1)[:, anp.newaxis] for X_c in X_list]
    n_list = [X_c.shape[1] for X_c in X_list]
    mu = anp.mean(X, axis=1)[:, anp.newaxis]
    Sw = anp.sum([(X_c - mu_c) @ (X_c - mu_c).T for (X_c, mu_c) in zip(X_list, mu_list)], axis=0)
    Sb = anp.sum([n_c * (mu_c - mu) @ (mu_c - mu).T for (n_c, mu_c) in zip(n_list, mu_list)], axis=0)
    return Sw, Sb


def generate_lda_data(X, y):
    center = anp.mean(X, axis=1)[:, anp.newaxis]
    X = X - center
    p, n = X.shape
    C = len(anp.unique(y))
    U_pca = anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :min(n-C, p)]
    if n-C < p:
        print("CAUTION: PCA done")
    X_pca = U_pca.T @ X
    Sw, Sb = LDA_scatter_matrices(X_pca, y)
    Sw = Sw + 10 ** (-5) * anp.trace(Sw) * anp.eye(X_pca.shape[0])  # regularization as in [Ngo2012]
    Sb = Sb + 10 ** (-5) * anp.trace(Sb) * anp.eye(X_pca.shape[0])  # regularization as in [Ngo2012]
    Sw /= anp.trace(Sw)
    Sb /= anp.trace(Sb)
    return U_pca, X_pca, Sw, Sb, center


def learn_Gr(p, q, Sb, Sw, init="random"):
    init_point = {"random": pymanopt.manifolds.grassmann.Grassmann(p, q).random_point(), "svd": anp.linalg.svd(Sb, full_matrices=True, compute_uv=True)[0][:, :q]}
    grassmann = Flag(p, (q,))

    @pymanopt.function.autograd(grassmann)
    def cost_gr(point):
        return - anp.trace(point.T @ Sb @ point) / anp.trace(point.T @ Sw @ point)

    problem_gr = pymanopt.Problem(grassmann, cost_gr)
    optimizer_gr = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_gr = optimizer_gr.run(problem_gr, initial_point=init_point[init])
    return result_gr.point


def learn_Fl(p, signature, Sb, Sw, init="random"):
    q = signature[-1]
    init_point = {"random": pymanopt.manifolds.grassmann.Grassmann(p, q).random_point(), "svd": anp.linalg.svd(Sb, full_matrices=True, compute_uv=True)[0][:, :q]}
    flag = Flag(p, signature)

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((p, p))
        for q_k in signature:
            flag_trick += 1 / len(signature) * point[:, :q_k] @ point[:, :q_k].T
        return - anp.trace(flag_trick @ Sb) / anp.trace(flag_trick @ Sw)

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_fl = optimizer_fl.run(problem_fl, initial_point=init_point[init])
    return result_fl.point


if __name__ == "__main__":
    anp.random.seed(42)

    # Nestedness
    # dataset = load_digits()  # load_digits / load_iris / load_wine / fetch_olivetti_faces / load_breast_cancer
    # X, y = dataset.data.T, dataset.target
    # U_pca, X_pca, Sw, Sb, center = generate_lda_data(X, y)
    # (p, n), C = X.shape, len(anp.unique(y))
    #
    # signature = tuple(anp.arange(1, p))
    # q = signature[-1]
    #
    # start_fl = time()
    # U_Fl = learn_Fl(p, signature, Sb, Sw, init="random")
    # time_fl = time() - start_fl
    # U_Gr_list = []
    # start_gr = time()
    # for dim in signature:
    #     U_Gr = learn_Gr(p, dim, Sb, Sw, init="random")
    #     U_Gr_list.append(U_Gr)
    # time_gr = time() - start_gr
    #
    # print(f"Gr: nestedness_errors = {[subspace_error(U_Gr_list[k], U_Gr_list[k+1], type='angle') for k in range(len(signature) - 1)]}, time = {time_gr}")
    # print(f"Fl: nestedness_errors = {[subspace_error(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]], type='angle') for k in range(len(signature) - 1)]}, time = {time_fl}")
    # plot_nestedness_scatter(X, U_Gr_list[0], U_Gr_list[1], U_Fl, y=y)
    # plot_nestedness_images(X[:, 0].reshape(8, 8) - center.reshape(8, 8), anp.zeros((8, 8)), U_Gr_list, U_Fl, signature)
    # plot_explained_variance(X, U_Gr_list, U_Fl, signature)
    # plot_subspace_errors(U_Gr_list, U_Fl, signature)

    # Classification
    dataset = load_breast_cancer()  # load_digits / load_iris / load_wine / fetch_olivetti_faces / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    U_pca, X_pca, Sw, Sb, center = generate_lda_data(X, y)
    (p, n), C = X.shape, len(anp.unique(y))
    signature = (1, 2, 5)  #  tuple(anp.arange(1, p))
    q = signature[-1]
    U_Gr = learn_Gr(p, q, Sb, Sw, init="svd")
    U_Fl = learn_Fl(p, signature, Sb, Sw, init="svd")

    X_Gr = U_Gr.T @ X_pca
    clf_Gr = KNeighborsClassifier(n_neighbors=5)
    clf_Gr.fit(X_Gr.T, y)
    y_Gr_pred = clf_Gr.predict_proba(X_Gr.T)
    print(f"Classification accuracy Gr({p, q})", log_loss(y, y_Gr_pred))

    y_Fl_preds = anp.zeros((n * C, len(signature)))
    for k, q_k in enumerate(signature):
        X_Fl_k = U_Fl[:, :q_k].T @ X_pca
        clf_Fl = KNeighborsClassifier(n_neighbors=5)
        clf_Fl.fit(X_Fl_k.T, y)
        y_Fl_pred = clf_Fl.predict_proba(X_Fl_k.T)
        print(log_loss(y, y_Fl_pred))
        y_Fl_preds[:, k] = y_Fl_pred.flatten()

    w_uniform = 1 / len(signature) * anp.ones(len(signature))
    y_Fl_pred_uniform = (y_Fl_preds @ w_uniform).reshape(n, C)  # proba @ w rather?
    print(f"Cross-Entropy Fl({p, signature})", log_loss(y, y_Fl_pred_uniform))

    lb = LabelBinarizer()
    lb.fit(y)
    y_bin = lb.transform(y)
    if C == 2:
        y_bin = anp.concatenate([1 - y_bin, y_bin], axis=1)  # 0 -> [1, 0]
    eps = anp.finfo(y_Fl_preds.dtype).eps  # / 1e-4
    proba = anp.clip(y_Fl_preds, eps, 1 - eps)  # clipping here ensures that proba @ w is clipped too...
    import cvxpy as cp
    w = cp.Variable(len(signature))  # TODO: CAUTION I may have done total nonsense on the definition of the objective... The second term should not exist... But well, it's probably the same quantity anyway
    objective = cp.Minimize(1 / n * (cp.sum(- (cp.multiply(y_bin.flatten(), cp.log(proba @ w)) + cp.multiply((1 - y_bin.flatten()), cp.log(1 - proba @ w))))))  # as in sklearn, we do sum on axis 1 and average on axis 0 / cp.Multiply?
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    print(w.value)
    # log_loss may cause undesirable behaviour, if so replace with squared loss
    y_Fl_pred = (y_Fl_preds @ w.value).reshape(n, C)  # proba @ w rather?
    print(f"Cross-Entropy Fl({p, signature})", log_loss(y, y_Fl_pred))
