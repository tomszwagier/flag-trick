import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces, load_breast_cancer, load_wine
from time import time

from flag import Flag
from utils import plot_explained_variance, plot_nestedness_images, plot_nestedness_scatter, subspace_error


def LDA_scatter_matrices(X, y):
    print(f"Computing LDA scatter matrices...")
    X_list = [X[:, y==c] for c in list(anp.unique(y))]
    mu_list = [anp.mean(X_c, axis=1)[:, anp.newaxis] for X_c in X_list]
    n_list = [X_c.shape[1] for X_c in X_list]
    mu = anp.mean(X, axis=1)[:, anp.newaxis]
    Sw = anp.sum([(X_c - mu_c) @ (X_c - mu_c).T for (X_c, mu_c) in zip(X_list, mu_list)], axis=0)
    Sb = anp.sum([n_c * (mu_c - mu) @ (mu_c - mu).T for (n_c, mu_c) in zip(n_list, mu_list)], axis=0)
    return Sw, Sb


if __name__ == "__main__":
    anp.random.seed(42)

    dataset = load_wine()  # load_digits / load_iris / load_wine / fetch_olivetti_faces / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    center = anp.mean(X, axis=1)[:, anp.newaxis]
    X = X - center
    p, n = X.shape
    C = len(anp.unique(y))
    U_pca = anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :min(n-C, p)]
    X_pca = U_pca.T @ X
    Sw, Sb = LDA_scatter_matrices(X_pca, y)
    Sw = Sw + 10 ** (-5) * anp.trace(Sw) * anp.eye(X_pca.shape[0])  # regularization as in [Ngo2012]
    Sb = Sb + 10 ** (-5) * anp.trace(Sb) * anp.eye(X_pca.shape[0])  # regularization as in [Ngo2012]
    Sw /= anp.trace(Sw)
    Sb /= anp.trace(Sb)

    signature = (1, 2, 5)
    q = signature[-1]

    start_fl = time()
    flag = Flag(p, signature)

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((flag._p, flag._p))
        for q_k in flag._signature:
            flag_trick += 1 / len(flag._signature) * point[:, :q_k] @ point[:, :q_k].T
        return - anp.trace(flag_trick @ Sb) / anp.trace(flag_trick @ Sw)

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent()
    result_fl = optimizer_fl.run(problem_fl, initial_point=pymanopt.manifolds.grassmann.Grassmann(p, q).random_point()) # pymanopt.manifolds.grassmann.Grassmann(p, q).random_point() / anp.linalg.svd(Sb, full_matrices=True, compute_uv=True)[0][:, :q]

    time_fl = time() - start_fl

    results_gr = []
    start_gr = time()
    for dim in signature:
        grassmann = Flag(p, (dim,))

        @pymanopt.function.autograd(grassmann)
        def cost_gr(point):
            return - anp.trace(point.T @ Sb @ point) / anp.trace(point.T @ Sw @ point)

        problem_gr = pymanopt.Problem(grassmann, cost_gr)
        optimizer_gr = pymanopt.optimizers.SteepestDescent()
        result_gr = optimizer_gr.run(problem_gr, initial_point=pymanopt.manifolds.grassmann.Grassmann(p, dim).random_point())  # pymanopt.manifolds.grassmann.Grassmann(p, dim).random_point() / anp.linalg.svd(Sb, full_matrices=True, compute_uv=True)[0][:, :dim]
        results_gr.append(result_gr)
    time_gr = time() - start_gr

    print(f"Gr: nestedness_errors = {[subspace_error(results_gr[k].point, results_gr[k+1].point, type='angle') for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_error(result_fl.point[:, :signature[k]], result_fl.point[:, :signature[k+1]], type='angle') for k in range(len(signature) - 1)]}, time = {time_fl}")
    plot_nestedness_scatter(X, results_gr[0].point, results_gr[1].point, result_fl.point, y=y)
    # plot_nestedness_images(X[:, 0].reshape(8, 8), center.reshape(8, 8), [res.point for res in results_gr], result_fl.point, signature)
    plot_explained_variance(X, [res.point for res in results_gr], result_fl.point, signature)


    # Classification
    X_gr = results_gr[-1].point.T @ X_pca
    clf_gr = KNeighborsClassifier(n_neighbors=5)
    clf_gr.fit(X_gr.T, y)
    y_gr_pred = clf_gr.predict(X_gr.T)
    print(accuracy_score(y, y_gr_pred))

    y_fl_preds = []
    for q_k in signature:
        X_fl_k = result_fl.point[:, :q_k].T @ X_pca
        clf_fl = KNeighborsClassifier(n_neighbors=5)
        clf_fl.fit(X_fl_k.T, y)
        y_fl_preds.append(clf_fl.predict_proba(X_fl_k.T))
    y_fl_pred = anp.mean(y_fl_preds, axis=0)
    print(accuracy_score(y, anp.argmax(y_fl_pred, axis=1)))
