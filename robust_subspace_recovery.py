import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces
from time import time

from flag import Flag
from utils import plot_nestedness_images, plot_nestedness_scatter, subspace_error


def digits(n_in, n_out, model="0vsOther"):
    digits = load_digits(n_class=8)
    data_X, data_y = digits.data, digits.target
    if model == "0vsOther":
        X_0 = data_X[digits.target == 0].T
        X_other = data_X[digits.target != 0].T
        X_in = X_0[:, anp.random.choice(X_0.shape[1], n_in, replace=False)]
        X_out = X_other[:, anp.random.choice(X_other.shape[1], n_out, replace=False)]
    elif model == "1vsOther":
        X_1 = data_X[digits.target == 1].T
        X_other = data_X[digits.target != 1].T
        X_in = X_1[:, anp.random.choice(X_1.shape[1], n_in, replace=False)]
        X_out = X_other[:, anp.random.choice(X_other.shape[1], n_out, replace=False)]
    elif model == "1vs7":
        X_1 = data_X[digits.target == 1].T
        X_7 = data_X[digits.target == 7].T
        X_in = X_1[:, anp.random.choice(X_1.shape[1], n_in, replace=False)]
        X_out = X_7[:, anp.random.choice(X_7.shape[1], n_out, replace=False)]
    # elif model == "DigitvsFace":
    #     X_in = data_X[anp.random.choice(data_X.shape[0], n_in, replace=False)].T
    #     faces = fetch_olivetti_faces(return_X_y=True)[0]
    #     X_out = []
    #     print(faces.shape[0])
    #     for im in faces[anp.random.choice(faces.shape[0], n_out, replace=False)]:
    #         X_out.append(resize(im, (8, 8)).flatten())
    #     X_out = anp.array(X_out).T
    X, y = anp.concatenate([X_in, X_out], axis=1), anp.array([0] * n_in + [1] * n_out)
    X = X - anp.mean(X_in, axis=1)[:, anp.newaxis]  # anp.median(X, axis=1)[:, anp.newaxis]
    return X, y


def plot_reconstruction_errors(X, n_in, U_Gr, U_Fl, signature):
    p, n = X.shape
    plt.figure()
    reconstruction_errors_gr = anp.linalg.norm(X - U_Gr @ U_Gr.T @ X, axis=0)
    reconstruction_errors_gr = anp.concatenate([anp.sort(reconstruction_errors_gr[:n_in]), anp.sort(reconstruction_errors_gr[n_in:])])
    flag_trick = anp.zeros((p, p))
    for q_k in signature:
        flag_trick += 1 / len(signature) * U_Fl[:, :q_k] @ U_Fl[:, :q_k].T
    reconstruction_errors_fl = anp.linalg.norm(X - flag_trick @ X, axis=0)  # TODO: check if better to apply flag trick or to do ensembling by averaging...
    reconstruction_errors_fl = anp.concatenate([anp.sort(reconstruction_errors_fl[:n_in]), anp.sort(reconstruction_errors_fl[n_in:])])
    plt.plot(anp.arange(1, n + 1), reconstruction_errors_gr, label='Gr reconstruction errors')
    plt.plot(anp.arange(1, n + 1), reconstruction_errors_fl, label='Fl reconstruction errors')
    plt.axvline(x=n_in+.5, color='k', ls='dashed', label='inliner-outlier separation')
    plt.legend()
    plt.interactive(False)
    plt.show(block=True)


if __name__ == "__main__":
    anp.random.seed(42)

    n_in, n_out = 90, 10
    X, y = digits(n_in, n_out, model="0vsOther")
    p, n = X.shape

    signature = (1, 2, 5)
    q = signature[-1]

    start_fl = time()
    flag = Flag(p, signature)

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((flag._p, flag._p))
        for q_k in flag._signature:
            flag_trick += 1 / len(flag._signature) * point[:, :q_k] @ point[:, :q_k].T
        return anp.sum(anp.linalg.norm(X - flag_trick @ X, ord=2, axis=0))

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent()
    result_fl = optimizer_fl.run(problem_fl, initial_point=anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :flag._q])
    time_fl = time() - start_fl

    results_gr = []
    start_gr = time()
    for dim in signature:
        grassmann = Flag(p, (dim,))

        @pymanopt.function.autograd(grassmann)
        def cost_gr(point):
            return anp.sum(anp.linalg.norm(X - point @ point.T @ X, ord=2, axis=0))

        problem_gr = pymanopt.Problem(grassmann, cost_gr)
        optimizer_gr = pymanopt.optimizers.SteepestDescent()
        result_gr = optimizer_gr.run(problem_gr, initial_point=anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :dim])
        results_gr.append(result_gr)
    time_gr = time() - start_gr

    plot_nestedness_scatter(X, results_gr[0].point, results_gr[1].point, result_fl.point)
    plot_reconstruction_errors(X, n_in, results_gr[-1].point, result_fl.point, signature)
    print(f"Gr: nestedness_errors = {[subspace_error(results_gr[k].point, results_gr[k+1].point, type='angle') for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_error(result_fl.point[:, :signature[k]], result_fl.point[:, :signature[k+1]], type='angle') for k in range(len(signature) - 1)]}, time = {time_fl}")