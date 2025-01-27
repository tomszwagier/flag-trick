import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from time import time

from data import digits_rsr, synthetic_rsr
from flag import Flag
from utils import plot_nestedness_scatter, subspace_distance, plot_reconstruction_errors, plot_scatter_3D


def learn(p, signature, X, init="random"):
    flag = Flag(p, signature)
    init_point = {"random": flag.random_point(), "svd": anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :signature[-1]]}

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((p, p))
        for q_k in signature:
            flag_trick += 1 / len(signature) * point[:, :q_k] @ point[:, :q_k].T
        return anp.sum(anp.linalg.norm(X - flag_trick @ X, ord=2, axis=0))

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_fl = optimizer_fl.run(problem_fl, initial_point=init_point[init])
    return result_fl.point


if __name__ == "__main__":
    anp.random.seed(42)

    # n_in, n_out = 450, 50
    # X, y = synthetic_rsr(n_in, n_out)
    # signature = (1, 2)

    n_in, n_out = 90, 10
    X, y, center = digits_rsr(n_in, n_out)
    signature = (1, 2, 5)

    p, n = X.shape
    q = signature[-1]

    start_gr = time()
    U_Gr_list = []
    for dim in signature:
        U_Gr = learn(p, (dim,), X, init="random")
        U_Gr_list.append(U_Gr)
    time_gr = time() - start_gr
    start_fl = time()
    U_Fl = learn(p, signature, X, init="random")
    time_fl = time() - start_fl

    print(f"Gr: nestedness_errors = {[subspace_distance(U_Gr_list[k], U_Gr_list[k+1]) for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_distance(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]]) for k in range(len(signature) - 1)]}, time = {time_fl}")
    cmap = plt.get_cmap('tab20c')
    colors = cmap(anp.array([0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]))  # issue if more than 20 classes
    plot_nestedness_scatter(X, U_Gr_list[0], U_Gr_list[1], U_Fl, y=colors[y])
    plot_reconstruction_errors(X, n_in, U_Gr_list[-1], U_Fl, signature)
    # plot_scatter_3D(X, y, U_Gr_list=None, U_Fl=None)
