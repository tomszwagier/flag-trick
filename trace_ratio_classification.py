import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from trace_ratio import learn, generate_lda_data

if __name__ == "__main__":
    anp.random.seed(42)

    plt.figure()

    for (dataset, signature, color, label) in zip([load_digits(), load_iris(), load_wine(), load_breast_cancer()], [(1, 2, 5, 10), (1, 2, 3), (1, 2, 5), (1, 2, 5)], ['tab:orange', 'tab:blue', 'tab:red', 'tab:green'], ['digits', 'iris', 'wine', 'breast']):

        # Loading data and learning the subspaces
        X, y = dataset.data.T, dataset.target
        C = len(anp.unique(y))
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)
        results = anp.empty((n_splits, 4+len(signature)))
        for i, (train_index, test_index) in enumerate(skf.split(X.T, y)):
            X_train, y_train, X_test, y_test = X[:, train_index], y[train_index], X[:, test_index], y[test_index]
            U_pca, X_pca, Sw, Sb, center = generate_lda_data(X_train, y_train)
            p, n_test = X_pca.shape[0], len(test_index)
            q = signature[-1]
            U_Gr = learn(p, (q,), Sb, Sw, init="svd")
            U_Fl = learn(p, signature, Sb, Sw, init="svd")

            # Classification on Gr(p, q)
            X_Gr_train = U_Gr.T @ X_pca
            X_Gr_test = U_Gr.T @ (U_pca.T @ (X_test - center))
            clf_Gr = KNeighborsClassifier(n_neighbors=5)
            clf_Gr.fit(X_Gr_train.T, y_train)
            y_Gr_pred = clf_Gr.predict_proba(X_Gr_test.T)
            print(f"Cross-Entropy Gr({p, q})", log_loss(y_test, y_Gr_pred))
            results[i, 0] = log_loss(y_test, y_Gr_pred)

            # Classification on the subspaces of Fl(p, signature)
            y_Fl_preds = anp.zeros((n_test * C, len(signature)))
            for k, q_k in enumerate(signature):
                X_Fl_k_train = U_Fl[:, :q_k].T @ X_pca
                X_Fl_k_test = U_Fl[:, :q_k].T @ (U_pca.T @ (X_test - center))
                clf_Fl = KNeighborsClassifier(n_neighbors=5)
                clf_Fl.fit(X_Fl_k_train.T, y_train)
                y_Fl_pred = clf_Fl.predict_proba(X_Fl_k_test.T)
                print(f"Cross-Entropy Fl({p, signature}) (q = {q_k})", log_loss(y_test, y_Fl_pred))
                y_Fl_preds[:, k] = y_Fl_pred.flatten()
                if q_k == signature[-1]:
                    results[i, 1] = log_loss(y_test, y_Fl_pred)

            # Uniform ensembling of the classifier on the subspaces of Fl(p, signature)
            w_uniform = 1 / len(signature) * anp.ones(len(signature))
            y_Fl_pred_uniform = (y_Fl_preds @ w_uniform).reshape(n_test, C)
            print(f"Cross-Entropy Fl({p, signature}) (uniform)", log_loss(y_test, y_Fl_pred_uniform))
            results[i, 2] = log_loss(y_test, y_Fl_pred_uniform)

            # Optimal ensembling of the classifier on the subspaces of Fl(p, signature)
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_bin = lb.transform(y_test)
            if C == 2:
                y_bin = anp.concatenate([1 - y_bin, y_bin], axis=1)
            eps = anp.finfo(y_Fl_preds.dtype).eps  # TODO: update with reg 1e-6? to avoid very large errors. But then renormalize...
            proba = anp.clip(y_Fl_preds, eps, 1 - eps)
            import cvxpy as cp
            w = cp.Variable(len(signature))
            objective = cp.Minimize(1 / n_test * (cp.sum(- (cp.multiply(y_bin.flatten(), cp.log(proba @ w))))))
            constraints = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(objective, constraints)
            result = problem.solve()
            y_Fl_pred = (y_Fl_preds @ w.value).reshape(n_test, C)
            print(f"Cross-Entropy Fl({p, signature}) (optimal weights = {w.value})", log_loss(y_test, y_Fl_pred))
            results[i, 3] = log_loss(y_test, y_Fl_pred)
            results[i, 4:] = anp.array(w.value)

        print(anp.mean(results, axis=0))
        print(anp.std(results, axis=0))

        plt.plot(['Gr', 'Fl', 'Fl-U', 'Fl-W'], results.T[:4], alpha=.2, color=color)
        plt.plot(['Gr', 'Fl', 'Fl-U', 'Fl-W'], anp.mean(results[:, :4], axis=0), alpha=.8, color=color, label=label)

    plt.legend()
    plt.yscale('linear')
    plt.show()

