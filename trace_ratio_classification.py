import autograd.numpy as anp
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier

from trace_ratio import learn, generate_lda_data

if __name__ == "__main__":
    anp.random.seed(42)

    # Loading data and learning the subspaces
    dataset = load_digits()  # load_digits / load_iris / load_wine / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    U_pca, X_pca, Sw, Sb, center = generate_lda_data(X, y)
    (p, n), C = X_pca.shape, len(anp.unique(y))
    signature = (1, 2, 5, 10)
    q = signature[-1]
    U_Gr = learn(p, (q,), Sb, Sw, init="svd")
    U_Fl = learn(p, signature, Sb, Sw, init="svd")

    # Classification on Gr(p, q)
    X_Gr = U_Gr.T @ X_pca
    clf_Gr = KNeighborsClassifier(n_neighbors=5)
    clf_Gr.fit(X_Gr.T, y)
    y_Gr_pred = clf_Gr.predict_proba(X_Gr.T)
    print(f"Cross-Entropy Gr({p, q})", log_loss(y, y_Gr_pred))

    # Classification on the subspaces of Fl(p, signature)
    y_Fl_preds = anp.zeros((n * C, len(signature)))
    for k, q_k in enumerate(signature):
        X_Fl_k = U_Fl[:, :q_k].T @ X_pca
        clf_Fl = KNeighborsClassifier(n_neighbors=5)
        clf_Fl.fit(X_Fl_k.T, y)
        y_Fl_pred = clf_Fl.predict_proba(X_Fl_k.T)
        print(f"Cross-Entropy Fl({p, signature}) (q = {q_k})", log_loss(y, y_Fl_pred))
        y_Fl_preds[:, k] = y_Fl_pred.flatten()

    # Uniform ensembling of the classifier on the subspaces of Fl(p, signature)
    w_uniform = 1 / len(signature) * anp.ones(len(signature))
    y_Fl_pred_uniform = (y_Fl_preds @ w_uniform).reshape(n, C)
    print(f"Cross-Entropy Fl({p, signature}) (uniform)", log_loss(y, y_Fl_pred_uniform))

    # Optimal ensembling of the classifier on the subspaces of Fl(p, signature)
    lb = LabelBinarizer()
    lb.fit(y)
    y_bin = lb.transform(y)
    if C == 2:
        y_bin = anp.concatenate([1 - y_bin, y_bin], axis=1)
    eps = anp.finfo(y_Fl_preds.dtype).eps
    proba = anp.clip(y_Fl_preds, eps, 1 - eps)
    import cvxpy as cp
    w = cp.Variable(len(signature))
    objective = cp.Minimize(1 / n * (cp.sum(- (cp.multiply(y_bin.flatten(), cp.log(proba @ w))))))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    y_Fl_pred = (y_Fl_preds @ w.value).reshape(n, C)
    print(f"Cross-Entropy Fl({p, signature}) (optimal weights)", log_loss(y, y_Fl_pred))
    print(w.value)
