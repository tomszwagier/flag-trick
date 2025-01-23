import autograd.numpy as anp
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score

from spectral_clustering import learn, normalized_graph_Laplacian

if __name__ == "__main__":
    anp.random.seed(42)

    # Loading data and learning the subspaces
    dataset = load_iris()  # load_digits / load_iris / load_wine / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    X = anp.concatenate([X[:, y==c][:, :100//len(anp.unique(y))] for c in anp.unique(y)], axis=1)  # 100 samples equally distributed between classes
    y = anp.concatenate([y[y==c][:100//len(anp.unique(y))] for c in anp.unique(y)])
    (p, n), C = X.shape, len(anp.unique(y))
    L = normalized_graph_Laplacian(X)
    signature = (1, 3, 5)
    q = signature[-1]
    beta = 0.001
    U_Gr = learn(n, (q,), L, beta, init="svd")
    U_Fl = learn(n, signature, L, beta, init="svd")

    # Classification on Gr(n, q_k)
    for k, q_k in enumerate(signature):
        U_Gr = learn(n, (q_k,), L, beta, init="svd")
        clus_Gr = KMeans(n_clusters=C, random_state=42)
        y_Gr_pred = clus_Gr.fit_predict(U_Gr / anp.linalg.norm(U_Gr, axis=1)[:, anp.newaxis])
        print(f"RI Gr({n, q_k})", rand_score(y, y_Gr_pred))

    # Classification on the subspaces of Fl(n, signature)
    y_Fl_pred_w = anp.zeros((n, C))
    weights = anp.zeros((len(signature)))
    for k, q_k in enumerate(signature):
        U_Fl_k = U_Fl[:, :q_k]
        clus_Fl = KMeans(n_clusters=C, random_state=42)
        y_Fl_pred = clus_Fl.fit_predict(U_Fl_k / anp.linalg.norm(U_Fl_k, axis=1)[:, anp.newaxis])
        w = rand_score(y, y_Fl_pred)
        weights[k] = w
        print(f"RI Fl({n, signature}) (q = {q_k})", w)
        lb = LabelBinarizer()
        lb.fit(y_Fl_pred)
        y_bin = lb.transform(y_Fl_pred)
        if C == 2:
            y_bin = anp.concatenate([1 - y_bin, y_bin], axis=1)
        y_Fl_pred_w = y_Fl_pred_w + w * y_bin
    y_Fl_pred_w_max = anp.argmax(y_Fl_pred_w, axis=1)
    print(f"RI Fl({n, signature}) (performance weights)", rand_score(y, y_Fl_pred_w_max))
    weights = weights / anp.sum(weights)
    print(weights)
