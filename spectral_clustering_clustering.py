import autograd.numpy as anp
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.model_selection import StratifiedKFold

from spectral_clustering import learn, normalized_graph_Laplacian

if __name__ == "__main__":
    anp.random.seed(42)

    # Loading data and learning the subspaces
    dataset = load_breast_cancer()  # load_digits / load_iris / load_wine / load_breast_cancer
    X, y = dataset.data.T, dataset.target
    X = anp.concatenate([X[:, y==c][:, :100//len(anp.unique(y))] for c in anp.unique(y)], axis=1)  # 100 samples equally distributed between classes
    y = anp.concatenate([y[y==c][:100//len(anp.unique(y))] for c in anp.unique(y)])
    C = len(anp.unique(y))
    signature = (1, 2, 5)  # (1, 5, 10, 20) / (1, 3, 5) / (1, 3, 5) / (1, 2, 5)
    q = signature[-1]

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits)
    results = anp.empty((n_splits, 2))
    for i, (train_index, test_index) in enumerate(skf.split(X.T, y)):
        X_train, y_train = X[:, train_index], y[train_index]
        n_train = len(train_index)
        L = normalized_graph_Laplacian(X_train)
        beta = 0.001
        U_Gr = learn(n_train, (q,), L, beta, init="svd")
        U_Fl = learn(n_train, signature, L, beta, init="svd")

        # Classification on Gr(n, q_k)
        clus_Gr = KMeans(n_clusters=C, random_state=42)
        y_Gr_pred = clus_Gr.fit_predict(U_Gr / anp.linalg.norm(U_Gr, axis=1)[:, anp.newaxis])
        print(f"RI Gr({n_train, q})", rand_score(y_train, y_Gr_pred))
        results[i, 0] = rand_score(y_train, y_Gr_pred)

        # Classification on the subspaces of Fl(n, signature)
        clus_Fl = KMeans(n_clusters=C, random_state=42)
        y_Fl_pred = clus_Fl.fit_predict(U_Fl / anp.linalg.norm(U_Fl, axis=1)[:, anp.newaxis])
        print(f"RI Fl({n_train, q})", rand_score(y_train, y_Fl_pred))
        results[i, 1] = rand_score(y_train, y_Fl_pred)

    print(anp.mean(results, axis=0))
    print(anp.std(results, axis=0))

