##############################
### SCRIPT FOR KFoldCV for TR
##############################
skf = StratifiedKFold(n_splits=10)
acc_Gr, acc_Fl, acc_Fl_u, acc_Fl_o = [], [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X.T, y)):
    X_train, y_train, X_test, y_test = X[:, train_index], y[train_index], X[:, test_index], y[test_index]
    U_pca, X_pca, Sw, Sb, center = generate_lda_data(X_train, y_train)
    p = X_pca.shape[0]
    U_Gr = learn_Gr(p, q, Sb, Sw, init="svd")
    U_Fl = learn_Fl(p, signature, Sb, Sw, init="svd")

    X_train_Gr = U_Gr.T @ X_pca
    clf_Gr = KNeighborsClassifier(n_neighbors=5)
    clf_Gr.fit(X_train_Gr.T, y_train)
    X_test_Gr = U_Gr.T @ U_pca.T @ (X_test - center)
    y_Gr_pred = clf_Gr.predict_proba(X_test_Gr.T)
    print(f"Classification accuracy Gr({p, q})", log_loss(y_test, y_Gr_pred))
    acc_Gr.append(log_loss(y_test, y_Gr_pred))

    n_test = X_test.shape[1]
    y_Fl_preds = anp.zeros((n_test * C, len(signature)))
    for k, q_k in enumerate(signature):
        X_train_Fl_k = U_Fl[:, :q_k].T @ X_pca
        clf_Fl = KNeighborsClassifier(n_neighbors=5)
        clf_Fl.fit(X_train_Fl_k.T, y_train)
        X_test_Fl_k = U_Fl[:, :q_k].T @ U_pca.T @ (X_test - center)
        y_Fl_pred = clf_Fl.predict_proba(X_test_Fl_k.T)
        print(log_loss(y_test, y_Fl_pred))
        y_Fl_preds[:, k] = y_Fl_pred.flatten()
    acc_Fl.append(log_loss(y_test, y_Fl_pred))

    w_uniform = 1 / len(signature) * anp.ones(len(signature))
    y_Fl_pred_uniform = (y_Fl_preds @ w_uniform).reshape(n_test, C)  # proba @ w rather?
    print(f"Classification accuracy Fl({p, signature})", log_loss(y_test, y_Fl_pred_uniform))
    acc_Fl_u.append(log_loss(y_test, y_Fl_pred_uniform))

    lb = LabelBinarizer()
    lb.fit(y)
    eps = anp.finfo(y_Fl_preds.dtype).eps  # / 1e-4
    proba = anp.clip(y_Fl_preds, eps, 1 - eps)  # clipping here ensures that proba @ w is clipped too...
    import cvxpy as cp
    w = cp.Variable(len(signature))
    objective = cp.Minimize(1 / n_test * (cp.sum(- (cp.multiply(lb.transform(y_test).flatten(), cp.log(proba @ w)) + cp.multiply((1 - lb.transform(y_test).flatten()), cp.log(1 - proba @ w))))))  # as in sklearn, we do sum on axis 1 and average on axis 0 / cp.Multiply?
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(w.value)
    # log_loss may cause undesirable behaviour, if so replace with squared loss
    y_Fl_pred = (y_Fl_preds @ w.value).reshape(n_test, C)  # proba @ w rather?
    print(f"Classification accuracy Fl({p, signature})", log_loss(y_test, y_Fl_pred))
    acc_Fl_o.append(log_loss(y_test, y_Fl_pred))
print(
f"Acc Gr = {anp.mean(acc_Gr):.3f} ({anp.std(acc_Gr):.3f})\n"
f"Acc Fl = {anp.mean(acc_Fl):.3f} ({anp.std(acc_Fl):.3f})\n"
f"Acc Fl Uniform = {anp.mean(acc_Fl_u):.3f} ({anp.std(acc_Fl_u):.3f})\n"
f"Acc Fl Optimal = {anp.mean(acc_Fl_o):.3f} ({anp.std(acc_Fl_o):.3f})\n"
)
