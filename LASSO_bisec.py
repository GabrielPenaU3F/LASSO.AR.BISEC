from LASSO_areas import area_tray_coef_lasso
from LASSO_best import best_lasso
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scenarios import scenario_1, scenario_2, scenario_3
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def seleccion_variables_bis_vs_lassomin(X, y, eps=0.05, mse='todas'):
    n, p = X.shape
    frecuencias = area_tray_coef_lasso(X, y)
    best_lambda, best_score, lambda_1se = best_lasso(X, y)
    indeces_ordenan_frecuencias = np.argsort(frecuencias).tolist()
    indeces_ordenan_frecuencias.reverse()
    clf = Lasso(alpha=best_lambda, fit_intercept=False)
    clf.fit(X, y)
    coeficientes = clf.coef_
    selected_features_LASSOMIN = [i for i in range(len(coeficientes)) if abs(coeficientes[i]) > 0.00001]

    if mse == 'LASSO.MIN':
        a = 0
        if len(selected_features_LASSOMIN) < n:
            b = len(selected_features_LASSOMIN)-1
        else:
            b = p-1
        mse = 0
        if b >= 1:
            kf = KFold(n_splits=5)
            for train, test in kf.split(X):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

                X_sel_train = [[X_train[i][j] for j in selected_features_LASSOMIN] for i in range(len(y_train))]
                reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
                X_sel_test = [[X_test[i][j] for j in selected_features_LASSOMIN] for i in range(len(y_test))]
                y_pred = reg.predict(X_sel_test)
                mse += mean_squared_error(y_test, y_pred)
            mse = mse / 5
            cant_variables = len(selected_features_LASSOMIN)
            stop = False
        else:
            stop = True
            cant_variables = 1
    if mse == 'todas':
        if p < n:
            b = p - 1
        else:
            b = n - 1

        a = 0
        mse = 0
        selected_vars = indeces_ordenan_frecuencias[:b]
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            X_sel_train = [[X_train[i][j] for j in selected_vars] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
            X_sel_test = [[X_test[i][j] for j in selected_vars] for i in range(len(y_test))]
            y_pred = reg.predict(X_sel_test)
            mse += mean_squared_error(y_test, y_pred)
        mse = mse / 5
        stop = False

        cant_variables = b

    while not stop:
        cant_variables_new = int((a+b)/2)
        if cant_variables_new != 0:
            selected_var = indeces_ordenan_frecuencias[:cant_variables_new]
            mse_new = 0
            for train, test in kf.split(X):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                X_sel_train = [[X_train[i][j] for j in selected_var] for i in range(len(y_train))]
                reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
                X_sel_test = [[X_test[i][j] for j in selected_var] for i in range(len(y_test))]
                y_pred = reg.predict(X_sel_test)
                mse_new += mean_squared_error(y_test, y_pred)
            mse_new = mse_new/5
            if mse_new < mse*(1+eps):
                #mse = mse_new
                if cant_variables_new != b:
                    b = cant_variables_new
                    cant_variables = cant_variables_new
                else:
                    stop = True
            elif cant_variables_new != a:
                a = cant_variables_new
            else:
                stop = True
        else:
            stop = True
    return indeces_ordenan_frecuencias[:cant_variables], selected_features_LASSOMIN



