import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from LASSO_areas import area_tray_coef_lasso
import matplotlib.pyplot as plt
from LASSO_bisec import seleccion_variables_bis_vs_lassomin


def grafico_areas_ordenadas(X, y, name=None, savefig=False, showfig=True, save_in=None):

    rb = RobustScaler()
    X = rb.fit_transform(X)
    yc = y - y.mean()

    n, p = X.shape

    areas = area_tray_coef_lasso(X, yc, fit_intercept=False)
    indices_ordenan_areas = np.argsort(areas).tolist()
    indices_ordenan_areas.reverse()
    areas_reversed= [areas[i] for i in indices_ordenan_areas]
    fig, ax = plt.subplots()
    tau_list=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for tau in tau_list:
        plt.axhline(y=tau, color='black', linestyle='--',
                    linewidth=1, label='bla')
    plt.plot([i for i in range(p)], areas_reversed, 'co', color='blue', )
    title = r'Ordered areas'
    if name is not None:
        title += ' - '
        title += name

    plt.title(title)
    if savefig:
        filename = '/%s.pdf' % name
        if save_in is not None:
            filename = save_in + filename
        plt.savefig(fname=filename)
    if showfig:
        plt.show()

    return indices_ordenan_areas


def results(X, y, eps=0.05):
    rb = RobustScaler()
    X = rb.fit_transform(X)
    yc = y - y.mean()

    '''LASSO.AR.BISEC'''

    selec_LASSO_AR_BIS,selected_features_LASSOMIN = seleccion_variables_bis_vs_lassomin(X, yc, eps=eps, mse='LASSO.MIN')

    return selec_LASSO_AR_BIS,selected_features_LASSOMIN


'''STUDENTs MAT'''

datos_estudiantes_mat = pd.read_csv('resources/student-mat.csv', delimiter=';')
feature_names = datos_estudiantes_mat.columns[:30]
target_1_name = datos_estudiantes_mat.columns[30]
target_2_name = datos_estudiantes_mat.columns[31]
target_3_name = datos_estudiantes_mat.columns[32]

X_Df = datos_estudiantes_mat[feature_names]
X_Df_encod = pd.get_dummies(X_Df)
features = X_Df_encod.columns
X = X_Df_encod.values.tolist()


#y_1 = np.array(datos_estudiantes_mat[target_1_name].tolist())
#y_2 = datos_estudiantes_mat[target_2_name].tolist()
y_3 = np.array(datos_estudiantes_mat[target_3_name].tolist())
y = y_3

name = 'Student-Mat'
grafico_areas_ordenadas(X, y, name, savefig=True, showfig=True, save_in='results/')

eps_list = [0.01, 0.005, 0.0025]
for eps in eps_list:
    selec_bisec, selec_LASSO_MIN = results(X, y_3, eps=eps)
    print(eps)
    print(selec_bisec, selec_LASSO_MIN)
    for i in selec_bisec:
        print(features[i])
    print(len(selec_bisec), len(selec_LASSO_MIN))

