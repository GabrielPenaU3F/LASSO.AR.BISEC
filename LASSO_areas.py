
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
import numpy as np
from scenarios import scenario_1,scenario_2,scenario_3
import matplotlib.pyplot as plt


def area_tray_coef_lasso(X, y, fit_intercept=False, selection='cyclic'):
    '''
    Consider a grid of lambda values and apply LASSO on all lambdas in the grid.
    :param X: Features
    :param y: target
    :return: The area under the coefficient curve, normalized (so that the vector of the areas adds up to one)
    '''
    n, p = X.shape
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X), y), np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K = 100
    lambdas = np.logspace(start, end, K)  # grid values
    areas = p * [0]
    for i in range(len(lambdas)-1):
        lambda_ = lambdas[i+1]
        clf = Lasso(alpha=lambda_, fit_intercept=fit_intercept, selection=selection)
        clf.fit(X, y)
        coeficientes = clf.coef_
        for j in range(p):
            areas[j] += abs(coeficientes[j])*(lambda_-lambdas[i])
    norm = [area / sum(areas) for area in areas]
    return norm


def grafico_areas_ordenadas(scenario, n, p, s, rho=None, showfig=False, savefig=True, save_in=None,
                            cant_simu=1, selection='cyclic'):
    '''

    :param scenario: the simulated scenario
    :param n: sample size
    :param p: number of features or variables
    :param s: number of informartive variables (1 to s)
    :param cantidad_sim: number of simulations
    :param rh0: correlation coefficient
    :return: plot the areas in decreasing order
    '''

    for sim in range(cant_simu):
        if scenario == '1':
            X, y = scenario_1(n, p, s, sigma2=0.9)
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)

        rb = RobustScaler()
        X = rb.fit_transform(X)

        frecuencias_ = area_tray_coef_lasso(X, y, selection=selection)


        indeces_ordenan_frecuencias = np.argsort(frecuencias_).tolist()
        indeces_ordenan_frecuencias.reverse()
        indeces_ordenan_frecuencias_true = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                            x in range(s)]
        indeces_ordenan_frecuencias_false = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                             x not in range(s)]
        frecuencias_ord_true_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i in range(s)]
        frecuencias_ord_false_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i not in range(s)]
        fig, ax = plt.subplots()



        plt.plot(indeces_ordenan_frecuencias_true, frecuencias_ord_true_var,
                 'co', color='#1f77b4', label='Informative variables')
        plt.plot(indeces_ordenan_frecuencias_false, frecuencias_ord_false_var,
                 'co', color='#ff7f0e', label='Uninformative variables' )

        title = r'Ordered areas scenario %s  n=%s, s=%s, $\rho$=%s' % (scenario, n, s, rho)
        plt.title(title)
        plt.legend()
        if savefig:
            filename = '/SC%s_AR%s_n%s_p%s_s%s_rho%s_%s.pdf' % (
                scenario, selection, n, p, s, rho, sim)
            if save_in is not None:
                filename = save_in + filename
            plt.savefig(fname=filename)
        if showfig:
            plt.show()


