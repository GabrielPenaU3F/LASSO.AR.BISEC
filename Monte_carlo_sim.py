
import numpy as np
from sklearn.preprocessing import RobustScaler
from scenarios import scenario_1, scenario_2, scenario_3
import matplotlib.pyplot as plt
from LASSO_bisec import seleccion_variables_bis_vs_lassomin

def monte_carlo_bisec_vs_lassomin(scenario, n, p, s,
                                  rho=None, cant_clusters = None, cant_sim=1, eps=0.05, mse='todas'):

    if scenario == '1' or scenario == '3' or scenario=='1random' or scenario=='3random':
        cant_false_fea_selec_LASSOAREAS = 0
        cant_true_fea_selec_LASSOAREAS = 0
        cant_false_fea_selec_LASSOMIN = 0
        cant_true_fea_selec_LASSOMIN = 0
    else:
        cant_clusters = 10
        cant_false_fea_selec_LASSOAREAS = cant_clusters * [0]
        cant_true_fea_selec_LASSOAREAS = cant_clusters * [0]
        cant_false_fea_selec_LASSOMIN = cant_clusters * [0]
        cant_true_fea_selec_LASSOMIN = cant_clusters * [0]

    for _ in range(cant_sim):
        if scenario == '1':
            X, y = scenario_1(n, p, s, sigma2=0.9)
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)

        rb = RobustScaler()
        X = rb.fit_transform(X)

        if scenario == '1' or scenario == '3':
            selected_features_LASSOAREAS, selected_features_LASSOMIN = seleccion_variables_bis_vs_lassomin(
                X, y, eps=eps, mse=mse)
            false_fea_selec_LASSOAREAS = [i for i in selected_features_LASSOAREAS if i not in range(s)]
            cant_false_fea_selec_LASSOAREAS += len(false_fea_selec_LASSOAREAS)
            false_fea_selec_LASSOMIN = [i for i in selected_features_LASSOMIN if i not in range(s)]
            cant_false_fea_selec_LASSOMIN += len(false_fea_selec_LASSOMIN)
            true_fea_selec_LASSOAREAS = [i for i in range(s) if i in selected_features_LASSOAREAS]
            cant_true_fea_selec_LASSOAREAS += len(true_fea_selec_LASSOAREAS)
            true_fea_selec_LASSOMIN = [i for i in range(s) if i in selected_features_LASSOMIN]
            cant_true_fea_selec_LASSOMIN += len(true_fea_selec_LASSOMIN)
        else:
            selected_features_LASSOAREAS, selected_features_LASSOMIN = seleccion_variables_bis_vs_lassomin(
                X, y, eps=eps, mse=mse)
            for resto in range(cant_clusters):
                false_fea_selec_LASSO_AREAS = [i for i in selected_features_LASSOAREAS if
                                               (i % cant_clusters) == resto and i not in range(s)]
                cant_false_fea_selec_LASSOAREAS[resto] += len(false_fea_selec_LASSO_AREAS)
                true_fea_selec_LASSO_AREAS = [i for i in range(s) if
                                            (i in selected_features_LASSOAREAS and (i % cant_clusters) == resto)]
                cant_true_fea_selec_LASSOAREAS[resto] += len(true_fea_selec_LASSO_AREAS)

                false_fea_selec_LASSO_MIN = [i for i in selected_features_LASSOMIN if
                                               (i % cant_clusters) == resto and i not in range(s)]
                cant_false_fea_selec_LASSOMIN[resto] += len(false_fea_selec_LASSO_MIN)
                true_fea_selec_LASSO_MIN = [i for i in range(s) if
                                              (i in selected_features_LASSOMIN and (i % cant_clusters) == resto)]
                cant_true_fea_selec_LASSOMIN[resto] += len(true_fea_selec_LASSO_MIN)



    if scenario == '1' or scenario == '3':
        mean_cant_false_fea_selec_LASSOAREAS = cant_false_fea_selec_LASSOAREAS / cant_sim
        mean_cant_true_fea_selec_LASSOAREAS = cant_true_fea_selec_LASSOAREAS / cant_sim

        mean_cant_false_fea_selec_LASSOMIN = cant_false_fea_selec_LASSOMIN / cant_sim
        mean_cant_true_fea_selec_LASSOMIN = cant_true_fea_selec_LASSOMIN / cant_sim

        return (mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS,
                mean_cant_false_fea_selec_LASSOMIN, mean_cant_true_fea_selec_LASSOMIN)

    else:
        mean_cant_false_fea_selec_LASSOAREAS = [cant / cant_sim for cant in
                                              cant_false_fea_selec_LASSOAREAS]
        mean_cant_true_fea_selec_LASSOAREAS = [cant / cant_sim for cant in cant_true_fea_selec_LASSOAREAS]

        mean_cant_false_fea_selec_LASSOMIN = [cant / cant_sim for cant in
                                                cant_false_fea_selec_LASSOMIN]
        mean_cant_true_fea_selec_LASSOMIN = [cant / cant_sim for cant in cant_true_fea_selec_LASSOMIN]

        return (mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS,
                mean_cant_false_fea_selec_LASSOMIN, mean_cant_true_fea_selec_LASSOMIN)


def grafico_montecarlo_bisec_vs_lassomin(scenario, n_list, p, s, mse='todas', rho=None, cant_clusters=None,
                                         cant_sim=1, eps=0.05, showfig=False, savefig=False, save_in=None):

    if scenario == '1' or scenario == '3':
        true_LASSOAREAS_nlist = []
        false_LASSOAREAS_nlist = []
        true_LASSOMIN_nlist = []
        false_LASSOMIN_nlist = []
        for n in n_list:
            mean_false_LASSOAREAS, mean_true_LASSOAREAS,mean_false_LASSOMIN, mean_true_LASSOMIN =\
                monte_carlo_bisec_vs_lassomin(scenario, n, p, s, rho, cant_sim=cant_sim, eps=eps, mse=mse)
            true_LASSOAREAS_nlist.append(mean_true_LASSOAREAS)
            false_LASSOAREAS_nlist.append(mean_false_LASSOAREAS)
            true_LASSOMIN_nlist.append(mean_true_LASSOMIN)
            false_LASSOMIN_nlist.append(mean_false_LASSOMIN)

        n_list_string = []
        for n in n_list:
            text = 'n=%s' % (str(n))
            n_list_string.append(text)

        fig, ax = plt.subplots()
        ax.bar(n_list_string, true_LASSOAREAS_nlist, width=1, edgecolor="white", linewidth=0.7,label='Informative variables')
        ax.bar(n_list_string, false_LASSOAREAS_nlist, bottom=true_LASSOAREAS_nlist,
               width=1, edgecolor="white", linewidth=0.7,label='Uninformative variables')
        x_label = 'sample size (n)'
        plt.xlabel(x_label)
        y_label = 'Selected variables'
        plt.ylabel(y_label)
        text = r'LASSO.AR Bisec $\epsilon = $ %s' %(eps)
        text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s, rho)
        plt.title(text_)
        plt.legend()

        if savefig:
            filename = save_in + 'scenario%s/SCE%s_LASSO_ARBIS_esp=%s_s%s_rho%s_cant_sim%s.pdf' % (
                scenario, scenario, eps, s, rho, cant_sim)
            plt.savefig(fname=filename)
        if showfig:
            plt.show()

        fig, ax = plt.subplots()
        ax.bar(n_list_string, true_LASSOMIN_nlist, width=1, edgecolor="white", linewidth=0.7,label='Informative variables')
        ax.bar(n_list_string, false_LASSOMIN_nlist, bottom=true_LASSOMIN_nlist,
               width=1, edgecolor="white", linewidth=0.7,label='Uninformative variables')
        x_label = 'sample size (n)'
        plt.xlabel(x_label)
        y_label = 'Selected variables'
        plt.ylabel(y_label)
        text = r'LASSO.MIN '
        text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s, rho)
        plt.title(text_)
        plt.legend()

        if savefig:
            filename = save_in + 'scenario%s/SCE%s_LASSO_MIN_s%s_rho%s_cant_sim%s.pdf' % (
                scenario, scenario, s, rho, cant_sim)
            plt.savefig(fname=filename)
        if showfig:
            plt.show()

    else:
        for n in n_list:
            mean_false_LASSOAREAS, mean_true_LASSOAREAS, mean_false_LASSOMIN, mean_true_LASSOMIN = (
                monte_carlo_bisec_vs_lassomin(scenario, n, p, s, rho=rho, cant_clusters=cant_clusters,
                                              cant_sim=cant_sim, eps=0.05, mse=mse))
            x = np.arange(cant_clusters)
            fig, ax = plt.subplots()
            ax.bar(x, mean_true_LASSOAREAS, width=1, edgecolor="white", linewidth=0.7,label='Informative variables')
            ax.bar(x, mean_false_LASSOAREAS, bottom=mean_true_LASSOAREAS,
                   width=1, edgecolor="white", linewidth=0.7, label='Uninformative variables')
            x_label = 'mod %s' % (str(cant_clusters))
            plt.xlabel(x_label)
            y_label = 'Selected variables'
            plt.ylabel(y_label)
            text = r'LASSO.AR Bisec $\epsilon = $ %s' %(eps)
            text_ = '%s p=%s, n=%s, s=%s, rho=%s' % (text, p, n, s, rho)
            plt.title(text_)
            plt.legend()

            if savefig:
                filename = save_in + 'scenario%s/SC%s_LASSO_AR_BISeps=%s_n=%s_p=%s_s=%s_rho=%s_cantsim=%s.pdf' % (
                scenario, scenario, eps, n, p, s, rho, cant_sim)
                plt.savefig(fname=filename)
            if showfig:
                plt.show()

            fig, ax = plt.subplots()
            ax.bar(x, mean_true_LASSOMIN, width=1, edgecolor="white", linewidth=0.7, label='Informative variables')
            ax.bar(x, mean_false_LASSOMIN, bottom=mean_true_LASSOMIN,
                   width=1, edgecolor="white", linewidth=0.7, label='Uninformative variables')
            x_label = 'mod %s' % (str(cant_clusters))
            plt.xlabel(x_label)
            y_label = 'Selected variables'
            plt.ylabel(y_label)
            text = r'LASSO.MIN'
            text_ = '%s p=%s, n=%s, s=%s, rho=%s' % (text, p, n, s, rho)
            plt.title(text_)
            plt.legend()

            if savefig:
                filename = save_in + 'scenario%s/SC%s_LASSO_MIN_n=%s_p=%s_s=%s_rho=%s_cantsim=%s.pdf' % (
                    scenario, scenario, n, p, s, rho, cant_sim)
                plt.savefig(fname=filename)
            if showfig:
                plt.show()


'''
Monte Carlo simulation for scenario 1
'''
n_list = [100,200,400]
p = 50
s = 10
scenario = '1'
rho_list = [None]
sigma2 = 0.9
cant_sim = 1000
eps_list = [0.01]
save_in = 'results/'
for eps in eps_list:
    for rho in rho_list:
        grafico_montecarlo_bisec_vs_lassomin(scenario, n_list, p, s, mse='LASSO.MIN', rho=rho,
                                             cant_sim=cant_sim, eps=eps,
                                             showfig=False, savefig=True, save_in=save_in)


'''
Monte Carlo simulation for scenario 2
'''
n_list = [400]
p = 50
s = 10
scenario = '2'
rho_list = [0.2,0.5,0.9]
cant_clusters=10
sigma2 = 0.9
cant_sim = 1000
eps_list = [0.01]
save_in = 'results/'
for eps in eps_list:
    for rho in rho_list:
        grafico_montecarlo_bisec_vs_lassomin(scenario, n_list, p, s, mse='LASSO.MIN', rho=rho,
                                             cant_clusters=cant_clusters, cant_sim=cant_sim, eps=eps,
                                             showfig=False, savefig=True, save_in=save_in)


'''
Monte Carlo simulation for scenario 3
'''
n_list = [100,200,400]
p = 50
s = 10
scenario = '3'
rho_list = [0.2, 0.5, 0.9]
sigma2 = 0.9
cant_sim = 1000
eps_list = [0.01]
save_in = 'results/'
for eps in eps_list:
    for rho in rho_list:
        grafico_montecarlo_bisec_vs_lassomin(scenario, n_list, p, s, mse='LASSO.MIN', rho=rho,
                                             cant_sim=cant_sim, eps=eps,
                                             showfig=False, savefig=True, save_in=save_in)
