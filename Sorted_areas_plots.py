from LASSO_areas import grafico_areas_ordenadas

'''
Sorted areas plot for scenario 1
'''

scenario = '1'
n_list = [100, 200, 400]
p = 50
s = 10
rho_list = [None]
cant_sim = 1
selection ='cyclic'
save_in = 'results/scenario%s' % scenario
for n in n_list:
    for rho in rho_list:
        grafico_areas_ordenadas(scenario, n, p, s, rho=rho, showfig=True, savefig=True,
                                save_in=save_in, cant_simu=cant_sim, selection=selection)



'''
Sorted areas plot for scenario 2
'''

scenario = '2'
n_list = [100, 200, 400]
p = 50
s_list = [10,15,20]
rho_list = [0.2, 0.5, 0.9]
cant_sim = 1
cant_clusters = 10
selection = 'cyclic'
save_in = 'results/scenario%s' % scenario
for n in n_list:
    for rho in rho_list:
        for s in s_list:
            grafico_areas_ordenadas(scenario, n, p, s, rho=rho, showfig=True, savefig=True,
                                save_in=save_in, cant_simu=cant_sim, selection=selection)


'''
Sorted areas plot for scenario 3
'''


scenario = '3'
n_list = [100, 200, 400]
p = 50
s = 10
rho_list = [0.2,0.5,0.9]
cant_sim = 1
selection = 'cyclic'
save_in = 'results/scenario%s' % scenario
for n in n_list:
    for rho in rho_list:
        grafico_areas_ordenadas(scenario, n, p, s, rho=rho, showfig=True, savefig=True,
                                save_in=save_in, cant_simu=cant_sim, selection=selection)
