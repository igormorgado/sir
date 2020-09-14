from sir import *
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile, memory_usage
import timeitplot as tp

# Define formato de exibicao do numpy
np.set_printoptions(suppress=True, precision=3)

# Define tamanho de fonte ideal para o paper
plt.rcParams["font.size"] = "10"

# Usa LaTeX para renderizar fontes
plt.rcParams["text.usetex"] = True


# Tamanho da população
N = int(1e7)

# Populacao infectada inicial
I_0 = 2

# Populacao recuperada inicial
r_0 = 0

# Decimagem. O dado final tera a ordem de $N/dec$ elementos.
# Uma boa performance de memoria deve ter $N/dec ~ [1e3, 1e5]$
dec = int(1e5)

# Fator de contagio
c = 2.81

# Probabilidade de contagio
p = 0.9

# Fator de recuperacao
gamma = 1/14

# Tempo de simulação em dias
T = 200

# Constantes de apoio
#beta = c * p
beta = 0.20214286 
R_0 = beta/gamma

seed = 0

print("""
#
# EXPERIMENT 1: Memory footprint
# 
""")
#
# EXPERIMENT 1
# 
# Memory footprint: Naive vs Optimized.
# 
# Computing 60 data points from 1e2 to 2e8
#
# Upper limit was defined by memory used in naive method (61GB)

experiments = []
memory_file = 'data/memory_naive.npz'
N_range = np.array([
    int(1e2), int(2e2), int(3e2), int(4e2), int(5e2), int(6e2), int(7e2), int(8e2), int(9e2),
    int(1e3), int(2e3), int(3e3), int(4e3), int(5e3), int(6e3), int(7e3), int(8e3), int(9e3),
    int(1e3), int(2e3), int(3e3), int(4e3), int(5e3), int(6e3), int(7e3), int(8e3), int(9e3),
    int(1e4), int(2e4), int(3e4), int(4e4), int(5e4), int(6e4), int(7e4), int(8e4), int(9e4),
    int(1e5), int(2e5), int(3e5), int(4e5), int(5e5), int(6e5), int(7e5), int(8e5), int(9e5),
    int(1e6), int(2e6), int(3e6), int(4e6), int(5e6), int(6e6), int(7e6), int(8e6), int(9e6),
    int(1e7), int(2e7), int(3e7), int(4e7), int(5e7), int(6e7), int(7e7), int(8e7), int(9e7),
    int(1.1e8), int(1.2e8), int(1.3e8), int(1.4e8), int(1.5e8), int(1.6e8), int(1.7e8), int(1.8e8), int(1.9e8), int(2.0e8) ])
if os.path.exists(memory_file):
    print(f'Experiment  {memory_file} exists. Loading')
    data = np.load(memory_file)
    resmem = data['resmem']
    args = data['args']
    N_range = data['N_range']
    print(f'Variables "resmem", "args", "N_range" loaded.')
else:
    print(f'Running experiment {memory_file}.')
    resmem = []
    for n in N_range:
        print(f'{n}, ', end='', flush=True)
        args = (n, I_0, r_0, beta, gamma, T)
        kw = {'seed': 0}
        rss = memory_usage((sir_sto_naive, args, kw))
        resmem.append([n, max(rss)])
    resmem = np.array(resmem)
    print('. Finished')
    np.savez(memory_file, resmem=resmem, N_range=N_range, args=(I_0, r_0, beta, gamma, T))

experiments.append(resmem)


memory_file = 'data/memory_simul.npz'
if os.path.exists(memory_file):
    print(f'Experiment  {memory_file} exists. Loading')
    data = np.load(memory_file)
    resmem = data['resmem']
    args = data['args']
    N_range = data['N_range']
    print(f'Variables "resmem", "args", "N_range" loaded.')
else:
    print(f'Running experiment {memory_file}.')
    resmem = []
    for n in N_range:
        print(f'{n}, ', end='', flush=True)
        args = (n, I_0, r_0, beta, gamma, T)
        d = 0
        if(n > 1000):
            d = int(n/100)
        kw = {'seed': 0, 'decimation': d}
        rss = memory_usage((sir_sto_simul, args, kw))
        resmem.append([n, max(rss)])
    resmem = np.array(resmem)
    print('. Finished')
    np.savez(memory_file, resmem=resmem, N_range=N_range, args=(I_0, r_0, beta,
        gamma, T, d))

experiments.append(resmem)

filename = 'imgs/enmc_exp_memory_naive_and_simul.png'
print(f"Salvando imagem do experimento 1 em {filename}")
fig, ax = plt.subplots(1,1, figsize=(6,3.37), dpi=150, constrained_layout=True)
ax.plot(experiments[0][:,0], experiments[0][:,1]/1000,'b-', markersize=2, alpha=.4, label='Regular');
ax.plot(experiments[1][:,0], experiments[1][:,1]/1000,'r-', markersize=2, alpha=.4, label='Otimizado');
ax.plot(experiments[0][:,0], experiments[0][:,1]/1000,'bo', markersize=2, alpha=.4);
ax.plot(experiments[1][:,0], experiments[1][:,1]/1000,'ro', markersize=2, alpha=.4);
ax.set_xlabel("Indivíduos")
ax.set_ylabel("Memória (GB)")
ax.set_yscale('linear')
ax.set_xscale('log')
ax.legend()
fig.savefig(filename, dpi=300)


print("""
#
# EXPERIMENT 2: Compute time
# 
""")
# Compute time: Naive vs Optimized.
# 
# Computing 60 data points from 1e2 to 2e8
#
# Upper limit was defined by memory used in naive method (61GB)
# Grafico que mostra o crescimento do processamento (do inocente com o
# crescimento da populacao)


runs = 5
N_range = np.array([
    int(1e2), int(2e2), int(3e2), int(4e2), int(5e2), int(6e2), int(7e2), int(8e2), int(9e2),
    int(1e3), int(2e3), int(3e3), int(4e3), int(5e3), int(6e3), int(7e3), int(8e3), int(9e3),
    int(1e4), int(2e4), int(3e4), int(4e4), int(5e4), int(6e4), int(7e4), int(8e4), int(9e4),
    int(1e5), int(2e5), int(3e5), int(4e5), int(5e5), int(6e5), int(7e5), int(8e5), int(9e5),
    int(1e6), int(2e6), int(3e6), int(4e6), int(5e6), int(6e6), int(7e6), int(8e6), int(9e6),
    int(1e7), int(2e7), int(3e7), int(4e7), int(5e7), int(6e7), int(7e7), int(8e7), int(9e7),
    int(1.1e8), int(1.2e8), int(1.3e8), int(1.4e8), int(1.5e8), int(1.6e8), int(1.7e8), int(1.8e8), int(1.9e8), int(2.0e8),
    ])

compute_file = 'data/compute_naive.json'
if os.path.exists(compute_file):
    print(f'Experiment  {compute_file} exists. Loading')
    with open(compute_file, 'r') as fd:
         perf = json.loads(fd.read())
    print(f'Variables "perf" loaded.')
else:
    print(f'Running experiment {compute_file}.')
    functions = [
        f"sir_sto_naive_nonumba({{0}}, {I_0}, {r_0}, {beta}, {gamma}, {T}, seed={seed})",
        f"sir_sto_simul({{0}}, {I_0}, {r_0}, {beta}, {gamma}, {T}, -1, seed={seed})",
    ]
    data = tp.timeit_compare(functions, N_range, setups='main', number=runs)
    labels = [ 'Regular', 'Otimizado' ]
    perf = {}
    for key,n_key in zip(data.keys(), labels):
        perf[n_key] = data[key]
    del(data)
    print('. Finished')
    with open(compute_file, 'w') as fd:
        fd.write(json.dumps(perf, cls=tp.NpEncoder))


#fig, ax = plt.subplots(figsize=(6,4.24),dpi=150, constrained_layout=True)
fig, ax = plt.subplots(figsize=(6,3.375),dpi=150, constrained_layout=True)
ax = tp.timeit_plot2D(perf, ax, 'Indivíduos', 'Comparativo de performance')
ax.set_xscale('log')

filename = f'imgs/enmc_exp_computing.png'
print(f"Saving {filename}")
fig.savefig(filename, dpi=300)


print("""
#
# EXPERIMENT 3: Results
# 
""")
# Experiment 3
#
# Results are similar: Naive vs Optimized
#
# Shows up that results from Naive and Optimized are equivalent
results_file = 'data/results.npz'
seed=0
if os.path.exists(results_file):
    print(f'Experiment {results_file} exists. Loading')
    data = np.load(results_file)
    expsto = data['expsto']
    expsto2 = data['expsto2']
    expnai = data['expnai']
    args = data['args']
    print(f'Variables "expsto", "expsto2", "expnai", "args" loaded.')
else:
    print(f'Running experiment {results_file}.')
    expsto = sir_sto_simul(N, I_0, r_0, beta, gamma, T, dec, seed=seed)
    expsto2 = sir_sto_simul(N, I_0, r_0, beta, gamma, T, dec/10, seed=seed)
    expnai = sir_sto_naive(N, I_0, r_0, beta, gamma, T, seed=seed)
    expnai = np.array(expnai).transpose()
    np.savez_compressed(results_file, expsto=expsto, expnai=expnai, expsto2=expsto2,
            args=(N, I_0, r_0, beta, gamma, T, dec, seed))

fig, ax = plt.subplots(1, 1, figsize=(6,3.375), dpi=150, constrained_layout=True)
sir_plot(ax, expnai[::1000,:], label=['S(t) regular',' I(t) regular', 'R(t) regular'], dashed=True);
sir_plot(ax, expsto, label=['S(t) otimizado',' I(t) otimizado', 'R(t) otimizado']);
ax.plot(expsto[:,0], expsto[:,2], 'ro', markersize=2, alpha=.3)
#ax.set_title(r'Comparison SIR Model (stochastic and deterministic) $\beta: {}$'.format(beta), pad=20)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Indivíduos")
leg = ax.legend()
filename = 'imgs/enmc_exp_results.png'
print(f"Salvando imagem do experimento 3 em {filename}")
fig.savefig(filename, dpi=300)

fig.set_figwidth(3)
fig.set_figheight(1.68)
ax.set_ylim([9.2e6, 1e7])
ax.set_xlim([70, 110])
leg.remove()
ax.plot(expsto[:,0], expsto[:,1], 'go', markersize=3, alpha=.7)
filename = 'imgs/enmc_exp_results_inset_low_dec.png'
print(f"Salvando imagem do experimento 3 em {filename}")
fig.savefig(filename, dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(3,1.68), dpi=150, constrained_layout=True)
sir_plot(ax, expnai[::1000,:], label=['S(t) regular',' I(t) regular', 'S(t) regular'], dashed=True);
sir_plot(ax, expsto2, label=['S(t) otimizado',' I(t) otimizado', 'S(t) otimizado']);
ax.plot(expsto2[:,0], expsto2[:,1], 'go', markersize=3, alpha=.7)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Indivíduos")
ax.set_ylim([9.2e6, 1e7])
ax.set_xlim([70, 110])
ax.get_legend().remove()
filename = 'imgs/enmc_exp_results_inset_good_dec.png'
print(f"Salvando imagem do experimento 3 em {filename}")
fig.savefig(filename, dpi=300)

