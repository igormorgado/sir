#!/usr/bin/env python3

from sir import *
import os
import numpy as np
import matplotlib.pyplot as plt

# Define tamanho de fonte ideal para o paper
plt.rcParams["font.size"] = "10"

# Usa LaTeX para renderizar fontes
plt.rcParams["text.usetex"] = True

#####################################
# Definicao de parametros iniciais
#####################################
# Tamanho da populacao
N = 17366188

# Populacao infectada inicial
I_0 = 1

# Populacao recuperada inicial
r_0 = 0

# Decimagem. O dado final tera a ordem de $N/dec$ elementos.
# Uma boa performance de memoria deve ter $N/dec ~ [1e3, 1e5]$
dec = 1e5

# Fator de contagio
c = 2.81

# Probabilidade de contagio
p = 0.9

# Fator de recuperacao
gamma = 1/14

# Tempo de simulação em dias
T = 190

# Constantes de apoio
#beta = c * p

# Beta do artigo
beta = 0.20214286 
R_0 = beta/gamma

# Aleatoriedade
seed = 3

# Numero de simulacoes de convergencia
NEP = 1000

######################################################################
# Experimento de comparacao entre modelo deterministico 
# e estocastico (Figura1)
######################################################################

filename = 'data/cba_exp1.npz'
if os.path.isfile(filename):
    print(f"Arquivo de experimento {filename} existe. Pulando a execucao.")
    npzdata = np.load(filename)
    exp_sto = npzdata['exp_sto']
    exp_det = npzdata['exp_det']
else:
    print(f"Rodando simulação com parametros:")
    print(f"N: {N}, I_0: {I_0}, r_0: {r_0}, beta: {beta}, gamma: {gamma}")
    print(f"T: {T}, dec: {dec}, seed: {seed}")
    exp_sto = sir_sto_simul(N, I_0, r_0, beta, gamma, T, dec, seed=seed)
    exp_det = sir_det_simul(N, I_0, r_0, beta, gamma, T, len(exp_sto))
    print(f"Salvando dados em {filename}.")
    np.savez(filename, exp_sto=exp_sto, exp_det=exp_det)


print("Salvando imagens em imgs...")
fig, ax = plt.subplots(1,1, figsize=(6, 4.24), dpi=300)
sir_plot(ax, exp_sto, label=['S(t) sto',' I(t) sto', 'R(t) sto']);
sir_plot(ax, exp_det, label=['S(t) det',' I(t) det', 'R(t) det'], dashed=True);
ax.set_title(r'Comparação modelo SIR (deterministico e estocástico) $\beta: {}$'.format(beta), pad=20)
fig.savefig('imgs/cba_exp_sir_sto_vs_det.png', dpi=300)

fig, ax = plt.subplots(1,1, figsize=(6, 4.24), dpi=300)
sir_plot(ax, exp_det, label=['S(t)',' I(t)', 'R(t)'], dashed=True);
ax.set_title(r'Modelo SIR deterministico $\beta: {}$'.format(beta), pad=20)
fig.savefig('imgs/cba_exp_sir_det.png', dpi=300)

fig, ax = plt.subplots(1,1, figsize=(6, 4.24), dpi=300)
sir_plot(ax, exp_sto, label=['S(t)',' I(t)', 'R(t)']);
ax.set_title(r'Modelo SIR estocástico $\beta: {}$'.format(beta), pad=20)
fig.savefig('imgs/cba_exp_sir_stop.png', dpi=300)


######################################################################
# Simulacao de probabilidade de epidemia (Figura2)
######################################################################

filename = 'data/cba_exp2.npz'
if os.path.isfile(filename):
    print(f"Arquivo de experimento {filename} existe. Pulando a execucao.")
    npzdata = np.load(filename)
    picos = npzdata['picos']
else:
    print(f'Rodando simulação de convergencia de epidemia N={NEP}', end='', flush=True)
    picos = []
    for x in range(NEP):
        exp_sto = sir_sto_simul(N, I_0, r_0, beta, gamma, T, dec)
        picos.append(epidemy_peak(exp_sto))
        print('.', end='', flush=True)
    picos=np.array(picos)
    print()
    print(f"Salvando dados em {filename}.")
    np.savez(filename, picos=picos)

picos_com_epidemia = picos[picos < 1000]

print("Salvando imagens em imgs...")
fig, ax = plt.subplots(1,1, figsize=(6, 4.24), dpi=300)
ax.hlines(picos_com_epidemia.mean(), 0, len(picos_com_epidemia)-1, color='r', linestyles='dashed')
ax.plot(picos_com_epidemia, 's', markersize=3, alpha=0.3)
ax.hlines(picos_com_epidemia.mean(), 0, len(picos_com_epidemia)-1, color='r', linestyles='dashed', label='Média')
ax.hlines(picos_com_epidemia.mean(), 0, len(picos_com_epidemia)-1, color='r', linestyles='dashed')
ax.set_xlabel('Experimentos')
ax.set_ylabel('Dia de epidemia')
ax.set_ylim([0, 200])
ax.legend()
ax.grid(True)
fig.savefig('imgs/cba_exp_epidemias.png', dpi=300)


######################################################################
# Histograma de prob de epidemia (figura 3)
######################################################################

fig, ax = plt.subplots(1,1, figsize=(6, 4.24), dpi=300)
ax.hist(picos_com_epidemia, 100, density=True)
ax.set_xlabel('Dia de epidemia')
ax.set_ylabel('Probabilidade')
fig.savefig('imgs/cba_exp_epidemias_histograma.png', dpi=300)

