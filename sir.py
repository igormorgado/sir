#!/usr/bin/env python

import sys
import math
import random
import numba
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def sir_plot(ax: plt.axes,
             data: np.ndarray,
             label: Tuple[str, str, str] = ['S(t), I(t), R(t)'],
             dashed: bool = False,
             title: str = 'SIR Model'
            ) -> plt.axes:
    """Returns a Plot to SIR Model given (time, S, I, R) array"""
    t, s, i, r = data[:,0], data[:, 1], data[:, 2], data[:, 3]
    line_0, = ax.plot(t, s, 'g-', alpha=.5, label=label[0])
    line_1, = ax.plot(t, i, 'r-', alpha=.5, label=label[1])
    line_2, = ax.plot(t, r, 'b-', alpha=.5, label=label[2])
    lines = [line_0, line_1, line_2]
    
    if dashed:
        for line in lines:
            line.set_linestyle('dashed')
        
    ax.set_xlabel('Tempo (dias)')
    ax.set_ylabel('População (indivíduos) ')
    ax.legend()
    return ax


def epidemy_peak(data):
    return data[np.argmax(data[:,2]),0]

@numba.njit
def sir_sto_iter(population: int,
                 infected: int,
                 recovered: int,
                 infection_rate: float,
                 recovery_rate: float,
                 seed: Optional[int] = None
                ) -> Tuple[float, int, int, int]:
    """Iteration step for sir stochastic model
    
    Parameters:
        population (int): $N$: Value of initial population
        infected (int): $I_0$: Value of initial infected population
        recovered (int): $r_0$: Value of inital recovered population
        infection_rate (float): $\beta$: Rate of disease propagation. It's computed
                as $\beta= c \cdot \rho$, where $c$ is the number of suscetible
                individuals exposed to a infected individual and $\rho$ is the 
                probability of infection given the contact.
        recovery_rate (float): $\gamma$: Rate of recovery.
        seed (int): A random seed value (default random).
    Returns:
        Tuple[time_evaluated (float), susceptible(int), infected(int), recovered(int)]:
            Time evaluated since the start, number of susceptible in total population,
            number of infected population and number of recovered population (cannot be 
            infected anymore)
    """
    t, s, i, r = 0, population-infected, infected, recovered
    yield (t, s, i, r)
    
    # Numba complains
    # eps = np.finfo(float).eps
    eps = 1e-16
    
    beta_normalized = infection_rate/population
    while True:
        beta_n_s = beta_normalized*s
        interevent_time = (beta_n_s*i) + (recovery_rate*i) + eps
        t -= math.log(random.random())/interevent_time
        prob = beta_n_s/(beta_n_s + recovery_rate)
        if random.random() <= prob:
                # Infected
                s -= 1
                i += 1
        else:
                # Recovered
                i -= 1
                r += 1
        
        yield (t, s, i, r)


@numba.njit
def sir_sto_simul(N: int,
              I_0: int,
              r_0: int,
              beta: float,
              gamma: float,
              T: float,
              decimation: int = 0,
              seed: Optional[int] = None
             ) -> np.ndarray:
    """SIR estochastic simulation
    
    Parameters:
        N (int): Value of initial population
        I_0 (int): Value of initial infected population
        r_0 (int): Value of initial recovered population
        beta (float): Rate of disease propagation. It's computed
            as $\beta= c \cdot \rho$, where $c$ is the number of suscetible
            individuals exposed to a infected individual and $\rho$ is the 
            probability of infection given the contact.
        gamma (float): Rate of recovery.
        T (float): Time of simulation in days,
        decimation (int): Just stores the i-th data sample. Reduces memory
            consuption to ratio $1/i$. Default is to store all data samples.
            Not recommended for large populations or very large simulation time.
        seed (int): A random seed value (default random).
    Returns:
        np.ndarray[time_evaluated (float), susceptible(float), infected(float), recovered(float)]:
            Time evaluated since the start, number of susceptible in total population,
            number of infected population and number of recovered population (cannot be 
            infected anymore)
    """
    # Creates the SIR generator
    sir = sir_sto_iter(N, I_0, r_0, beta, gamma, seed)
    
    data = [next(sir)]
    
    # While there is simulation time and infecteds
    while (data[-1][0] < T) and (data[-1][2] > 0):
        for _ in range(decimation):
            next(sir)
        data.append(next(sir))
    
    return np.array(data)


def sir_det_iter(y: Tuple[float, float, float],
                 t: float,
                 N: float,
                 beta: float,
                 gamma: float
                ) -> Tuple[float, float, float, float]:
    S, I, R = y
    return ((-beta/N)*S*I, (beta/N)*S*I - gamma*I, gamma*I)    


def sir_det_simul(N: float,
                  I: float,
                  R: float,
                  beta: float,
                  gamma: float,
                  T: float,
                  n_samples: int
                 ) -> np.ndarray:
    """Executes the deterministic simulation of SIR model
 
    Parameters:
        N (int): Value of initial population
        I (int): Value of initial infected population
        R (int): Value of initial recovered population
        beta (float): Rate of disease propagation. It's computed
            as $\beta= c \cdot \rho$, where $c$ is the number of suscetible
            individuals exposed to a infected individual and $\rho$ is the 
            probability of infection given the contact.
        gamma (float): Rate of recovery.
        T (float): Time of simulation in days,
        n_samples (int): Number of samples to simulate
    Returns:
        np.ndarray[time_evaluated (float), susceptible(float), infected(float), recovered(float)]:
            Time evaluated since the start, number of susceptible in total population,
            number of infected population and number of recovered population (cannot be 
            infected anymore)
    """
    S = N - (I + R)
    y = (S, I, R)
    t = np.linspace(0, T, n_samples)
    sol = scipy.integrate.odeint(sir_det_iter, y, t, args=(N, beta, gamma))
    return np.concatenate((np.atleast_2d(t).transpose(), sol), axis=1)


def SIR_stochastic(N, infectados, beta, gamma, tempo):
    # SIR MIchelle // manter para historico
    # Criando listas para o tempo e os compartimentos
    t = [] 
    s = []
    i = []
    r = []
    
    # Acrescenta-se a cada lista o valor inicial de cada compartimento e 
    # também do instante inicial t = 0.
    # No modelo SIR, considera-se que o compartimento de indíviduos 
    # r(recuperados) = 0. Logo, a população total N é dividida entre os
    # compartimentos s(suscetíveis) e i(infectados).
    t.append(0)
    s.append(N-infectados)
    i.append(infectados)
    r.append(0)
    j = 0
    # Enquanto houver indivíduos infectados na população e os instantes 
    # de tempo avaliados sejam menores que o tempo total considerado
    while i[j]>0 and t[j]<tempo:                             
        # São utilizadas duas variáveis escolhidas aleatoriamente. 
        # Uma para ser utilizada no interevent time e outra para 
        # compararmos com a probabilidade de infecção estabelecida.
        u1 = random()                                        
        u2 = random()
        
        a = (beta/N)*s[j]*i[j] + gamma*i[j]
        # fórmula para calcular o interevent time
        t.append(t[j] - math.log(u1)/a)                      

        # Quando comparamos a variável escolhida aleatoriamente com a 
        #probabilidade de infecção, verifica-se que se u2 for menor ou 
        # igual a prob significa que ...
        
        # probabilidade de infecção estabelecida.
        prob = (beta*s[j]/N)/(beta*s[j]/N + gamma)           
        if u2 <= prob:
            # ... um indíviduo suscetível foi infectado. 
            # Logo, há alteração nesses compartimentos.
            s.append(s[j] - 1)                               
            i.append(i[j] + 1)
            # Enquanto que no compartimento de recuperados não há alteração.
            r.append(r[j])                                   
        else:                                                
            # Se u2 for maior que essa prob significa que algum indíviduo
            #infectado se recuperou. Logo, ...
            
            # ... não há alteração nos suscetíveis
            s.append(s[j])                                   
            # e há alteração nos infectados e recuperados.
            i.append(i[j] - 1)
            # e há alteração nos infectados e recuperados.
            r.append(r[j] + 1)
        # Faremos isso novamente para o próximo índice.
        j = j + 1                                            
    
    return t, s, i, r

def sir_write(filename):
    pass

def sir_read(filename):
    pass


def main():
    # Tamanho da população
    N = 1e9

    # Populacao infectada inicial
    I_0 = 1

    # Populacao recuperada inicial
    r_0 = 0

    # Decimagem. O dado final tera a ordem de $N/dec$ elementos.
    # Uma boa performance de memoria deve ter $N/dec ~ [1e3, 1e5]$
    dec = 1e6

    # Fator de contagio
    c = 2.81

    # Probabilidade de contagio
    p = 0.9

    # Fator de recuperacao
    gamma = 1/14

    # Tempo de simulação em dias
    T = 65

    # Constantes de apoio
    beta = c * p
    R_0 = beta/gamma

    print(f"Rodando simulação com parametros:")
    print(f"N: {N}, I_0: {I_0}, r_0: {r_0}, beta: {beta}, gamma: {gamma}")
    print(f"T: {T}, dec: {dec}, seed: {seed}")
    exp_sto = sir_sto_simul(N, I_0, r_0, beta, gamma, T, dec, seed=0)
    exp_det = sir_det_simul(N, I_0, r_0, beta, gamma, T, exp_sto.shape[0])

    fig, ax = plt.subplots(1,1, dpi=150)
    sir_plot(ax, exp_sto, label=['S(t) sto',' I(t) sto', 'S(t) sto']);
    sir_plot(ax, exp_det, label=['S(t) det',' I(t) det', 'S(t) det'], dashed=True);
    ax.set_title(r'Comparison SIR Model (stochastic and deterministic) $\beta: {}$'.format(beta), pad=20)
    fig.tight_layout()
    fig.savefig('imgs/sir.png', dpi=300)


if __name__ == '__main__':
    main()
