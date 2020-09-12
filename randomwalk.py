#!/usr/bin/env python3

import json
import random
import numba
import numpy as np
import timeitplot as tp
import matplotlib.pyplot as plt


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def random_walk_plot(ax0, data):
    time, position, velocity, acceleration = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    ax1 = ax0.twinx()
    ax2 = ax0.twinx()
    
    ax2.spines['right'].set_position(('axes', 1.2))
    make_patch_spines_invisible(ax2)
    ax2.spines['right'].set_visible(True)
    
    line0, = ax0.plot(time, position, color='tab:blue', alpha=.8, label='Position')
    line1, = ax1.plot(time, velocity, color='tab:red', alpha=.8, label='Velocity')
    line2, = ax2.plot(time, acceleration, color='tab:green', alpha=.4, label='Acceleration')
    
    ax0.set_xlabel(r'Time ($s$)')
    ax0.set_ylabel(r'Position ($m$)')
    ax1.set_ylabel(r'Velocity ($ms^{-1}$)')
    ax2.set_ylabel(r'Acceleration ($ms^{-2}$)')
    
    #ax0.set_ylim(-abs(position).max(), abs(position).max())
    ax1.set_ylim(-abs(velocity).max()*1.1, abs(velocity).max()*1.1)
    ax2.set_ylim(-abs(acceleration).max()*1.1, abs(acceleration).max()*1.1)
    
    ax0.yaxis.label.set_color(line0.get_color())
    ax1.yaxis.label.set_color(line1.get_color())
    ax2.yaxis.label.set_color(line2.get_color())
    
    tkw = dict(size=4, width=1.5)
    ax0.tick_params(axis='x', **tkw)
    ax0.tick_params(axis='y', colors=line0.get_color(), **tkw)
    ax1.tick_params(axis='y', colors=line1.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=line2.get_color(), **tkw)
    
    lines = [line0, line1, line2]
    
    return ax0


def random_walk_naive(initial_position = 0, acceleration = 0, 
                      prob_increase=0.005, prob_decrease=0.005, 
                      max_distance=1e5, simul_time=1e3,
                      seed=None):
    """Emulates same behavior as random_walk_simul, but use append instead numpy and generators"""
    if seed is not None:
        random.seed(seed)
    T, X, V, A = [0], [initial_position], [0], [acceleration]
    t, x, v, a = T[-1], X[-1], V[-1], A[-1]

    while (t < simul_time) and (abs(x) < max_distance):
        god_wishes = random.random()
        if god_wishes <= prob_increase:
            # Increase acceleration
            a += .005
        elif god_wishes <= prob_increase+prob_decrease:
            # Reduce acceleration
            a -= .005

        # Lets avoid too much acceleration
        lower, upper = -0.2, 0.2
        a = lower if a < lower else upper if a > upper else a

        # Next iter
        dt = random.random()
        t += dt
        v += dt*a
        x += dt*v

        # Storing next simulation step
        T.append(t)
        X.append(x)
        V.append(v)
        A.append(a)

    return np.array((T, X, V, A)).transpose()


@numba.njit
def random_walk_naive_numba(initial_position = 0, acceleration = 0, 
                            prob_increase=0.005, prob_decrease=0.005, 
                            max_distance=1e5, simul_time=1e3,
                            seed=None):
    if seed is not None:
        random.seed(seed)
    T, X, V, A = [0], [initial_position], [0], [acceleration]
    t, x, v, a = T[-1], X[-1], V[-1], A[-1]

    while (t < simul_time) and (abs(x) < max_distance):
        rnd = random.random()
        if rnd <= prob_increase:
            # Increase acceleration
            a += .005
        elif rnd <= prob_increase+prob_decrease:
            # Reduce acceleration
            a -= .005

        # Lets avoid too much acceleration
        #lower, upper = -0.2, 0.2
        #a = lower if a < lower else upper if a > upper else a
        a = -0.2 if a < -0.2 else 0.2 if a > 0.2 else a

        # Next iter
        dt = random.random()
        t += dt
        v += dt*a
        x += dt*v

        # Storing next simulation step
        T.append(t)
        X.append(x)
        V.append(v)
        A.append(a)

    return np.array((T, X, V, A)).transpose()


@numba.njit
def random_walk_naive_numba2(initial_position=0, acceleration=0, 
                            prob_increase=0.005, prob_decrease=0.005, 
                            max_distance=1e5, simul_time=1e3,
                            seed=None):
    if seed is not None:
        random.seed(seed)
    
    data = [(0, initial_position, 0, acceleration)]
    t, x, v, a = data[-1]

    while (t < simul_time) and (abs(x) < max_distance):
        rnd = random.random()
        if rnd <= prob_increase:
            # Increase acceleration
            a += .005
        elif rnd <= prob_increase+prob_decrease:
            # Reduce acceleration
            a -= .005

        # Lets avoid too much acceleration
        a = -0.2 if a < -0.2 else 0.2 if a > 0.2 else a

        # Next iter
        dt = random.random()
        t += dt
        v += dt*a
        x += dt*v

        # Storing next simulation step
        data.append((t, x, v, a))

    return np.array(data)


@numba.njit
def random_walk(s_0, a_0, pa, pb, seed=None):
    """Initial position (often 0), acceleration, 0 < pa < pb < 1"""
    if seed is not None:
        random.seed(seed)
    # Time, x-position, Velocity, Acceleration
    t, x, v, a = 0, s_0, 0, a_0
    yield (t, x, v, a)
    
    while True:        
        # Roll the dices
        rnd = random.random()
        if rnd <= pa:
                # Increase acceleration
                a += .005
        elif rnd <= pa+pb:
                # Reduce acceleration
                a -= .005
                
        # Lets avoid too much acceleration
        #lower, upper = -0.2, 0.2
        a = -0.2 if a < -0.2 else 0.2 if a > 0.2 else a
        
        # How much time has passed, since last update?
        dt = random.random()
        v += dt*a
        x += dt*v
        t += dt
        
        yield (t, x, v, a)


def random_walk_simul_nnp(initial_position = 0, acceleration = 0, 
                      prob_increase=5e-3, prob_decrease=5e-3, 
                      max_distance=1e5, simul_time=1e3,
                      seed=None):
    
    rw = random_walk(initial_position, 
                     acceleration, 
                     prob_increase, 
                     prob_decrease, 
                     seed)
          
    # Runs the first iteraction
    data = [next(rw)]
        
    # While there is simulation time or not too far away
    while (data[-1][0] < simul_time) and (abs(data[-1][1]) < max_distance):
        data.append(next(rw))
        
    return np.array(data)


@numba.njit
def random_walk_simul_numba(initial_position = 0, acceleration = 0, 
                      prob_increase=5e-3, prob_decrease=5e-3, 
                      max_distance=1e5, simul_time=1e3,
                      seed=None):
    
    rw = random_walk(initial_position, 
                     acceleration, 
                     prob_increase, 
                     prob_decrease, 
                     seed)
          
    # Runs the first iteraction
    data = []
    data.append(next(rw))
        
    # While there is simulation time or not too far away
    while (data[-1][0] < simul_time) and (abs(data[-1][1]) < max_distance):
        data.append(next(rw))
        
    return np.array(data)


def random_walk_simul(initial_position = 0, acceleration = 0, 
                      prob_increase=0.005, prob_decrease=0.005, 
                      max_distance=1e5, simul_time=1e3,
                      seed=None):
    """Runs a random walk simulation given parameters
    
    Particle initial state (initial position and acceleration)
    State change probability (prob_increase, prob_decrease)
    Stop criteria (max_distance, simul_time)
    
    Returns a random_walk particle data
    """
    assert (0 < prob_increase+prob_decrease < 1), "Total probability should be in range [0, 1]"
    
    rw = random_walk(initial_position, 
                     acceleration, 
                     prob_increase, 
                     prob_decrease, 
                     seed)
    
    # Over estimated given by law of large numbers expected value of a
    # uniform distribution
    estimated_N = int(simul_time * 2.2)
    
    data = np.empty((estimated_N, 4))
    
    # Runs the first iteraction
    n = 0
    (t, x, v, a) = next(rw)
    data[n] = (t, x, v, a)
        
    # While there is simulation time or not too far away
    while (t < simul_time) and (np.abs(x) < max_distance):
        n += 1
        (t, x, v, a) = next(rw)
        data[n] = (t, x, v, a)
        
    return data[:n+1]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def write(data, filename):
    with open(filename, 'w') as fd:
        fd.write(json.dumps(data, cls=NpEncoder))


def read(filename):
    with open(filename, 'r') as fd:
        data = json.loads(fd.read())

    return data


def test():
    functions = [
        "random_walk_naive(max_distance={0})",
        "random_walk_naive_numba(max_distance={0})",
        "random_walk_naive_numba2(max_distance={0})",
        "random_walk_simul(max_distance={0})",
        "random_walk_simul_nnp(max_distance={0})",
        "random_walk_simul_numba(max_distance={0})",
    ]

    ranges = [np.arange(100,1000,100)]
    data = tp.timeit_compare(functions, ranges, setups='main', number=1000, print_conditions=True)

    return data


def test_run(filename='data/randomwalk_test.json'):
    data = test()
    write(data, filename) 
    print(f"Data written to {filename}")


def test_plot(filename='data/randomwalk_test.json'):
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "10"
    plt.rcParams["text.usetex"] = True

    data = read(filename)

    labels = [
        'naive',
        'naivenumba1',
        'naivenumba2',
        'simul',
        'simulnnp',
        'simulnumba'
    ]

    data_n = {}
    for key,n_key in zip(data.keys(), labels):
        data_n[n_key] = data[key]

    del(data)

    sizes = [[6, 4.24],
             [3, 2.125],
             [6, 3.375],
             [3, 1.6875]]
         
    for w, h in sizes:
        fig, ax = plt.subplots(figsize=(w,h),dpi=150)
        ax = tp.timeit_plot2D(data_n, ax, 'max distance (m)', 'computing methods')
        filename = f'imgs/random_walk_comparison_{int(w)}_{int(h)}_300.png'
        print(f"Saving {filename}")
        fig.savefig(filename, dpi=300)


def test_experiments():
    md = 1e5
    st = 1e3
    seed = 0

    experiments = []
    experiments.append(random_walk_naive(max_distance=md, simul_time=st, seed=seed))
    experiments.append(random_walk_naive_numba(max_distance=md, simul_time=st, seed=seed))
    experiments.append(random_walk_naive_numba2(max_distance=md, simul_time=st, seed=seed))
    experiments.append(random_walk_simul(max_distance=md, simul_time=st, seed=seed))
    experiments.append(random_walk_simul_nnp(max_distance=md, simul_time=st, seed=seed))
    experiments.append(random_walk_simul_numba(max_distance=md, simul_time=st, seed=seed))

    for i, e in enumerate(experiments):
        fig, ax = plt.subplots(dpi=150)
        random_walk_plot(ax, e);
        filename = f"imgs/randomwalk_exp_{i}.png"
        print(f"Saving experiment {filename}")
        plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    #test_run()
    test_plot()
    test_experiments()


