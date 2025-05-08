import numpy as np
import networkx as nx

def simulate_network_diffusion(num_nodes, p, steps, initial_infected=None, random_seed=None):
    """
    Simulate simple network diffusion: x_{t+1} = A x_t
    where A is adjacency matrix of an Erdős-Rényi graph.
    Returns an array of shape (steps+1, num_nodes).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    G = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=random_seed)
    A = nx.to_numpy_array(G)
    if initial_infected is None:
        x_t = np.zeros(num_nodes)
        x_t[0] = 1
    else:
        x_t = np.array(initial_infected, dtype=float)
    spread = [x_t.copy()]
    for _ in range(steps):
        x_t = A.dot(x_t)
        spread.append(x_t.copy())
    return np.array(spread)
