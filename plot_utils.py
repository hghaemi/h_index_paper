import networkx as nx
import numpy as np
from utils import *


def get_all_indices(G : nx.Graph, P):
        
    indices = dict()
    H, frac = calculate_H(G=G)
    indices['H'], indices['frac'] = H, frac
   
    m2 = specteral_moment_calculator(matrix=P, l=2)
    m3 = specteral_moment_calculator(matrix=P, l=3)
    m4 = specteral_moment_calculator(matrix=P, l=4)
    indices['m2'], indices['m3'], indices['m4'] = m2, m3, m4
    
    
    W1, W2, Q = synchronizability_calculator(G)
    indices['W1'], indices['W2'], indices['Q'] = W1, W2, Q
    try:
        R = normalized_graph_resistance(G)
    except Exception as e:
        print('Graph is not connected (setting R to 0).')
        R = 0
        
    sum_R_v = total_vertex_resistance(G)
    indices['R'], indices['sum_R_v'] = R, sum_R_v
    
    Estrada, Estrada_P = get_Estrada_indices(G, P)
    indices['Estrada'], indices['Estrada_P'] = Estrada, Estrada_P
    
    Energy = graph_energy_calculator(nx.to_numpy_array(G))
    indices['Energy'] = Energy
    
    n = G.number_of_nodes()
    indices['n'] = n
    
    #print(f'Type indices:{type(indices)}\nindices:{indices}')
    return indices
    

def append_values_plot_1(G:nx.Graph, P, Graph_name:str, values_dict, excluded_metrics=None):
    """
    Calls get_plot1_indices() and append values to the dictionaries.
    Arguments:
        G (nx.Graph) - Graph.
        P - stochastix matrix of the Graph.
        Graph_name (string) - Graph names in the values_dict keys.
        values_dict (dict) - Dictionary containing values for plot 1
        n - Num of iteration
        excluded_metrics
    Returns:
        values_dict (dict) - Dictionary after appending values for plot 1. 
    """
    indices = get_all_indices(G=G, P=P)
    if excluded_metrics is None:
        excluded_metrics = ['frac','m3','m4','W1', 'W2']
        
    for metric in indices.keys():
        if metric not in excluded_metrics:
            if Graph_name not in values_dict:
                values_dict[Graph_name] = {}
    
            if metric not in values_dict[Graph_name]:
                values_dict[Graph_name][metric] = []
    
    values_dict[Graph_name]['H'].append(indices['H'])
    values_dict[Graph_name]['m2'].append(indices['m2'])
    values_dict[Graph_name]['Q'].append(indices['Q'])
    values_dict[Graph_name]['Estrada'].append(indices['Estrada'])
    values_dict[Graph_name]['Estrada_P'].append(indices['Estrada_P'])
    values_dict[Graph_name]['Energy'].append(indices['Energy'])
    values_dict[Graph_name]['R'].append(indices['R'])
    values_dict[Graph_name]['sum_R_v'].append(indices['sum_R_v'])
    values_dict[Graph_name]['n'].append(indices['n'])
    
    return values_dict


def calculate_AUC_fraction(fraction_values : list, n_values : list) -> list:
    """
    Calculate and return a list containing AUC of fraction using np.trapz().
    frac_AUC[i] is the AUC of fraction between(0, i+1).
    --------------------------------------------------------
    Arguments:
        fraction_values (list) - A list containing fraction values of a graph for n in n_values.
        n_values (list) -  A list containing n (#Nodes) values.
    
    Returns:
        frac_AUC (list) - A list containing AUC of fraction Vs delta n.
    """
    frac_AUC = list()
    
    for i in range(1, len(fraction_values)):
        #print(f'Fraction values:{fraction_values[:i]}, n_values:{n_values[:i]}')
        AUC = np.trapz(y=fraction_values[:i+1], x=n_values[:i+1])
        frac_AUC.append(AUC)
        
    #print(f'Len frac_AUC:{len(frac_AUC)}, Len fraction_values:{len(fraction_values)}')
    
    return frac_AUC


def calculate_AUC_H(H_values:list, n_values:list) -> list:
    """
    Calculate and return a list containing AUC of H using np.trapz().
    H_AUC[i] is the AUC of H between(0, i+1).
    --------------------------------------------------------
    Arguments:
        H_values (list) - A list containing fraction values of a graph for n in n_values.
        n_values (list) -  A list containing n (#Nodes) values.
    
    Returns:
        H_AUC (list) - A list containing AUC of H Vs delta n.
    """
    H_AUC = list()
    
    for i in range(1, len(H_values)):
        #print(f'Fraction values:{fraction_values[:i]}, n_values:{n_values[:i]}')
        AUC = np.trapz(y=H_values[:i+1], x=n_values[:i+1])
        H_AUC.append(AUC)
        
    #print(f'Len frac_AUC:{len(frac_AUC)}, Len fraction_values:{len(fraction_values)}')
    
    return H_AUC