import random
import os
import itertools
import math 

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from igraph import Graph
# from Entropy_Single_Layer import *


def get_adjacency_matrices(layers):
    adj_matrices = list()
    for layer in layers:
        adj_matrix = nx.to_numpy_array(layer)
        adj_matrices.append(adj_matrix)
       
        
def simulate_random_attack(layers,dependencies,p_r, dependency_type):
    n = layers[0].number_of_nodes() #number of nodes
    L = len(layers) #Number of layers
    n_r  = int(n * p_r) # number of removed nodes
    removed_nodes = random.sample(list(layers[0].nodes()), n_r)
    
    if dependency_type == 'multiplex':
        
        for l in range(1,L+1):
            layers[l-1].remove_nodes_from(removed_nodes)
        
        return layers
    
    elif dependency_type == 'interdependent':    
        for node in removed_nodes:
            for l in range(1,L+1):
                dependent_nodes = dependencies[node][l]
                layers[l-1].remove_nodes_from(dependent_nodes) #remove dependent nodes
        
        layers[0].remove_nodes_from(removed_nodes) #Remove attacked nodes
    
        return layers
    else:
        raise Exception('Invalid dependency type')
        return None
            
    
    
def simulate_targeted_attack(layers, dependencies, p_r, metric):
    n = layers[0].number_of_nodes() #number of nodes
    L = len(layers) #Number of layers
    n_r  = int(n * p_r) # number of removed nodes
    G = layers[0] #remove nodes from the first layer
    
    if   metric == 'degree':
        
        degree_centrality = nx.degree_centrality(G)
        sorted_nodes  = sorted(degree_centrality, key=degree_centrality.get, reverse=True)
        removed_nodes = sorted_nodes[:n_r]

        for node in removed_nodes:
            for l in range(1,L+1):
                dependent_nodes = dependencies[node][l]
                layers[l-1].remove_nodes_from(dependent_nodes) #remove dependent nodes
                #layers is a list so layers[0] is the first layer
        
        layers[0].remove_nodes_from(removed_nodes) #Remove attacked nodes
        
        
    elif metric == 'closeness':
        
        closeness_centrality = nx.closeness_centrality(G)
        sorted_nodes  = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)
        removed_nodes = sorted_nodes[:n_r]
        
        for node in removed_nodes:
            for l in range(1,L+1):
                dependent_nodes = dependencies[node][l]
                layers[l-1].remove_nodes_from(dependent_nodes) #remove dependent nodes
        
        layers[0].remove_nodes_from(removed_nodes) #Remove attacked nodes
        
        
    elif metric == 'betweenness':
        
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_nodes  = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
        removed_nodes = sorted_nodes[:n_r]
        
        for node in removed_nodes:
            for l in range(1,L+1):
                dependent_nodes = dependencies[node][l]
                layers[l-1].remove_nodes_from(dependent_nodes) #remove dependent nodes
                #layers is a list so layers[0] is the first layer
        
        layers[0].remove_nodes_from(removed_nodes) #Remove attacked nodes
        
    return layers
    

def aggregate_adjacency_matrix(adjacency_matrices):
    agg_adj_matrix = np.zeros_like(adjacency_matrices[0], dtype=int)
    
    for adj_matrix in adjacency_matrices:
        
        agg_adj_matrix = np.logical_or(agg_adj_matrix, adj_matrix)
    
    agg_adj_matrix = agg_adj_matrix.astype(int)
    
    return agg_adj_matrix


def aggregate_graph(layers, adj_matrices):
    agg_matrix = aggregate_adjacency_matrix(adj_matrices)
    if type(layers[0]) == nx.DiGraph:
        agg_g = nx.from_numpy_array(agg_matrix, create_using=nx.DiGraph())
    else:
        agg_g = nx.from_numpy_array(agg_matrix)
        
    return agg_g
    

def intersection_adjacency_matrix(adjacency_matrices):

    intersection_adj_matrix = np.ones_like(adjacency_matrices[0])
    
    for adj_matrix in adjacency_matrices:
        intersection_adj_matrix = np.logical_and(intersection_adj_matrix, adj_matrix)
        
    intersection_adj_matrix = intersection_adj_matrix.astype(int)
    
    return intersection_adj_matrix


def intersection_graph(layers, adj_matrices): 
    inter_matrix = intersection_adjacency_matrix(adj_matrices)
    if type(layers[0]) == nx.DiGraph:
        inter_g = nx.from_numpy_array(inter_matrix, create_using=nx.DiGraph())
    else:
        inter_g = nx.from_numpy_array(inter_matrix)
        
    return inter_g
    

def create_interdependent_dependencies(num_layers, number_of_nodes,layers):

    dependencies = dict()
    degree_centrality_list = list()
    
    for layer in layers:
        degree_centrality = nx.degree_centrality(layer)
        sorted_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)
        degree_centrality_list.append(sorted_nodes)
    
    
    for i in range(number_of_nodes):
        current_node = degree_centrality_list[0][i]
        dependencies[current_node] = dict()
        
        for l in range(1, num_layers + 1):
            
            if l == 1:
                dependencies[current_node][l] = set()
            
            else:
                dependencies[current_node][l] = {degree_centrality_list[l-1][i]}
    
    return dependencies


def create_multiplex_dependencies(num_layers,number_of_nodes):
    dependencies = dict()

    for n in range(0, number_of_nodes):
        dependencies[n] = dict()
        for l in range(1, num_layers + 1):
            if l == 1:
                dependencies[n][l] = set()
            else:
                dependencies[n][l] = {n}
    return dependencies


def create_network_inputs():
    
    # Set the number of layers
    num_layers = int(input("Enter the number of layers in the network: "))

    # Initialize lists to store layer types and parameters
    layer_types = []
    layer_parameters = []

    # Set the number of nodes for Multiplex network
    n = int(input("Enter the number of nodes for the Network: "))

    # Get the type of network for each layer
    for i in range(num_layers):
        layer_type = input(f"Enter the type of network (ER, BA or SW) for layer {i+1}: ")
        layer_types.append(layer_type)

        if layer_type == 'ER':
            p = float(input("Enter the probability of edge creation (between 0 and 1): "))
            layer_parameters.append({'n': n, 'p': p})
        
        elif layer_type == 'BA':
            m = int(input("Enter the number of edges to attach from a new node (m): "))
            layer_parameters.append({'n': n, 'm': m})
        
        elif layer_type == 'SW':
            k = int(input("Enter the K value for SW network:"))
            p = float(input("Enter the probability of rewiring each node (p):"))
            layer_parameters.append({'n':n ,'k':k, 'p':p})
        else:
            raise Exception("Invalid layer type. Please choose 'ER','BA' or 'SW'.")

    network_type = input("Input Network type(multiplex or interdependent): ")
    
    return network_type, num_layers, layer_types, layer_parameters


def create_network():
    net_type, num_layers, layer_types, layer_parameters = create_network_inputs()
    
    # Create an empty list to store layers
    layers = []
    adj_matrices = []
    number_of_nodes = (layer_parameters[0])['n']
    # Generate layers based on specified types and parameters
    for i in range(num_layers):
        layer_type = layer_types[i]
        params = layer_parameters[i]

        if layer_type == 'ER':
            n = params['n']
            p = params['p']
            layer = nx.erdos_renyi_graph(n, p)
            adj_matrix = nx.to_numpy_array(layer)
            
        elif layer_type == 'BA':
            n = params['n']
            m = params['m']
            layer = nx.barabasi_albert_graph(n, m)
            adj_matrix = nx.to_numpy_array(layer)
            
        elif layer_type == 'SW': #Small-world graph
            n = params['n']
            k = params['k'] #Each node is joined with its `k` nearest neighbors in a ring topology.
            p = params['p']
            layer = nx.watts_strogatz_graph(n,k,p) #The probability of rewiring each edge
            adj_matrix = nx.to_numpy_array(layer)
        
        else:
            raise ValueError(f"Invalid layer type '{layer_type}'. Use 'ER' or 'BA'.")

        layers.append(layer)
        adj_matrices.append(adj_matrix)
    
    if net_type == 'multiplex':
        dependencies  = create_multiplex_dependencies(num_layers, number_of_nodes)
    elif net_type == 'interdependent':
        dependencies  = create_interdependent_dependencies(num_layers, number_of_nodes,layers)
    else:
        raise Exception("Invalid Network type!")
    
    print(f"Multiplex Network created with {num_layers} layers with {number_of_nodes} nodes.")

    return layers,adj_matrices,dependencies,net_type



def import_lazega():
    layers = list() # list containng diff layers of the network
    adj_matrices = list() # list containing adj matrix of each layer
    
    
    for _ in range(0,3):
        G = nx.DiGraph()
        G.add_nodes_from(range(0,71)) # note that the network is created with nodes from 0 to 71 rather than 1 to 72
        layers.append(G)
        
    
    with open('Datasets/Lazega/Lazega-Law-Firm_multiplex.edges', 'r') as file:
    # Read each line in the file
        for line in file:
            # Split the line by whitespace
            parts = line.split()
            # Extract layerID and nodeIDs
            layer_id = int(parts[0]) - 1
            node_1 = int(parts[1]) - 1
            node_2 = int(parts[2]) - 1
            # Append the extracted data to the lists
            layers[layer_id].add_edge(node_1, node_2)
    
    for layer in layers:
        adj_matrix = nx.to_numpy_array(layer)
        adj_matrices.append(adj_matrix)
    
    
    dependencies = create_multiplex_dependencies(3,71)
    
    print(layers[0])
    print(layers[1])
    print(layers[2])
    
    
    return layers, adj_matrices, dependencies
    

def import_aarhus():
    layers = list() # list containng diff layers of the network
    adj_matrices = list() # list containing adj matrix of each layer
    
    
    for _ in range(0,5):
        G = nx.Graph()
        G.add_nodes_from(range(0,61)) # note that the network is created with nodes from 0 to 60 rather than 1 to 61
        layers.append(G)
        
    
    with open('Datasets/Aarhus/CS-Aarhus_multiplex.edges', 'r') as file:
    # Read each line in the file
        for line in file:
            # Split the line by whitespace
            parts = line.split()
            # Extract layerID and nodeIDs
            layer_id = int(parts[0]) - 1
            node_1 = int(parts[1]) - 1
            node_2 = int(parts[2]) - 1
            # Append the extracted data to the lists
            layers[layer_id].add_edge(node_1, node_2)
    
    for layer in layers:
        adj_matrix = nx.to_numpy_array(layer)
        adj_matrices.append(adj_matrix)
    
    
    dependencies = create_multiplex_dependencies(5,61)
    print(layers[0])
    print(layers[1])
    print(layers[2])
    print(layers[3])
    print(layers[4])
    
    return layers, adj_matrices, dependencies


def import_kapferer():
    layers = list() # list containng diff layers of the network
    adj_matrices = list() # list containing adj matrix of each layer
    
    
    for _ in range(0,4):
        G = nx.DiGraph()
        G.add_nodes_from(range(0,39)) # note that the network is created with nodes from 0 to 38 rather than 1 to 39
        layers.append(G)
        
    
    with open('Datasets/Kapferer/Kapferer-Tailor-Shop_multiplex.edges', 'r') as file:
    # Read each line in the file
        for line in file:
            # Split the line by whitespace
            parts = line.split()
            # Extract layerID and nodeIDs
            layer_id = int(parts[0]) - 1
            node_1 = int(parts[1]) - 1
            node_2 = int(parts[2]) - 1
            # Append the extracted data to the lists
            layers[layer_id].add_edge(node_1, node_2)
    
    for layer in layers:
        adj_matrix = nx.to_numpy_array(layer)
        adj_matrices.append(adj_matrix)
    
    
    dependencies = create_multiplex_dependencies(4,39)
    print(layers[0])
    print(layers[1])
    print(layers[2])
    print(layers[3])
    
    
    return layers, adj_matrices, dependencies
    

def import_celegans():
    layers = list() # list containng diff layers of the network
    adj_matrices = list() # list containing adj matrix of each layer
    
    
    for _ in range(0,3):
        # In the readme file CElegans is stated as a undirected graph, however in entropy paper it is considered directed
        G = nx.DiGraph()
        G.add_nodes_from(range(0,279)) # note that the network is created with nodes from 0 to 278 rather than 1 to 279
        layers.append(G)
        
        #adj_matrix = np.zeros(279,279)
        #adj_matrixe
        
    
    with open('Datasets/CElegans/celegans_connectome_multiplex.edges', 'r') as file:
    # Read each line in the file
        i=0
        for line in file:
            # Split the line by whitespace
            parts = line.split()
            # Extract layerID and nodeIDs
            layer_id = int(parts[0]) - 1
            node_1 = int(parts[1]) - 1
            node_2 = int(parts[2]) - 1
            # Append the extracted data to the lists
            layers[layer_id].add_edge(node_1, node_2)
            
            # Make the adj_matrices
            adj_matrices 
    
    for layer in layers:
        adj_matrix = nx.to_numpy_array(layer)
        adj_matrices.append(adj_matrix)
    
    
    dependencies = create_multiplex_dependencies(3,279)
    print(layers[0])
    print(layers[1])
    print(layers[2])
    print(np.sum(adj_matrices[0])/2)
    print(np.sum(adj_matrices[1])/2)
    print(np.sum(adj_matrices[2])/2)
    
    return layers, adj_matrices, dependencies
    
    
def calculate_X(intersection_matrix):    
    X = [] 
    rows_sums = np.sum(intersection_matrix, axis=1)
    X = [(np.where(rows_sums >= 1)[0])]
    X = X[0]
    cardinality_of_X = len(X)

    return X, cardinality_of_X


def calculate_k(X, agg_matrix):
    k = [] # the list containig k values (degree of nodes in X list in agg_network)
    k = np.sum(agg_matrix[X],axis=1)
    k_sum = np.sum(k)

    return k, k_sum


def degree_matrix_calculator(G):
    degrees = dict(G.degree())
    D = np.diag([degrees[node] for node in G.nodes()])
    return D
    

def stochastic_matrix_calculator(G, A=None):
    if A is None:
        A = nx.to_numpy_array(G)
    else:
        pass
    
    degrees = dict(G.degree())
    # These two line added for handeling division by zero
    inv_degrees = [1 / degrees[node] if degrees[node] != 0 else 0 for node in G.nodes()]
    D_inv = np.diag(inv_degrees)

    #D_inv = np.diag(1 / np.array([degrees[node] for node in G.nodes()]))
    
    P = D_inv @ A
    #P = A @ D_inv 
    
    return P

def specteral_moment_calculator(matrix,  l: int=2):
    eigenvalues = np.linalg.eigvals(matrix)
    real_eigenvalues = np.real(eigenvalues)
    return np.mean(np.power(real_eigenvalues, l))


def graph_energy_calculator(matrix):    
    eigenvalues = np.linalg.eigvals(matrix)
    real_eigenvalues = np.real(eigenvalues)
    energy = np.sum(np.abs(eigenvalues))
    return energy


def laplacian_energy_calculator(G, use_P=False, P=None):
    m = G.number_of_edges()
    n = G.number_of_nodes()
    if use_P:
        if P is None:
            P = stochastic_matrix_calculator(G)
        else:
            pass
        eigenvalues = np.linalg.eigvals(P)
        real_eigenvalues = np.real(eigenvalues)
        laplacian_eigenvalues = 1 - real_eigenvalues
        laplacian_energy = np.sum(np.abs(laplacian_eigenvalues - (2*m/n)))
        return laplacian_energy
    else:
        L = nx.laplacian_matrix(G).toarray() # Laplacian graph
        laplacian_eigvals = np.real(np.linalg.eigvals(L))
        laplacian_energy = np.sum(np.abs(laplacian_eigvals - (2*m/n)))
        return laplacian_energy


def get_multiplex_entropies(layers, adj_matrices, epsilon=0.5, q=0.5):
    results_dict = {'layers':[],
                   'E_D':[],
                   'fractional_E_D':[],
                   'E_M':[]}
    
    # Calculate all the combination of layers
    layers_nums = []
    for r in range(2, len(layers) + 1): # Combinations with atleast 2 elements
        layers_nums.extend(itertools.combinations(range(len(layers)), r))

    #print(layers_nums) # index of the layers we want our model to calculate the entropy of
    # Layers are 0 to n-1
    
    for layers_num in layers_nums:
        

        # Choose the desired matrices 
        #desired_layers = [layers[layers_num[i]] for i in range(len(layers_num))] # List of Layers we want to calculate the enrtopy of
        desired_matrices = [adj_matrices[layers_num[i]] for i in range(len(layers_num))] # List of matrices of graphs we want to calculate the entropy of 

        # Create the intersection and agg networks
        intersection_matrix = intersection_adjacency_matrix(desired_matrices)
        agg_matrix = aggregate_adjacency_matrix(desired_matrices)

        # Create agg net and intersection network
        if type(layers[0]) == nx.DiGraph:
            intersection_net = nx.from_numpy_array(intersection_matrix, create_using=nx.DiGraph())
            agg_net = nx.from_numpy_array(agg_matrix, create_using=nx.DiGraph())
        else:
            intersection_net = nx.from_numpy_array(intersection_matrix) 
            agg_net = nx.from_numpy_array(agg_matrix)


        # Calculate X list, the list containing nodes with Multi-link
        X, cardinality_of_X = calculate_X(intersection_matrix)

        # Calculate k (list) and k_sum (value), given X (degrees of nodes in X in the agg network)
        k, k_sum = calculate_k(X, agg_matrix)

        #print(X.shape)#, k_sum)

        # Calculate the power of Epsilon
        overlapping_edge_set = intersection_net.number_of_edges() #np.sum(intersection_matrix) // 2 # Number of edges in intersection network
        connected_edges_set  = agg_net.number_of_edges() #np.sum(agg_matrix) // 2 # Number of edges in agg network
        power_of_epsilon = overlapping_edge_set / connected_edges_set
        #print(overlapping_edge_set)#, connected_edges_set)

        # Caculate the epsilon part
        epsilon_part = pow(epsilon, power_of_epsilon)

        # Calculate the denominator
        denominator = pow(2, cardinality_of_X) - 1

        #E_M = 0 #
        # Define E_M_values & p , for i node in X list
        E_M_values = list() # List of E_M_values for each node in X
        fractional_E_D_values = list() # List of fractional deng entropies for each node in X
        E_D_values = list() # List of Deng entropies
        p = [] # Values of p for n in X

        for i in range(len(X)):
            p.append(k[i]/k_sum)
            numerator = p[i]

            fraction = numerator / denominator

            E_M_value = math.log2(fraction * epsilon_part)
            E_M_values.append(E_M_value)

            E_D_log_part = math.log2(fraction)


            fractional_E_D_value = np.power(np.negative(E_D_log_part), q)
            fractional_E_D_values.append(fractional_E_D_value)

            E_D_value = E_D_log_part
            E_D_values.append(E_D_value)

        E_M = - np.sum(np.multiply(p, E_M_values))

        fractional_E_D = np.sum(np.multiply(p, fractional_E_D_values))

        E_D = - np.sum(np.multiply(p, E_D_values))
        
        results_dict['layers'].append(layers_num)
        results_dict['E_D'].append(E_D)
        results_dict['fractional_E_D'].append(fractional_E_D)
        results_dict['E_M'].append(E_M)
         
        layers_num = tuple(x + 1 for x in layers_num)
        print(f"Layers: {layers_num} | Overlapping edges: {overlapping_edge_set} | |X|: {X.shape[0]} | E_M value: {E_M} | E_D: {E_D} |fractional E_D:{fractional_E_D} \n ")
    
    return results_dict

  
def calculate_H_1(G):
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    degrees = [G.degree(n) for n in G.nodes()]
    degrees_geo_mean = np.exp(np.mean(np.log(degrees)))
    avg_degree = 2*m/n
    
    A = nx.to_numpy_array(G)
    P = stochastic_matrix_calculator(G, A)
    m_2 = specteral_moment_calculator(matrix=P, l=2)
    inside_value = 0
    edges = G.edges()

    for edge in edges:
        i, j = edge

        inside_numerator = 1

        inside_denominator = G.degree(i) * G.degree(j)
        inside_value += inside_numerator / inside_denominator
    
    
        inside_value /= m 
    
        inv_expected_degree = np.divide(inside_value, m_2)    
        fraction = np.power((degrees_geo_mean * inv_expected_degree), n)
        fraction = np.floor(fraction * 1e3) / 1e3
        H = 1 - fraction
        
    return H, fraction


def calculate_H(G):

    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    degrees = [G.degree(n) for n in G.nodes()]
    degrees_geo_mean = np.exp(np.mean(np.log(degrees)))
    
    A = nx.to_numpy_array(G)
    P = stochastic_matrix_calculator(G, A)
    m_2 = specteral_moment_calculator(matrix=P, l=2)
    inside_value = 0
    edges = G.edges()

    for edge in edges:
        i, j = edge

        inside_numerator = 1

        inside_denominator = G.degree(i) * G.degree(j)
        inside_value += inside_numerator / inside_denominator


    inside_value /= m 

    inv_expected_degree = np.divide(inside_value, m_2)    
    fraction = degrees_geo_mean * inv_expected_degree
    fraction = np.floor(fraction * 1e3) / 1e3
    H = 1 - fraction
    
    return H, fraction


def calculate_H_geometric(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    degrees = [G.degree(n) for n in G.nodes()]
    degrees_geo_mean = np.exp(np.mean(np.log(degrees)))
    avg_degree = 2*m/n
    
    H = np.divide((avg_degree - degrees_geo_mean), avg_degree, dtype=np.float32)
    H = np.floor(H * 1e3) / 1e3
    fraction = 1-H      

    return H, fraction


def H_calculator_test(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    A = nx.to_numpy_array(G)
    #A=nx.Adjacency_matrix(G).todense()

    P = stochastic_matrix_calculator(G, A)
    m_2 = specteral_moment_calculator(matrix=P, l=2)
    inside_value = 0
    degrees = [G.degree(n) for n in G.nodes()]
    degrees_geo_mean = np.exp(np.mean(np.log(degrees)))    
    print(f'Geo mean of degrees:: {degrees_geo_mean} \n')
    #pi_of_degrees = np.prod(degrees, dtype=np.int64) # Due to overflow set to np.int64
    edges = G.edges()
        
    
    for edge in edges:
        i, j = edge
        inside_numerator = 1

        inside_denominator = G.degree(i) * G.degree(j)
        inside_value += inside_numerator / inside_denominator
        
    #inside_value /= 2 * m 
    
    inside_value /= m 

    inv_expected_degree = np.divide(inside_value, m_2)
    print(f"\nInverse Exopected Degree:: {inv_expected_degree}")
    
    # Value for testing
    fraction = degrees_geo_mean * inv_expected_degree
    print(f'degrees_geo_mean / E(Degree) : {degrees_geo_mean * inv_expected_degree }')
    fraction_power = np.round(np.power(fraction, n, dtype=np.float64),6)
    
    #expected_degree_power = np.power(expected_degree,n, dtype=np.float64)
     
    H = 1 - fraction_power    
    return H , fraction_power


def energy_approximation_calculator(G, A=None, K:int=20, use_P=False, P=None):
    if use_P is False:
        if (A is None):
            matrix = nx.to_numpy_array(G)
        else:
            matrix = A

    else:
        if P is None:
            matrix = stochastic_matrix_calculator(G)
        else:
            matrix = P
    
    eigenvalues = np.real(np.linalg.eigvals(matrix))
    lambda_1 = (np.sort(eigenvalues)[::-1])[0]
    eye_matrix = np.eye(matrix.shape[0])
    sigma_value = 0 
    
    for k in range(K):
        inside_trace_numerator = np.linalg.matrix_power(matrix, 2)
        inside_trace_denominator = pow(lambda_1, 2)
        inside_trace_fraction = inside_trace_numerator / inside_trace_denominator
        inside_trace_value = np.linalg.matrix_power(inside_trace_fraction - eye_matrix, k)
        trace_value = np.trace(inside_trace_value)
        
        numerator = pow(-1, (k+1))
        denominator = pow(2, (2*k)) * (2*k - 1)
        fraction = numerator / denominator
        coefficient = math.comb(2 * k, k)
        
        sigma_value += coefficient * fraction * trace_value
        
    energy_approx= lambda_1 * sigma_value
    
    return energy_approx        
        
    
def synchronizability_calculator(g, use_laplacian=True, for_real_networks=False):
    if use_laplacian:
        if for_real_networks is False:
            eig_lap = list(nx.laplacian_spectrum(g))

            W1 = (eig_lap[-1] - eig_lap[0]) / (eig_lap[-1] - eig_lap[1])
            W2 = (eig_lap[-1] - eig_lap[1]) / (eig_lap[1] - eig_lap[0])
            Q = eig_lap[-1]/ eig_lap[1]
        
            return W1, W2, Q
        
        else:
            L = nx.laplacian_matrix(g).astype(float)
            largest_eigval, _ = eigsh(L, k=1, which='LA')
            smallest_eigvals, _ = eigsh(L,k=2, which='SA')
            smallest_eigval, second_smallest_eigval = smallest_eigvals[0], smallest_eigvals[1]
            
            W1 = (largest_eigval - smallest_eigval) / (largest_eigval - second_smallest_eigval)
            W2 = (largest_eigval - second_smallest_eigval) / (second_smallest_eigval - smallest_eigval)
            Q  = float(largest_eigval/second_smallest_eigval)
            
            return W1, W2, Q
    
    else:
        eig_adj = list(nx.adjacency_spectrum(g).real)
        eig_adj1 = list(reversed(sorted(eig_adj)))
        W1 = (eig_adj1[1] - eig_adj1[-1]) / (eig_adj1[0] - eig_adj1[1])
        W2 = (eig_adj1[0] - eig_adj1[-1]) / (eig_adj1[1] - eig_adj1[-1])
        
        return W1, W2


def est_moment(graph, number_of_walks, number_of_steps):
    node_list = list(nx.nodes(graph))
    moments = np.zeros(number_of_steps);
    for i in range(0, number_of_walks):
        #print(f"Walk number:{i}")
        node = random.choice(node_list)
        #print(f'Selected Node: {node}')
        w = node
        for step in range(0, number_of_steps):
            #print(f'step = {step}')
            nbr_list = list(nx.all_neighbors(graph, w))
            #print(f"Neighbors of node {w} = {nbr_list}")
            if len(nbr_list) == 0:
                # W is an isolated nodes
                break
            w = random.choice(nbr_list)
            #print(f"New Node selected:{w}")
            if w==node:
                #print(f"node:{node} equals w:{w}")
                moments[step] = moments[step] + 1;
                #print(f"New value of moments[{step}]={moments[step]}")
    #print(f"Moments before average:{moments}")                    
    moments = moments/number_of_walks; 
    #print(f'Moments after average:{moments}')
    return moments


def calculate_estrada_index_P(G : nx.Graph,P=None, k=100):
    estrada = 0
    estrada_trace = 0
    n = G.number_of_nodes()
    if P is None:
        P = stochastic_matrix_calculator(G)
        
    for i in range(2, k):
            #trace_part = np.trace(np.linalg.matrix_power(P,i))
            #estrada_trace += np.divide(trace_part, factorial)
            
            factorial = math.factorial(i)
            m = specteral_moment_calculator(P, i)
            estrada+=np.divide(m, factorial)
        
    estrada = n * estrada
    
    #return estrada, estrada_trace
    return estrada

def get_Estrada_indices(G, P):
    estrada = nx.estrada_index(G)
    estrada_P = calculate_estrada_index_P(G,P)
    return estrada, estrada_P


def graph_resistance_calculator(G : nx.Graph, use_laplacian : bool=False , eig_laplacian : list=None):
    if use_laplacian:
        
        if eig_laplacian is None:
            eig_lap = np.real((nx.laplacian_spectrum(G)))
    
        n = G.number_of_nodes()
    
        eig_lap = eig_lap[1:] # Delete the lowest eigenvalue (it equals zero)
        inv_eig_lap = [1 / value for value in eig_lap]
        R_G = n * np.sum(inv_eig_lap)

        return R_G
    
    else:
        R = nx.resistance_distance(G)
        R_G = 0
        nodes = G.nodes()
        for i in nodes:
            for j in nodes:
                R_G+=R[i][j]
        R_G /= 2 
        
        return R_G


def normalized_graph_resistance(G, R_G=None):
    n = G.number_of_nodes()
    if R_G is None:
        R_G = graph_resistance_calculator(G)
    
    normalized_R_G = R_G / (n-1)
    
    return normalized_R_G
    
    
def calculate_customize_degree_betweenness(G : nx.Graph, v):
    sigma_st_v = 0
    nodes = list(G.nodes)
    
    for s in nodes:
        for t in nodes:
            if s != t and s != v and t != v:
                all_shortest_paths = list(nx.all_shortest_paths(G, source=s, target=t))
                paths_through_v = [path for path in all_shortest_paths if v in path]
                sigma_st_v += len(paths_through_v)
    
    return sigma_st_v
    
    
def vertex_resistance_calculator(G, v:int, G_betweenness=None):
    #B_v = calculate_customize_degree_betweenness(G,v)
    if G_betweenness is None:
        G_betweenness = nx.betweenness_centrality(G)
        
    B_v = G_betweenness[v]
    d_v = G.degree(v)
    R_v = float(B_v/d_v)
    return R_v

def total_vertex_resistance(G : nx.Graph):
    nodes = list(G.nodes())
    G_betweenness =  nx.betweenness_centrality(G)

    sum_R_v = 0
    
    for node in nodes:
        sum_R_v += vertex_resistance_calculator(G=G, v=node, G_betweenness=G_betweenness)
    
    return sum_R_v
    
    
def calculate_hitting_time(G : nx.Graph, L : np.ndarray=None):
    if L is not None:
        lap_eigvals = np.real(np.linalg.eigvals(L))
        
    else:
        lap_eigvals = np.real(nx.laplacian_spectrum(G))
        
    m = G.number_of_edges()
    n = G.number_of_nodes()
    
    sorted_lap_eigvals = np.sort(lap_eigvals)[1:] # Exclude the Mio_1 which is close to zero
    inv_lap_eigvals = 1 / sorted_lap_eigvals
    sigma_term = np.sum(inv_lap_eigvals)
    multiplier = (2*m) / (n-1)
    hitting_time = multiplier * sigma_term
    
    return hitting_time


def calculate_mixing_time(G : nx.Graph, P=None, A=None):
    if P is None:
        P = stochastic_matrix_calculator(G, A)
    
    eig_vals = np.linalg.eigvals(P)
    sorted_eig_vals = np.sort(eig_vals)[::-1] # Sorted Eigenvalues of P in descending order
    #lambda_2 = np.abs(sorted_eig_vals[1]) # Absolute value of the second biggest Eigenvalue
    lambda_2 = sorted_eig_vals[1]
    T_m = 1 / (1-lambda_2)
    return T_m


def create_agave_graph(n, steps=1):
    G = nx.Graph()
    G.add_nodes_from([x for x in range(n)])
    if steps < 0:
        steps = 0
    elif steps >= n:
        steps = n-1
    for i in range(steps+1):
        for j in range(i+1, n):
            G.add_edge(i,j)
    return G


def create_scale_free_graph(n, gamma):
    for i in range(0,20):
        seq = list(np.round(nx.utils.powerlaw_sequence(n, gamma))) 
        for i in range(0, len(seq)):
            if (seq[i] % 2) != 0:
                seq[i] = int(seq[i] + 1)
            else:
                seq[i] = int(seq[i])
        g = nx.configuration_model(seq)
        g.remove_edges_from(nx.selfloop_edges(g))
        if nx.is_connected(g) == True:
            break
    return g


def create_cayley_tree(coord, depth):
    if depth == 0:
        return Graph(1)
    if depth == 1:
        return Graph.Tree(4, 3)
    
    d = coord - 1
    n1 = int((d ** (depth + 1) - 1) / (d - 1))
    n2 = int((d ** depth - 1) / (d - 1))
    
    tree1 = Graph.Tree(n1, d)
    tree2 = Graph.Tree(n2, d)
    
    combined_tree = tree1 + tree2
    combined_tree.add_edge(0, n1)
    G = nx.Graph([(e.source, e.target) for e in combined_tree.es])
    A = nx.adjacency_matrix(G)
    g = nx.from_numpy_array(A.todense())
    return g


def read_real_networks(folder_path):
    graph_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"): 
            file_path = os.path.join(folder_path, file_name)

            #graph = nx.read_edgelist(file_path, delimiter=',', nodetype=int)
            #graph = nx.read_edgelist(file_path, nodetype=int)
            graph = nx.read_edgelist(file_path)

            graph_name = os.path.splitext(file_name)[0]
            graph_dict[graph_name] = graph
            if not nx.is_connected(graph):
                print(f'{graph_name} is not connected!')

    return graph_dict


def create_supernova(clique, leaf):

    g = nx.Graph()
    g = nx.complete_graph(clique)
    
    new_nodes = range(clique, clique + leaf)
    g.add_nodes_from(new_nodes)
    
    for new_node in new_nodes:
        for clique_node in range(clique):
            g.add_edge(new_node, clique_node)
    return g