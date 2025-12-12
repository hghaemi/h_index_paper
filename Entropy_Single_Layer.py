from __future__ import division
import networkx as nx
import numpy as np
import math


def make_graph(fileName):
    gr = nx.read_edgelist(fileName,create_using=nx.Graph())
    return gr


def adjacent_matrix(g):
    return nx.adjacency_matrix(g).todense()


def laplacian_matrix(g):
    return nx.laplacian_matrix(g).todense()


def normalized_laplacian_matrix(g):
    return nx.normalized_laplacian_matrix(g).todense()
    

def signless_laplacian_matrix(g):
    degree_view_list = g.degree()
    degree_list = []
    for i in range(0,len(degree_view_list)):

        degree_list.append(degree_view_list[i])

    degree_list = np.array(degree_list)

    d = np.diag(degree_list)
    a = nx.adjacency_matrix(g).todense()
    return d + a


def normalized_signless_laplacian_matrix(g):
    q = signless_laplacian_matrix(g)

    degree_view_list = g.degree()

    degree_list = []
    for i in range(0, len(degree_view_list)):
        degree_list.append(degree_view_list[i])

    degree_list = np.array(degree_list)

    d = np.diag(degree_list)

    d_1_2 = np.sqrt(d)
    return np.dot(np.dot(d_1_2, q), d_1_2)


def incidence_matrix(g):
    return nx.incidence_matrix(g).todense()


def randic_adjacent_matrix(G):
    degree_view_list = G.degree()
    degree_list = []
    for i in range(0, len(degree_view_list)):
        degree_list.append(degree_view_list[i])

        
    adj = adjacent_matrix(G)

    ram = np.zeros([G.number_of_nodes(),G.number_of_nodes()])

    for i in range(0, G.number_of_nodes()):
        for j in range(0, G.number_of_nodes()):
            if adj[i,j] == 1 and i!=j:
                ram[i][j] = 1/np.sqrt(degree_list[i]*degree_list[j])

    return ram


def randic_general_matrix(g, beta):
    n = g.number_of_nodes()
    r = np.zeros((n, n), dtype=np.float64)
    adj_matrix = adjacent_matrix(g)
    ii = 0
    jj = 0
    for i in list(g.nodes()):
        for j in list(g.nodes()):
            if adj_matrix.item((ii, jj)) == 1:
                r[ii, jj] = (math.pow(g.degree(i) * g.degree(j), beta))
            jj += 1
        jj = 0
        ii += 1
    return r


def randic_incident_matrix(G):

    degree_view_list = G.degree()
    degree_list = []
    for i in range(0, len(degree_view_list)):
        degree_list.append(degree_view_list[i])

    inc_matrix = incidence_matrix(G)

    rim = np.zeros([G.number_of_nodes(),G.number_of_edges()])

    for i in range(0, G.number_of_nodes()):
        for j in range(0, G.number_of_edges()):

            if inc_matrix[i,j] == 1:
                rim[i][j] = 1/np.sqrt(degree_list[i])

    return rim


def distance_matrix(g):
    g = nx.convert_node_labels_to_integers(g)
    nodes = g.nodes()
    n = g.number_of_nodes()
    d = np.zeros((n, n))
    for node in nodes:
        for node1 in nodes:
            try:
                di = nx.shortest_path_length(g, node, node1)
                d[node, node1] = di
            except:
                d[node, node1] = 0
    return d


def distance_sum_matrix(g):
    d = distance_matrix(g)
    return np.cumsum(d, 1)


def make_output(g):
    print("#adjacent_matrix: ",adjacent_matrix(g))
    print("#signless_laplacian_matrix: ",signless_laplacian_matrix(g))
    print("#normalized_signless_laplacian_matrix: ",normalized_signless_laplacian_matrix(g))
    print("#incidence_matrix: ",incidence_matrix(g))
    print("#distance_matrix: ",distance_matrix(g))
    print("#randic_adjacent_matrix: ",randic_adjacent_matrix(g))
    print("#randic_general_matrix: ",randic_general_matrix(g,-1))
    print("#randic_incident_matrix: ",randic_incident_matrix(g))
    print("#normalized_laplacian_matrix: ",normalized_laplacian_matrix(g))


def get_wiener_index(G,y):
    d = distance_matrix(G)
    return np.sum(d**y)/2


def get_hyper_wiener_index(G):
    return 0.5*(get_wiener_index(G,1)+get_wiener_index(G,2))


def get_eig_values(m):
    return np.linalg.eigvals(m)


def get_eig_values_abs(m):
    return np.abs(np.linalg.eigvals(m))


def get_energy(m):
    return sum(get_eig_values_abs(m))


def randic_index(g, beta):
    rand_index = 0
    edges = list(g.edges())
    for i in range(len(edges)):
        d1 = g.degree(edges[i][0])
        d2 = g.degree(edges[i][1])
        rand_index = rand_index + (math.pow(d1 * d2, beta))
    return rand_index


def get_first_zagreb_index(m):
    return sum([x ** 2 for x in get_eig_values_abs(m)])


def get_singular_values(m, to_pow):
    U, S, Vh = np.linalg.svd(m)
    return [x ** to_pow for x in S]


def get_randic_incidence_energy(m):
    sv = get_singular_values(m, 1)
    return sum(sv)


def entropy_adjacent(G, mat):
    return 1 - ((2 * nx.number_of_edges(G)) / (get_energy(mat) ** 2))


def renyi_entropy_adjacent(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / (1 - alpha)) * math.log10(m_star / (get_energy(mat) ** alpha))


def daroczy_entropy_adjacent(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (get_energy(mat) ** alpha)) - 1)


def entropy_signless_laplacian(G, mat):
    m = nx.number_of_edges(G)
    return 1 - ((1 / (4 * (m ** 2))) * (get_first_zagreb_index(mat) + 2 * m))


def renyi_entropy_signless_laplacian(G, mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    m = nx.number_of_edges(G)
    return (1 / (1 - alpha)) * math.log10(m_star / ((2 * m) ** alpha))


def daroczy_entropy_signless_laplacian(G, mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    m = nx.number_of_edges(G)
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / ((2 * m) ** alpha)) - 1)


def entropy_normalized_signless_laplacian(G):
    n = nx.number_of_nodes(G)
    return 1 - ((1 / (n ** 2)) * (n + (2 * randic_index(G, -1))))


def renyi_entropy_normalized_signless_laplacian(G, mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    n = nx.number_of_nodes(G)
    if m_star == 0:
        m_star = 0.001
    return (1 / (1 - alpha)) * math.log10(m_star / (n ** alpha))


def daroczy_entropy_normalized_signless_laplacian(G, mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    n = nx.number_of_nodes(G)
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (n ** alpha)) - 1)


def entropy_incidence(G, mat):
    incidence_energy = sum([math.sqrt(x) for x in get_singular_values(mat, 1)])
    m = nx.number_of_edges(G)
    return 1 - ((2 * m) / (incidence_energy ** 2))


def renyi_entropy_incidence(mat, alpha):
    incidence_energy = sum([math.sqrt(x) for x in get_singular_values(mat, 1)])
    m_star = sum([(math.sqrt(x)) ** alpha for x in get_singular_values(mat, 1)])
    return (1 / (1 - alpha)) * (math.log10(m_star / (incidence_energy ** alpha)))


def daroczy_entropy_incidence(mat, alpha):
    incidence_energy = sum([math.sqrt(x) for x in get_singular_values(mat, 1)])
    m_star = sum([(math.sqrt(x)) ** alpha for x in get_singular_values(mat, 1)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (incidence_energy ** alpha)) - 1)


def entropy_distance(G, mat):
    return 1 - ((4 / (get_energy(mat) ** 2)) * (2 * get_hyper_wiener_index(G) - get_wiener_index(G,1)))


def renyi_entropy_distance(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / (1 - alpha)) * math.log10(m_star / (get_energy(mat) ** alpha))


def daroczy_entropy_distance(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (get_energy(mat) ** alpha)) - 1)


def entropy_randic_adjacent(G, mat):
    return 1 - ((2 * randic_index(G, -1)) / (get_energy(mat) ** 2))


def renyi_entropy_randic_adjacent(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / (1 - alpha)) * math.log10(m_star / (get_energy(mat) ** alpha))


def daroczy_entropy_randic_adjacent(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (get_energy(mat) ** alpha)) - 1)


def entropy_randic_incidence(G, mat):
    return 1 - (randic_index(G, -1) / (get_randic_incidence_energy(mat) ** 2))


def renyi_entropy_randic_incidence(mat, alpha):
    m_star = sum([(math.sqrt(x) ** alpha) for x in get_singular_values(mat, 1)])
    return (1 / (1 - alpha)) * math.log10(m_star / (get_randic_incidence_energy(mat) ** alpha))


def daroczy_entropy_randic_incidence(mat, alpha):
    m_star = sum([(math.sqrt(x) ** alpha) for x in get_singular_values(mat, 1)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (get_randic_incidence_energy(mat) ** alpha)) - 1)


def entropy_randic_general(G, mat, beta):
    return 1 - ((2 * randic_index(G, 2 * beta)) / (get_energy(mat) ** 2))


def renyi_entropy_randic_general(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / (1 - alpha)) * math.log10(m_star / (get_energy(mat) ** alpha))


def daroczy_entropy_randic_general(mat, alpha):
    m_star = sum([x ** alpha for x in get_eig_values_abs(mat)])
    return (1 / ((2 ** (1 - alpha)) - 1)) * ((m_star / (get_energy(mat) ** alpha)) - 1)


def quantumEntropy(mat):

    eig_vals = get_eig_values(mat)
    sum=0
    for i in range(0,len(eig_vals)):
        sum=sum+((eig_vals[i]/len(eig_vals))*np.log(eig_vals[i]/len(eig_vals)))
    return -sum