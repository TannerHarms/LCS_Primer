
#%%
import os, sys, copy, re
import numpy as np
import networkx as nx
from networkx.algorithms import community
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import Delaunay
from scipy.linalg import eig
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Callable, List, Union
from sklearn.cluster import SpectralClustering

class FlowNet:
    
    def __init__(self, 
                 trajectories: np.ndarray, interval: float = None, 
                 metric_function: Union[str, Callable] = 'kinematic_dissimilarity',
                 kNN: int = False,
                 delaunay = False,
                 ) -> None:
        
        self.trajectories = trajectories
        self.kNN = kNN
        
        # Size of the data
        self.n_trajectories = trajectories.shape[0]
        self.dim = trajectories.shape[2]
        self.timesteps = trajectories.shape[1]
        
        if interval < self.timesteps:
            self.interval = interval
        else:
            self.interval = self.timesteps
        
        if metric_function == "kinematic_dissimilarity":
            self.metric_function = kinematic_dissimilarity
        elif metric_function == "kinematic_similarity":
            self.metric_function = kinematic_similarity
        elif metric_function == "cumulative_distance":
            self.metric_function = cumulative_distance
        elif metric_function == "inverse_cumulative_distance":
            self.metric_function = inverse_cumulative_distance
        elif metric_function == "correlation_coefficient":            
            self.metric_function = correlation_coefficient
        else:
            self.metric_function = metric_function
        
        # Compute the graph
        if delaunay:
            self.graph_from_delaunay()
        else:
            # Compute Adjacency matrix
            self.compute_adjacency()
            
            # Turn into a graph
            self.G = nx.from_numpy_array(self.adjacency)
        
        # Compute the Laplacian
        
    def compute_adjacency(self):
        
        A = np.zeros((self.n_trajectories, self.n_trajectories))
        
        # behavior whether or not using nearest neighbors
        if not self.kNN:
            indices = list(range(self.n_trajectories))
        else:
            X = self.trajectories[:,0,:].squeeze()  # based off of initial conditions
            nbrs = NearestNeighbors(n_neighbors=self.kNN, algorithm='ball_tree').fit(X)
        
        T = self.trajectories
        for i in range(self.n_trajectories):
            
            if self.kNN:
                _, indices = nbrs.kneighbors([T[i,0,:].squeeze()])
                indices = indices.squeeze()
            
            c=0
            for j in indices:
                A[i,j] = self.metric_function(T[i,:,:], T[j,:,:])
                # if i != j:
                #     A[i,j] = A[i,j]/dist
                c+=1
        
        self.adjacency = A
        
    def graph_from_delaunay(self):
        
        # initial positions of the trajectories
        points = self.trajectories[:,0,:].squeeze()
        T = self.trajectories
        
        # compute Delaunay triangulation
        tri = Delaunay(points)
        
        plt.triplot(points[:,0], points[:,1], tri.simplices)
        plt.plot(points[:,0], points[:,1], 'o')

        # create networkx graph
        G = nx.Graph()

        # loop over triangles in triangulation, add each edge to graph
        for i in range(tri.nsimplex):
            # point1 = points[tri.simplices[i, 0]].reshape(-1,1)
            # point2 = points[tri.simplices[i, 1]].reshape(-1,1)
            # point3 = points[tri.simplices[i, 2]].reshape(-1,1)
            
            # the indices
            u = tri.simplices[i, 0]
            v = tri.simplices[i, 1]
            w = tri.simplices[i, 2]
            
            # compute the weigths
            weight1 = self.metric_function(T[u,:,:], T[v,:,:])
            weight2 = self.metric_function(T[u,:,:], T[w,:,:])
            weight3 = self.metric_function(T[v,:,:], T[w,:,:])
            
            # add edges with weights
            G.add_edge(u, v, weight=weight1)
            G.add_edge(u, w, weight=weight2)
            G.add_edge(v, w, weight=weight3)
        
        # store variables
        self.G = G
        self.adjacency = nx.adjacency_matrix(G).toarray()   
        
    def coherent_structure_coloring(self):
        
        # Compute the degree matrix D 
        D = np.diag(np.sum(self.adjacency, axis=1))
        self.degree_matrix = D
        
        A = self.adjacency.copy()
        if np.any(A == 0):
            A += 0.00001
            for i in range(np.shape(A)[1]):
                A[i,:] /= np.sum(A[i,:])
        
        # Compute the graph Laplacian
        L = D - A
        self.laplacian = L
        
        # Perform a generalized eigenvalue decomposition
        eigenvalues, eigenvectors = eig(L, D, right=True)
        
        r_evals, r_evecs = np.abs(eigenvalues), np.abs(eigenvectors)
        
        # sort them by decreasing order
        idx = r_evals.argsort()[::-1]
        r_evals = r_evals[idx]
        r_evecs = r_evecs[:,idx]
        
        # take the first eigenvector as the coherent structure coloring.
        self.CSC = r_evecs[:,0]
        
        return r_evals, r_evecs
        
        

# Adjacency functions
def kinematic_similarity(t_i: np.ndarray, t_j: np.ndarray):
    
    # Make sure that there are no self loops
    if np.array_equal(t_i, t_j):
        return 0
    
    distance = np.linalg.norm(t_i-t_j, axis=1)
    avg_distance = np.mean(distance)
    T = len(distance)
    
    return 1/(1/(avg_distance*np.sqrt(T))*np.sqrt(np.sum((avg_distance - distance)**2)))

def kinematic_dissimilarity(t_i: np.ndarray, t_j: np.ndarray):
    
    # Make sure that there are no self loops
    if np.array_equal(t_i, t_j):
        return 0
    
    distance = np.linalg.norm(t_i-t_j, axis=1)
    avg_distance = np.mean(distance)
    T = len(distance)
    
    return 1/(avg_distance*np.sqrt(T))*np.sqrt(np.sum((avg_distance - distance)**2))
    
    
def cumulative_distance(t_i: np.ndarray, t_j: np.ndarray):
    
    # Make sure that there are no self loops
    if np.array_equal(t_i, t_j):
        return 0
    
    distance = np.linalg.norm(t_i-t_j, axis=1)
    
    return np.sum(distance)


def inverse_cumulative_distance(t_i: np.ndarray, t_j: np.ndarray):
    if np.array_equal(t_i, t_j):
        return 0
    return 1/cumulative_distance(t_i, t_j)

def correlation_coefficient(t_i: np.ndarray, t_j: np.ndarray):
    
    # Make sure that there are no self loops
    if np.array_equal(t_i, t_j):
        return 0
    
    # get the norm of each position
    t_i_normed = np.linalg.norm(t_i, axis=1).squeeze() + 1e-15 * np.random.randn(len(t_i))
    t_j_normed = np.linalg.norm(t_j, axis=1).squeeze() + 1e-15 * np.random.randn(len(t_i))
    
    return np.corrcoef(t_i_normed, t_j_normed)[1][0]

def remove_edges_under_threshold(G, threshold):
    # Get a list of all edges with weights below the threshold
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
  
    # Remove these edges from the graph
    G.remove_edges_from(edges_to_remove)
    return G

def remove_nodes_under_threshold(G, threshold):
    # Get a list of all edges with weights below the threshold
    nodes_to_remove = [n for n in G.nodes() if G.degree(n, weight='weight') < threshold]
  
    # Remove these edges from the graph
    G.remove_nodes_from(nodes_to_remove)
    return G, nodes_to_remove

def identify_communities(G):
    communities = community.greedy_modularity_communities(G, weight="weight")
    # communities = community.asyn_lpa_communities(G, weight="weight")

    node_color = []
    for node in G:
        for i, com in enumerate(communities):
            if node in com:
                node_color.append(i)
                break

    # Draw the graph
    return node_color

# For testing...
if __name__ == "__main__":
    sys.path.append(r"D:\OneDrive\Research\PhD\Code\Tutorials\LCS_Primer\Python_Tutorial\Flows")
    from Flows import Flow
    
    # Initialize a flow object
    function_name = "BickleyJet"
    flow = Flow()

    ''' Now, we need to define the initial conditions for our integrations. We will define them on a grid for now.  '''
    # Specify the flow domain using dim[0] = x axis and dim[1] = y axis
    # domain = np.array([[0, 2],[0, 1]])

    # # Now, make vectors associated with each axis.
    # n_y = 25            # number of rows
    # dx = 1/n_y          # row spacing
    # x_vec = np.arange(domain[0,0],domain[0,1]+dx,dx)     # 50 columns
    # y_vec = np.arange(domain[1,0],domain[1,1]+dx,dx)     # 25 rows

    # # Then, make the mesh grid and flatten it to get a single vector of positions.  
    # mesh = np.meshgrid(x_vec, y_vec, indexing= 'xy')
    # x = mesh[0].reshape(-1,1)
    # y = mesh[1].reshape(-1,1)
    # initial_conditions = np.append(x, y, axis=1)
    # n_particles = len(x)
    
    # # Sample particles randomly.
    n_particles = 480
    if function_name == "Gyre":
        initial_conditions = np.random.rand(n_particles, 2)
        initial_conditions[:,0] = initial_conditions[:,0] * 2
        
        # Flow parameters
        parameters = { 
            "A": 0.1,
            "epsilon":0.1,
            "omega":2*np.pi/10
        }
        time_multiplier = 1
    elif function_name == "BickleyJet":
        initial_conditions = np.random.rand(n_particles, 2)
        initial_conditions[:,0] = initial_conditions[:,0] * 20 * 10**6
        initial_conditions[:,1] = initial_conditions[:,1] * 6 * 10**6 - 3 * 10**6
        
        # Flow parameters
        U_default = 62.66
        parameters = {
            "U" : U_default,
            "L" : 1.77 * 10**6,
            "r0" : 6.371 * 10**6,
            "epsilon" : [0.0075, 0.15, 0.3],
            "c" : [0.1446*U_default, 0.205*U_default, 0.461*U_default]
        }
        time_multiplier = 24 * 60 * 60

    # Next, we need to make a time vector
    t0 = 0 * time_multiplier    # initial time
    t1 = 40 * time_multiplier   # final time
    dt = 0.5 * time_multiplier  # time increment                                 
    
    time_vector = np.arange(t0,t1+dt,dt)
    '''We now need to specify the flow that the Flow object will operate on.'''
    
    flow.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters, include_gradv=True)

    # Integrate the particles over the defined time vector
    flow.integrate_trajectories()
    
    #%%
    # Generate a flow network:
    flownet = FlowNet(flow.states, t1, 
                      metric_function='kinematic_similarity',
                      kNN = round(n_particles/60),
                      delaunay=False)
    
    G = flownet.G
    # communities = nx.community.louvain_communities(G, weight="weight")
    # coloring = [G.degree(n, weight='weight') for n in sorted(list(G.nodes()))]
    # for i, com in enumerate(communities):
    #     for c in com:
    #         coloring[c] = i+1
    Gold = G
    
    flownet.coherent_structure_coloring()
    coloring = flownet.CSC
    
    #%
    coloring = np.diag(flownet.degree_matrix)
    # coloring = [G.degree(n, weight='weight') for n in sorted(list(G.nodes()))]
    # centrality = nx.katz_centrality(G, weight='weight')
    centrality = nx.pagerank(G, weight='weight')
    # comm_color = identify_communities(G)
    # coloring = [c for v, c in centrality.items()]
    
    all_degrees = [G.degree(n, weight='weight') for n in G.nodes()]
    # G = remove_edges_under_threshold(G, np.mean(flownet.adjacency)+2.5*np.std(flownet.adjacency))
    # G, removed = remove_nodes_under_threshold(G, 1*np.mean(all_degrees)+0*np.std(all_degrees))
    adj = nx.adjacency_matrix(G).todense()
    G = remove_edges_under_threshold(G, np.mean(adj)+2*np.std(adj))
    largest_cc = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(largest_cc)
    
    # Calculate the degrees of each node
    degrees_idx = [(n, Gsub.degree(n, weight='weight')) for n in Gsub.nodes()]
    degrees = [Gsub.degree(n, weight='weight') for n in Gsub.nodes()]
    sorted_pairs = sorted(degrees_idx, key=lambda pair: pair[1], reverse=False)
    sorted_nodes = [node for node, degree in sorted_pairs]
    deg_sort = [degree for node, degree in sorted_pairs]
    
    # compare old and new networks
    # nodes_Gsub = set(Gsub.nodes())
    # nodes_Gold = set(Gold.nodes())
    # removed = nodes_Gold - nodes_Gsub
    
    # for nod in removed:
    #     coloring[nod] = 0
        
    plt.figure(figsize=(8,6)) 
    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.hist(deg_sort, bins = 20)
    plt.show()
    
    # Calculate the weight (thickness) of each edge
    weights = [Gsub[u][v]['weight']/10 for u,v in Gsub.edges()]

    # Set up the color map (scale colors by degree)
    colors = plt.cm.viridis(np.linspace(0, 1, len(degrees)))

    # Setting the positions of nodes and keeping it same for node and edges
    pos = nx.spring_layout(Gsub, iterations=200, weight='weight')
    # pos = nx.spectral_layout(G, weight="weight")

    #%
    # Normalize color range with Power Law (Gamma) scaling
    gamma = 0.85
    norm = Normalize(vmin=min(deg_sort)**gamma, vmax=max(deg_sort)**gamma)
    colors = plt.cm.RdBu(norm(deg_sort)**gamma)
    
    # # Draw the nodes, specifying the color map and the degree sequence as the node size
    # nx.draw_networkx_nodes(Gsub, pos=pos, cmap=plt.get_cmap('RdBu'),
    #                     node_color=colors, node_size=100, alpha=0.9,
    #                     edgecolors=[0.9, 0.9, 0.9])

    # Define sizes of 5 communities
    sz = np.random.randint(6,7)
    sizes = np.random.randint(30,60,size=sz)
    
    diag = np.diag(np.random.rand(sz)*0.05 + 0.3)
    mat = np.random.rand(sz,sz)*0.01
    p_matrix = mat + diag
    p_matrix = (p_matrix + p_matrix.T)/2
    print(p_matrix)

    # Use the stochastic block model to generate the graph
    G = nx.stochastic_block_model(sizes, p_matrix)

    # Get the degree for each node
    degrees = dict(G.degree())

    # Map degrees to colors
    degree_color = [degrees[node] for node in G.nodes]
    
    # Normalize color range with Power Law (Gamma) scaling
    gamma = 0.75
    norm = Normalize(vmin=min(degree_color)**gamma, vmax=max(degree_color)**gamma)
    colors = plt.cm.RdBu(norm(degree_color)**gamma)
    
    # Draw the graph
    # Create a figure with a black background
    plt.figure(figsize=(6, 8), facecolor='black')
    pos = nx.spring_layout(G)  # positions for all nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_color=degree_color, cmap=plt.cm.RdBu, 
                                   node_size=100, alpha=0.9, edgecolors=[0.9, 0.9, 0.9])
    edges = nx.draw_networkx_edges(G, pos, edge_color=[0.8, 0.8, 0.8])
    # labels = nx.draw_networkx_labels(G, pos)
    ax=plt.gca()
    ax.set_facecolor('none')
    
    #%
    
    # # Create a figure with a black background
    # plt.figure(figsize=(12, 6), facecolor='black')
    
    # # Draw the nodes in order of their degree
    # for n, node in enumerate(sorted_nodes):
    #     nx.draw_networkx_nodes(Gsub, pos,
    #                         nodelist=[node],
    #                         cmap=plt.get_cmap('RdBu'),
    #                         node_color=[colors[n]],  # Adjust to match node's index in colors
    #                         node_size=100,
    #                         alpha=0.9,
    #                         edgecolors=[0.9, 0.9, 0.9])
    
    # # Draw the edges, specifying the weights as the line thicknesses
    # nx.draw_networkx_edges(Gsub, pos=pos, width=weights,
    #                        edge_color=[0.8, 0.8, 0.8])

    # Show plot
    # Set the axes background color
    ax = plt.gca()
    ax.set_facecolor('none')
    plt.show()
    
    sc = SpectralClustering(
        n_clusters=2,
        affinity='precomputed',
        n_init=100,
        assign_labels='discretize'
    )
    
    # Run the algorithm on the adjacency matrix
    # clusters = sc.fit_predict(flownet.adjacency)
        
    # %
    # Plot the initial conditions and color by degree.
    # Plotting a test frame
    frame = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # cc = ax.scatter(gyre.states[:,frame,0], gyre.states[:,frame,1], s=40, c=comm_color, cmap="inferno")  #c=lavd_array[:,frame]
    sc = ax.scatter(flow.states[:,frame,0], flow.states[:,frame,1], s=15, c=coloring, cmap="viridis")  #c=lavd_array[:,frame]
    
    sc.set_clim(vmin=np.min(coloring), vmax=np.max(coloring))
    fig.colorbar
    # fig.set_facecolor('black')
    ax.set_facecolor('black')
    
    # ax.set_xlim([0,2])
    # ax.set_ylim([0,1])
    ax.axis('scaled')
    
    #%%