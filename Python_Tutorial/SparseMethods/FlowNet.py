
#%%
import os, sys, copy
import numpy as np
import networkx as nx
from networkx.algorithms import community
import sympy as sym
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
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


def computeCSC(
    trajectories: np.ndarray, interval: List = None, 
    adjacency: Union[str, Callable] = 'kinematic_dissimilarity') -> np.ndarray:
    
    pass


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


def identify_communities(G):
    communities = community.greedy_modularity_communities(G, weight="weight")
    communities = community.asyn_lpa_communities(G, weight="weight")

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
    function_name = "Gyre"
    gyre = Flow()

    ''' Now, we need to define the initial conditions for our integrations. We will define them on a grid for now.  '''
    # Specify the flow domain using dim[0] = x axis and dim[1] = y axis
    domain = np.array([[0, 2],[0, 1]])

    # Now, make vectors associated with each axis.
    # n_y = 10            # number of rows
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
    n_particles = 100
    initial_conditions = np.random.rand(n_particles, 2)
    initial_conditions[:,0] = initial_conditions[:,0] * 2

    # Next, we need to make a time vector
    t0 = 0      # initial time
    t1 = 15      # final time
    dt = 0.25    # time increment <- # For standard FTLE we will only need the first and last time, but 
                                    # it will be helpful when computing LAVD to have increments.
    interval = 200  # Time for computing the LAVD.
    
    time_vector = np.arange(t0,t1+dt,dt)
    '''We now need to specify the flow that the Flow object will operate on.'''
    parameters = {  # These are specified as defaults as well. 
        "A": 0.1,
        "epsilon":0.1,
        "omega":2*np.pi/10
    }
    gyre.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters, include_gradv=True)

    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    
    #%%
    # Generate a flow network:
    flownet = FlowNet(gyre.states, 15*dt, 
                      metric_function='kinematic_similarity',
                      kNN = round(n_particles/10),
                      delaunay=False)
    
    G = flownet.G
    coloring = [G.degree(n, weight='weight') for n in sorted(list(G.nodes()))]
    centrality = nx.information_centrality(G, weight='weight')
    comm_color = identify_communities(G)
    # coloring = [c for v, c in centrality.items()]
    
    G = remove_edges_under_threshold(G, 0)#np.mean(flownet.adjacency)+2.5*np.std(flownet.adjacency))
    largest_cc = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(largest_cc)
    
    # Calculate the degrees of each node
    degrees = [Gsub.degree(n, weight='weight') for n in Gsub.nodes()]
    deg_sort = sorted(degrees, reverse=True)
    
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
    pos = nx.spring_layout(Gsub, iterations=100, weight='weight')
    # pos = nx.spectral_layout(G, weight="weight")

    # Draw the nodes, specifying the color map and the degree sequence as the node size
    nx.draw_networkx_nodes(Gsub, pos=pos, cmap=plt.get_cmap('viridis'), 
                        node_color=degrees, node_size=200, alpha=0.9)

    # Draw the edges, specifying the weights as the line thicknesses
    nx.draw_networkx_edges(Gsub, pos=pos, width=weights)

    # Show plot
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
    sc = ax.scatter(gyre.states[:,frame,0], gyre.states[:,frame,1], s=15, c=coloring, cmap="viridis")  #c=lavd_array[:,frame]
    
    sc.set_clim(vmin=np.min(coloring), vmax=np.max(coloring))
    fig.colorbar
    # fig.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.set_xlim([0,2])
    ax.set_ylim([0,1])
    ax.axis('scaled')
    
    #%%