import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sklearn.cluster as cluster
import sklearn.neighbors
import sklearn.metrics as metrics
from numpy import linalg

def question_1():
    # (a)
    print("----Part a----")
    df = pd.read_csv('Groceries.csv')
    # count unique items in each customer
    data = df.groupby(['Customer'])['Item'].count()
    print(len(data))
    # histogram
    plt.hist(data)
    plt.xlabel('Unique Items')
    plt.ylabel('Count')
    plt.title("Histogram of unique items")
    plt.show()
    # get 25,50,75 percentile
    quarts = np.percentile(data, [25, 50, 75])
    print(f"25%: {quarts[0]}, 50%: {quarts[1]}, 75%: {quarts[2]}")
    
    # (b)
    print("----Part b----")
    # group by items, and make it into a list of lists
    data_items = df.groupby(['Customer'])['Item'].apply(list).values.tolist()
    # apriori alg
    te = TransactionEncoder()
    te_ary = te.fit(data_items).transform(data_items)
    item_indicators = pd.DataFrame(te_ary, columns = te.columns_)
    frequent_item_sets = apriori(item_indicators, 
                                 min_support = 75 / len(data_items),
                                 use_colnames = True, max_len = None) # how to determine max_len?
    total_item_sets = len(frequent_item_sets)
    print(f"{total_item_sets} itemsets")
    largest_k = len(frequent_item_sets['itemsets'][total_item_sets - 1])
    print(f"Largest k: {largest_k}")
    
    # (c)
    print("----Part c----")
    ass_rules = association_rules(frequent_item_sets, metric = "confidence",
                                  min_threshold = 0.01)
    print(f"{len(ass_rules)} Association rules")
    
    # (d)
    print("----Part d----")
    plt.scatter(ass_rules['confidence'], ass_rules['support'], 
                c = ass_rules['lift'], s = ass_rules['lift'])
    plt.xlabel("Confidence")
    plt.ylabel("Support")
    plt.title("Support vs Confidence")
    color_bar = plt.colorbar()
    color_bar.set_label("Lift")
    plt.show()
    print("Just a graph for this part")
    
    # (e)
    print("----Part e----")
    ass_rules_e = association_rules(frequent_item_sets, metric = "confidence",
                                    min_threshold = 0.6)
    print(ass_rules_e.to_string())
    
def question_4():
    clusters = [[-2, -1, 1, 2, 3], [4, 5, 7, 8]]
    
    # silhoutte Width
    print("----Silhoutte Width----")
    print("Observation 2 in cluster 0: -1")
    a = (1 / (len(clusters[0]) - 1)) * sum([abs(-1 - x) for x in clusters[0]])
    print(f"a(i) = {a}")
    b = min([1 / len(clusters[1]) * sum([abs(-1 - x) for x in cluster_x]) 
             for cluster_x in clusters if -1 not in cluster_x])
    print(f"b(i) = {b}")
    s = (b - a) / max([a, b])
    print(f"s(i) = {s}")
    
    # davies bouldin
    print("----Davies Bouldin----")
    s0 = (1/len(clusters[0])) * sum([abs(x - (sum(clusters[0])/len(clusters[0]))) for x in clusters[0]])
    s1 = (1/len(clusters[1])) * sum([abs(x - (sum(clusters[1])/len(clusters[1]))) for x in clusters[1]])
    print("Cluster-wise Davies-Bouldin values:")
    print(f"Cluster 0: {s0}")
    print(f"Cluster 1: {s1}")
    m01 = abs((sum(clusters[0])/len(clusters[0])) - (sum(clusters[1])/len(clusters[1])))
    r01 = (s0 + s1) / m01
    # tf does rk = max(rk01) mean in the slides
    # if more than 2 clusters- does it choose the max of r01 found?
    # only 2 clusters now, so can disregard it
    dbi = (1/len(clusters)) * sum([r01])
    print(f"Davies-Bouldin index: {dbi}")
    
def question_5():
    # (a)
    print("----Part a----")
    data = pd.read_csv('FourCircle.csv')
    plt.scatter(data['x'], data['y'])
    plt.title('Four Circle')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print('Looking at the graph, there seems to be 4 clusters')
    
    # (b)
    print("----Part b----")
    cl_data = data[['x','y']]
    kmeans = cluster.KMeans(n_clusters = 4, random_state = None).fit(cl_data)
    plt.scatter(data['x'], data['y'], c = kmeans.labels_)
    plt.title('Four Circle Part b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print('Just a graph')
    
    # (c)
    print("----Part c----")
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors = 6, algorithm = 'brute',
                                               metric = 'euclidean')
    nbrs = neigh.fit(cl_data)
    d3, i3 = nbrs.kneighbors(cl_data)
    # get distances
    distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    distances = distObject.pairwise(cl_data)
    # adjacency matrix
    adjacency = np.zeros((len(cl_data), len(cl_data)))
    for i in range(len(cl_data)):
        for j in i3[i]:
            adjacency[i, j] = math.exp(-(distances[i][j])**2)
    # make it symmetric
    adjacency = 0.5 * (adjacency + adjacency.transpose())
    # degree matrix
    degree = np.zeros((len(cl_data), len(cl_data)))
    for i in range(len(cl_data)):
        summ = 0
        for j in range(len(cl_data)):
            summ += adjacency[i, j]
        degree[i, i] = summ
    # laplacian matrix
    lap = degree - adjacency
    # get eigenval/vec
    evals, evecs = linalg.eigh(lap)
    # plot
    sequence = np.arange(1, 15, 1)
    plt.plot(sequence, evals[0:14], marker = "o")
    plt.xlabel('Sequence')
    plt.ylabel('Eigenvalue')
    plt.title("6 Neighbors")
    plt.xticks(sequence)
    plt.grid()
    plt.show()
    
    # (d)
    print("----Part d----")
    # print eigenvalues that are practically zero
    print(f"{evals[:4]} eigenvalues that are practically zero")
    
    # (e)
    print("----Part e----")
    # kmeans on 4 evecs that associates with the 4 smallest evals
    c = evecs[:, [0,1,2,3]]
    kmeans = cluster.KMeans(n_clusters = 4, random_state = None).fit(c)
    plt.scatter(data['x'], data['y'], c = kmeans.labels_)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Four Circle Part e")
    print('Just a graph')

if __name__ == '__main__':
    question_1()
    #question_4()
    #question_5()
    