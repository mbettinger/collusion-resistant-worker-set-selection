import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

#Given a list of nodes, give distances between nodes
def calcDistBtwnNodes(graph,chosenSourceNames,chosenTargetNames=None):
    if chosenTargetNames is None:
        return graph.shortest_paths_dijkstra(source=chosenSourceNames, target=chosenSourceNames)
    else:
        return graph.shortest_paths_dijkstra(source=chosenSourceNames, target=chosenTargetNames)

def graphDistances(F,workerIdsSrc,workerIdsTarget=None,cluster=None):
    matPCC=None
    if workerIdsTarget is None:
        workerIdsTarget=workerIdsSrc
    matPCC=calcDistBtwnNodes(F,workerIdsSrc,workerIdsTarget)

    PCCList=[]
    for sublist in matPCC:
        PCCList.extend(sublist)
    maxDist=max(PCCList,default=1)
    PCCSelf=[]
    PCCSameCluster=[]
    PCCOtherCluster=[]

    for i, dists in enumerate(matPCC):
        source=F.vs.find(workerIdsSrc[i])
        for j in range(len(matPCC[i])):
            target=F.vs.find(workerIdsTarget[j])
            #print(source,target)
            if source["name"]==target["name"]:
                PCCSelf.append(matPCC[i][j])
            elif source["cluster"]==target["cluster"] and (cluster is None or source["cluster"]==cluster):
                PCCSameCluster.append(matPCC[i][j])
            else:
                PCCOtherCluster.append(matPCC[i][j])

    return PCCSelf,PCCSameCluster,PCCOtherCluster


def workerDistances(graph,workerIds,imgPath):
    data=list(graphDistances(graph,workerIds))
    maxDist=max([max(l,default=0) for l in data],default=0)
    colors=["grey","red","blue"]
    labels=["self","same community","other community"]
    # fixed bin size
    bins = np.arange(0, 100, 1) # fixed bin size

    plt.xlim([0, maxDist+1])
    plt.yscale("log")
    plt.hist(data, bins=bins, color=colors, label=labels, stacked=True)
    plt.title('Shortest path lengths between workers')
    plt.xlabel('Shortest path length')
    plt.ylabel('Count of workers')

    plt.savefig(imgPath)
    plt.close()
    
def sumWorkerDists(graph,workerIds):
    return sum([sum(l) for l in graphDistances(graph,workerIds)])

def workerInClusterDistances(graph,partition,workerIds,imgPath):
    nbClusters=len(partition.subgraphs())
    nbCol=4
    fig,ax = plt.subplots(nrows = nbClusters//nbCol, ncols = nbCol,sharex=True, sharey=True,figsize=(nbCol*4,3*nbClusters//nbCol))

    for idx, subgraph in enumerate(partition.subgraphs()):
        inClusterWorkers=[w for w in workerIds if w in subgraph.vs["name"]]
        data=list(graphDistances(graph,inClusterWorkers,workerIds,subgraph.vs[0]["cluster"]))
        colors=["grey","red","blue"]
        labels=["self","same community","other community"]
        maxDist=max([max(l,default=0) for l in data],default=0)
        
        # fixed bin size
        bins = np.arange(0, maxDist+1, 1) # fixed bin size

        ax[idx//nbCol][idx%nbCol].hist(data, bins=bins, color=colors, label=labels, stacked=True)
        ax[idx//nbCol][idx%nbCol].title.set_text('SPLs from {} worker{} in cluster {}'.format(len(inClusterWorkers),"s" if len(inClusterWorkers)>1 else "",idx))
    fig.xlim=[0, maxDist+1]
    fig.yscale="log"
    fig.suptitle='Shortest path lengths between workers'
    fig.xlabel='Shortest path length'
    fig.ylabel='Count of workers'
    fig.savefig(imgPath)
    plt.close(fig)
    
def diameters(partition,imgPath):
    subgraphs=partition.subgraphs()
    diams=list([subgraph.diameter() for subgraph in subgraphs])
    plt.plot(diams)
    plt.savefig(imgPath)
    
def radii(partition,imgPath):
    subgraphs=partition.subgraphs()
    radii=list([subgraph.radius() for subgraph in subgraphs])
    plt.plot(radii)
    plt.savefig(imgPath)
    
def nodesPerCommunity(graph,imgPath):
    nbClusters=len(Counter(graph.vs["cluster"]))
    bins = np.arange(0, nbClusters+1, 1)
    plt.hist(graph.vs["cluster"], bins=bins)
    plt.title('Number of nodes in communities')
    plt.xlabel('Community')
    plt.ylabel('Count of nodes')

    plt.savefig(imgPath)
    
def workersPerCommunity(graph,workerIds,imgPath):
    nbClusters=len(Counter(graph.vs["cluster"]))
    data=[graph.vs.find(worker)["cluster"] for worker in workerIds]
    bins = np.arange(0, nbClusters+1, 1)
    plt.hist(data, bins=bins)
    plt.title('Number of workers in communities')
    plt.xlabel('Community')
    plt.ylabel('Count of workers')

    plt.savefig(imgPath)
    
def optimalWorkerCountByDiameters(partition):
    diameters=[subgraph.diameter() for subgraph in partition.subgraphs()]

    workers=[diameter//2+diameter%2 for diameter in diameters]
    assignedWorkers=sum(workers)
    return assignedWorkers