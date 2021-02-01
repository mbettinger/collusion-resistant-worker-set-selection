import numpy as np
import pandas as pd
import igraph as ig

from matplotlib import pyplot as plt
from collections import Counter

import signal
import math

class TimeoutException(Exception):
    pass

def deadline(timeout, *args):
    """is a the decotator name with the timeout parameter in second"""
    def decorate(f):
        """ the decorator creation """
        def handler(signum, frame):
            """ the handler for the timeout """
            raise TimeoutException() #when the signal have been handle raise the exception

        def new_f(*args):
            """ the initiation of the handler, 
            the lauch of the function and the end of it"""
            signal.signal(signal.SIGALRM, handler) #link the SIGALRM signal to the handler
            signal.alarm(timeout) #create an alarm of timeout second
            res = f(*args) #lauch the decorate function with this parameter
            signal.alarm(0) #reinitiate the alarm
            return res #return the return value of the fonction
    
        new_f.__name__ = f.__name__
        return new_f
    return decorate

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


def workerDistances(partition,workerIds,imgPath=None):
    graph=partition.graph
    data=list(graphDistances(graph,workerIds))
    data=[[d if d!=math.inf else -1 for d in l] for l in data]
    diameter=graph.diameter()
    
    colors=["grey","red","blue"]
    labels=["self","same community","other community"]
    # fixed bin size
    bins = np.arange(-1, diameter+2, 1) # fixed bin size
    
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    fig.xlim=[-1, diameter+2]
    ax.set_yscale("log")
    ax.hist(data, bins=bins, color=colors, label=labels, stacked=True)
    fig.suptitle='Shortest path lengths between workers'
    fig.xlabel='Shortest path length'
    fig.ylabel='Count of workers'

    ax = plt.gca()
    p = ax.patches
    comHeights = [int(patch.get_height()) for patch in p]
    heights = {"self":comHeights[:len(comHeights)//3],
               "same":comHeights[len(comHeights)//3:2*len(comHeights)//3],
               "other":comHeights[2*len(comHeights)//3:]
              }
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    
    return heights
    
def sumWorkerDists(graph,workerIds):
    return sum([sum(l) for l in graphDistances(graph,workerIds)])

def workerInClusterDistances(partition,workerIds,imgPath=None):
    graph=partition.graph
    nbClusters=len(partition.subgraphs())
    diameter=graph.diameter()
    nbCol=4
    fig,ax = plt.subplots(nrows = (nbClusters+nbCol-1)//nbCol, ncols = min(nbCol,nbClusters),sharex=True, sharey=True,figsize=(nbCol*4,3*nbClusters//nbCol))
    if nbClusters==1:
        ax=[[ax]]
    elif nbClusters<=nbCol:
        ax=[ax]
    heights={}
    maxMaxDist=0
    for idx, subgraph in enumerate(partition.subgraphs()):
        inClusterWorkers=[w for w in workerIds if w in subgraph.vs["name"]]
        data=list(graphDistances(graph,inClusterWorkers,workerIds,subgraph.vs[0]["cluster"]))
        data=[[d if d!=math.inf else -1 for d in l] for l in data]
        colors=["grey","red","blue"]
        labels=["self","same community","other community"]
        
        # fixed bin size
        bins = np.arange(-1, diameter+2, 1) # fixed bin size
        
        ax[idx//nbCol][idx%nbCol].set_yscale("log")
        ax[idx//nbCol][idx%nbCol].hist(data, bins=bins, color=colors, label=labels, stacked=True)
        ax[idx//nbCol][idx%nbCol].title.set_text('SPLs from {} worker{} in cluster {}'.format(
                            len(inClusterWorkers),
                            "s" if len(inClusterWorkers)>1 else "",
                            idx))

        p = ax[idx//nbCol][idx%nbCol].patches
        comHeights = [int(patch.get_height()) for patch in p]
        heights[idx]={"self":comHeights[:len(comHeights)//3],
                      "same":comHeights[len(comHeights)//3:2*len(comHeights)//3],
                      "other":comHeights[2*len(comHeights)//3:]
                     }
        
    fig.xlim=[-1, diameter+2]
    fig.suptitle='Shortest path lengths between workers'
    fig.xlabel='Shortest path length'
    fig.ylabel='Count of workers'
    
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    
    return heights
    
def diameters(partition,imgPath=None):
    subgraphs=partition.subgraphs()
    diams=list([subgraph.diameter() for subgraph in subgraphs])
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.plot(diams)
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    return diams

def radii(partition,imgPath=None):
    subgraphs=partition.subgraphs()
    radii=list([subgraph.radius() for subgraph in subgraphs])
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.plot(radii)
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    
    return radii
    
def densities(partition,imgPath=None):
    subgraphs=partition.subgraphs()
    densities=list([subgraph.density() for subgraph in subgraphs])
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.plot(densities)
    ax.grid()
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    
    return densities
    
def nodesPerCommunity(partition,imgPath=None):
    graph=partition.graph
    nbClusters=len(Counter(graph.vs["cluster"]))
    bins = np.arange(0, nbClusters+1, 1)
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.hist(graph.vs["cluster"], bins=bins)
    
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    return nbClusters
    
def workersPerCommunity(partition,workerIds,imgPath=None):
    graph=partition.graph
    nbClusters=len(Counter(graph.vs["cluster"]))
    data=[graph.vs.find(worker)["cluster"] for worker in workerIds]
    bins = np.arange(0, nbClusters+1, 1)
    
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.hist(data, bins=bins)
    
    if imgPath is not None:
        fig.savefig(imgPath)
    plt.close(fig)
    return nbClusters
    
def optimalWorkerCountByDiameters(partition):
    diameters=[subgraph.diameter() for subgraph in partition.subgraphs()]

    workers=[diameter//2+diameter%2 for diameter in diameters]
    assignedWorkers=sum(workers)
    return assignedWorkers

def closenessCliques(graph, maxDist, nodesSubset=None, imgPath=None):
    criterion=lambda df,mD:(df <= mD) & (df > 0)
    return graphCliques(criterion, graph, maxDist, nodesSubset, imgPath)

def farnessCliques(graph, minDist, nodesSubset=None, imgPath=None):
    criterion=lambda df,mD:df >= mD
    return graphCliques(criterion, graph, minDist, nodesSubset, imgPath)

def graphCliques(criterion, graph, minDist, nodesSubset=None, imgPath=None):
    nodes=graph.vs
    candidates=set()
    if nodesSubset is None:
        matPCC=graph.shortest_paths_dijkstra()
        nodesSubset=nodes["name"]
    else:
        nodesIdx=[graph.vs.find(n).index for n in nodesSubset]
        matPCC=graph.shortest_paths_dijkstra(nodesIdx,nodesIdx)

    dfPCC=pd.DataFrame(matPCC,nodesSubset,nodesSubset)
    restrictDf=(dfPCC.where(criterion(dfPCC,minDist))).values.tolist()
    g = ig.Graph.Adjacency(restrictDf).as_undirected()
    g.vs["name"]=nodesSubset
    if imgPath is not None:
        ig.plot(g,imgPath)

    try:
        cliques=deadline(10)(g.largest_cliques)()
        candidates=sorted([[nodesSubset[idx] for idx in clique] for clique in cliques],reverse=True)
        
    except TimeoutException:
        print("Timeout")
        candidates=["Timeout"]
    return candidates