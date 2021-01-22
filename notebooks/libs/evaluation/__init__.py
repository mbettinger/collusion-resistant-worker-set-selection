import numpy as np
import pandas as pd
import igraph as ig

from matplotlib import pyplot as plt
from collections import Counter

import signal

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


def workerDistances(graph,workerIds,imgPath):
    data=list(graphDistances(graph,workerIds))
    colors=["grey","red","blue"]
    labels=["self","same community","other community"]
    
    maxDist=max([max(l,default=0) for l in data],default=0)

    # fixed bin size
    bins = np.arange(0, maxDist+2, 1)
    
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    fig.xlim=[0, maxDist+2]
    fig.yscale="log"
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
    
    fig.savefig(imgPath)
    plt.close(fig)
    
    return heights
    
def sumWorkerDists(graph,workerIds):
    return sum([sum(l) for l in graphDistances(graph,workerIds)])

def workerInClusterDistances(graph,partition,workerIds,imgPath):
    nbClusters=len(partition.subgraphs())
    nbCol=4
    fig,ax = plt.subplots(nrows = nbClusters//nbCol, ncols = nbCol,sharex=True, sharey=True,figsize=(nbCol*4,3*nbClusters//nbCol))
    heights={}
    maxMaxDist=0
    for idx, subgraph in enumerate(partition.subgraphs()):
        inClusterWorkers=[w for w in workerIds if w in subgraph.vs["name"]]
        data=list(graphDistances(graph,inClusterWorkers,workerIds,subgraph.vs[0]["cluster"]))
        colors=["grey","red","blue"]
        labels=["self","same community","other community"]
        maxDist=max([max(l,default=0) for l in data],default=0)
        maxMaxDist=max(maxDist,maxMaxDist)
        
        # fixed bin size
        bins = np.arange(0, maxDist+2, 1) # fixed bin size

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
        
    fig.xlim=[0, maxMaxDist+2]
    fig.yscale="log"
    fig.suptitle='Shortest path lengths between workers'
    fig.xlabel='Shortest path length'
    fig.ylabel='Count of workers'
    fig.savefig(imgPath)
    plt.close(fig)
    
    return heights
    
def diameters(partition,imgPath):
    subgraphs=partition.subgraphs()
    diams=list([subgraph.diameter() for subgraph in subgraphs])
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.plot(diams)
    fig.savefig(imgPath)
    plt.close(fig)
    return diams

def radii(partition,imgPath):
    subgraphs=partition.subgraphs()
    radii=list([subgraph.radius() for subgraph in subgraphs])
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.plot(radii)
    fig.savefig(imgPath)
    plt.close(fig)
    
    return radii
    
def nodesPerCommunity(graph,imgPath):
    nbClusters=len(Counter(graph.vs["cluster"]))
    bins = np.arange(0, nbClusters+1, 1)
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.hist(graph.vs["cluster"], bins=bins)

    fig.savefig(imgPath)
    plt.close(fig)
    return nbClusters
    
def workersPerCommunity(graph,workerIds,imgPath):
    nbClusters=len(Counter(graph.vs["cluster"]))
    data=[graph.vs.find(worker)["cluster"] for worker in workerIds]
    bins = np.arange(0, nbClusters+1, 1)
    
    fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax.hist(data, bins=bins)

    fig.savefig(imgPath)
    plt.close(fig)
    return nbClusters
    
def optimalWorkerCountByDiameters(partition):
    diameters=[subgraph.diameter() for subgraph in partition.subgraphs()]

    workers=[diameter//2+diameter%2 for diameter in diameters]
    assignedWorkers=sum(workers)
    return assignedWorkers

def closenessCliques(graph, minDist, imgPath, nodesSubset=None):
    nodes=graph.vs
    candidates=set()
    if nodesSubset is None:
        matPCC=graph.shortest_paths_dijkstra()
        nodesSubset=nodes["name"]
    else:
        nodesIdx=[graph.vs.find(n).index for n in nodesSubset]
        matPCC=graph.shortest_paths_dijkstra(nodesIdx,nodesIdx)

    dfPCC=pd.DataFrame(matPCC,nodesSubset,nodesSubset)
    restrictDf=(dfPCC.where((dfPCC <= minDist) & (dfPCC > 0))).values.tolist()
    g = ig.Graph.Adjacency(restrictDf).as_undirected()
    g.vs["name"]=nodesSubset
    ig.plot(g,imgPath)

    try:
        cliques=deadline(30)(g.largest_cliques)()
        candidates=sorted([[nodesSubset[idx] for idx in clique] for clique in cliques],reverse=True)
        
    except TimeoutException:
        print("Timeout")
        candidates=["Timeout"]
    return candidates