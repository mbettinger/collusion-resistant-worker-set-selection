#!/usr/bin/env python
# coding: utf-8
import math
from itertools import combinations,permutations
import random
import igraph as ig

from bisect import bisect_left

def binSearch(a, x):
    i= bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


def searchFromIn(sublist,orderedList):
    def idxInOrderedListOfItemInSublist(sublistIdx):
        element=sublist[sublistIdx]
        return binSearch(orderedList, element)
    return idxInOrderedListOfItemInSublist


def valueInList(searchList):
    def returnValue(idx):
        return searchList[idx]
    return returnValue

def generatePermutationNumber(sublist,totalList):
    arrNumber=0
    #Sum(i=0:V-1|i!(sum(k=0:i|-1{xk<xi})))
    N=len(sublist)
    fact=math.factorial
    x=valueInList(sublist)
    
    arrNumber=sum([fact(N-1-i)*sum(1 for k in range(i+1,N) if x(i)>x(k)) for i in range(0,N-1)])
    return arrNumber

def rangeProduct(productRange):
    res=1
    for i in productRange:
        res*=i
    return res

def nbCombi(toFind,among):
    fact=math.factorial
    if among-toFind>toFind:
        return int(rangeProduct(range(among-toFind+1,among+1))/fact(toFind))
    else:
        return int(rangeProduct(range(toFind+1,among+1))/fact(among-toFind))

def generateCombinationNumber(sublist, totalList):
    eligibleList=sorted(totalList)
    voterList=sorted(sublist)
    combiNumber=0

    #Simplified expression? Sum(i=0:V-1|sum(k=1:(j(i)-j(i-1))-i|C(V-1 among N-j(i)+1)))
    
    nVoters=len(voterList)
    nEligible=len(eligibleList)
    nHoles=nEligible-nVoters
    
    foundHoles=0
    foundVoters=0
    
    j=searchFromIn(voterList,eligibleList)
    
    eligibleRef=0
    #print(voterList,eligibleList)
    for i in range(nVoters):
        voter=voterList[i]
        eligibleIdx=j(i)
        
        diff=eligibleIdx-eligibleRef
        #print("voter",voter,"idx",i,"EIdx",eligibleIdx,"diff",diff)
        foundVoters+=1
        for hole in range(diff):#if diff=0 range outputs []
            foundHoles+=1
            #print("nV",nVoters,"fV",foundVoters,"nE",nEligible,"fH",foundHoles)
            #print("Combi",nVoters-foundVoters,"among",nEligible-foundHoles-foundVoters+1,":",nbCombi(nVoters-foundVoters,nEligible-foundHoles-foundVoters+1))
            combiNumber+=nbCombi(nVoters-foundVoters,nEligible-foundHoles-foundVoters+1)
            
        eligibleRef=eligibleIdx+1

    return combiNumber

def generateArrangementNumber(sublist, totalList):
    totalList=sorted(totalList)
    arrNumber=0
    #ArrangementNumber =
    #  Sum(Count of arrangements of 0 to V-1 elements among N) 
    #+ combinationNumber(V among N)*NbPermutations(V non reoccurring elements)
    #+ PermutationNumber(V non reoccurring elements)
    #= Sum(i=0:V-1|N!/(N-i)!)+combiNumber(V el,N el)*V!+NumPermut(V el)
    N=len(totalList)
    V=len(sublist)
    fact=math.factorial
    j=searchFromIn(sublist,totalList)
    x=valueInList(totalList)
    
    previousArrCount=sum([rangeProduct(range(N-i+1,N+1)) for i in range(V)])
    combiNumber=generateCombinationNumber(sublist,totalList)
    permutNumber=generatePermutationNumber(sublist,totalList)
    
    arrNumber=previousArrCount+combiNumber*fact(V)+permutNumber
    return arrNumber

def orderBasedWorkerSelection(graph, nVoters, nWorkers, voterSeed, numGenFunction=generateArrangementNumber):
    
    vertices = graph.vs["name"]
    
    random.seed(voterSeed) 
    
    voters = random.sample(vertices, nVoters)
    
    seed = numGenFunction(voters, vertices)
    
    random.seed(seed)
    
    workers = random.sample(vertices, nWorkers)
    
    return workers
