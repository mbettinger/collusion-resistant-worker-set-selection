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
        if len(sublist)>0:
            element=sublist[sublistIdx]
            return binSearch(orderedList, element)
        else:
            return -1
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

    #Simplified expression:
    #=Sum(k=0:(j(0)-1)|C(V-1 among N-j(0)+k))
    #+Sum(i=1:V-1|sum(k=0:(j(i)-j(i-1)-2)|C(V-(i+1) among N-j(i)+k)))
    
    nVoters=len(voterList)
    nEligible=len(eligibleList)
    
    j=searchFromIn(voterList,eligibleList)

    firstElement=sum([nbCombi(nVoters-1,nEligible-j(0)+k)  for k in range(0,j(0))])
    combiNumber=firstElement+sum([
                                sum([
                                    nbCombi(nVoters-i-1,nEligible-j(i)+k) 
                                    for k in range(0,j(i)-j(i-1)-1)]) 
                                for i in range(1,nVoters)])
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
