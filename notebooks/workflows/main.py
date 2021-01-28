from workflow_manager.pipeline import Pipeline
from workflow_manager.step import Step
from workflow_manager.parameter_grid import parameter_grid as PG
from workflow_manager.metastep import MetaStep
from workflow_manager import meta
from workflow_manager import util

from libs import community_detection
from libs import evaluation
from libs.order_based_seed_gen import numbering
from libs import preprocessing
from itertools import permutations
import igraph as ig

import os

import numpy as np
import math
dirPath="graphs/formatted/"
filePathList=preprocessing.getFileNamesInDir(dirPath)
imgDirPath="graphs/img/"

N_VOTERS=10
VOTER_SEED=range(10,100)
COMMUNITY_WORKER_REPARTITION=[
                                community_detection.worker_selection.sizeProRataWorkerAssignement,
                                community_detection.worker_selection.sizeOrderedRoundRobinWorkerAssignement,
                                #community_detection.worker_selection.diameterProRataWorkerAssignement,
                                #community_detection.worker_selection.diameterWorkerAssignment,
                                #community_detection.worker_selection.nodeEdgeRatioProRataWorkerAssignement,
                                #community_detection.worker_selection.nodeDensityRatioProRataWorkerAssignement
                             ]
WITH_BOUNDARY=[True,
              False
             ]

#COMMUNITY_DETECTION
MIN_DIST_CLOSE=range(1,4)
#MIN_DIST_FAR

communityBasedPipeline=Pipeline([
        Step(
            util.add_params,
            params=PG({"community_worker_repartition":COMMUNITY_WORKER_REPARTITION,
                       "withBoundary":WITH_BOUNDARY
                      }),
            outputs=["community_worker_repartition","withBoundary"]
        ),
        Step(
            community_detection.worker_selection.assignWorkers,
            args=["partition","nWorkers","community_worker_repartition","withBoundary"],
            outputs=["workers"],
            read_only_outputs=set("workers")
        ),
        Step(
            lambda func:func.__name__,
            args=["community_worker_repartition"],
            outputs=["assignment"],
            keep_inputs=False,
            name="Keep only function name"
        ),
            Step(
            util.add_params,
            params={"voterSeed":"NA","nVoters":"NA"},
            outputs=["voterSeed"]
        )
    ],name="Community-based worker selection")

voterOrderBasedPipeline=Pipeline([
        Step(
            util.add_params,
            params=PG({"nVoters":N_VOTERS,"voterSeed":VOTER_SEED,
                       "numGenFunction":numbering.generateArrangementNumber
                      }),
            outputs=["nVoters","voterSeed","numGenFunction"]
        ),
        Step(
            lambda p,nV,nW,vS,nGF: numbering.orderBasedWorkerSelection(p.graph,nV,nW,vS,nGF),
            args=["partition","nVoters","nWorkers","voterSeed","numGenFunction"],
            outputs=["workers"],
            read_only_outputs=set("workers")
        ),
        Step(
            lambda func:func.__name__,
            args=["numGenFunction"],
            outputs=["assignment"],
            keep_inputs=False,
            name="Keep only function name"
        ),
        Step(
            util.add_params,
            params={"withBoundary":False},
            outputs=["withBoundary"]
        )
    ],name="Voting-order-based worker selection")
                    

pipeline=Pipeline([
                Step(
                    util.add_params,
                    params=PG({"filename":filePathList}),
                    outputs=["filefullname"]
                ),
                Step(
                    lambda fn:fn.split(".")[0],
                    args=["filefullname"],
                    outputs=["filename"]
                ),
                Step(
                    lambda fn:dirPath+fn,
                    args=["filefullname"],
                    outputs=["filepath"]
                ),
                Pipeline([
                    Step(
                        lambda fn:imgDirPath+fn+"/",
                        args=["filename"],
                        outputs=["imgDirPath"]
                    ),
                    Step(
                        lambda imgDP: os.mkdir(imgDP) if not os.path.isdir(imgDP) else False,
                        args=["imgDirPath"]
                    ),
                ],name="Create folders"),
                Step(
                    lambda fp:ig.Graph.Read(fp,format="ncol").as_undirected(),
                    args=["filepath"],
                    outputs=["graph"],
                    name="Load graph"
                ),
                Pipeline([
                    Step(
                        lambda g:[g.community_multilevel,
                                    g.community_label_propagation,
                                    #g.community_leading_eigenvector
                                     ],#COMMUNITY_DETECTION
                        args=["graph"],
                        outputs=["community_detection_method"],
                        keep_inputs=False
                    ),
                    MetaStep(
                        meta.split,
                        params={"key":"community_detection_method"}
                    ),
                    Step(
                        community_detection.worker_selection.findCommunities,
                        args=["community_detection_method"],
                        outputs=["partition"],
                        read_only_outputs=set(["partition"])
                    ),
                    Step(
                        lambda func:func.__name__,
                        args=["community_detection_method"],
                        outputs=["comDet"],
                        keep_inputs=False,
                        name="Keep only community detection method name"
                    )
                ],name="Community detection"),
                Pipeline([
                    Step(
                        lambda fn,iDP,comDet: (
                                    iDP+fn+"_"+comDet+"_diameters.png",
                                    iDP+fn+"_"+comDet+"_radii.png",
                                    iDP+fn+"_"+comDet+"_nodesPerCom.png",
                                   ),
                        args=["filename","imgDirPath","comDet"],
                        outputs=["imgDiamPath","imgRadPath","imgNpCPath"]
                    ),
                    Step(
                        lambda p: p.graph.diameter(),
                        args=["partition"],
                        outputs=["diameter"]
                    ),
                    Step(
                        evaluation.diameters,
                        args=["partition"],#,"imgDiamPath"],
                        outputs=["diameters"]
                    ),
                    Step(
                        evaluation.radii,
                        args=["partition"],#,"imgRadPath"],
                        outputs=["radii"]
                    ),
                    Step(
                        evaluation.optimalWorkerCountByDiameters,
                        args=["partition"],
                        outputs=["optimalNWorkers"]
                    ),
                    Step(
                        evaluation.nodesPerCommunity,
                        args=["partition","imgNpCPath"],
                        outputs=["nodesPerCommunity"]
                    )
                ],name="Graph & Community metrics"),
                Pipeline([
                    Step(
                        lambda p:[int(nW) for nW in np.logspace(1, math.log10(len(p.graph.vs)//5), 10, endpoint=True,dtype=int)],
                        args=["partition"],
                        outputs=["nWorkers"],
                        read_only_outputs=set("nWorkers")
                    ),
                    MetaStep(
                        meta.split,
                        params={"key":"nWorkers"}
                    )
                ],name="nWorkers spacing"),
                [
                    #voterOrderBasedPipeline,
                    communityBasedPipeline
                ],
                Pipeline([
                    Step(
                        lambda fn,iDP,nw,vs,comDet,assignment,wB: (
                                    "{}{}_{}_{}_wb{}_Graph_{}w{}s.png".format(iDP,fn,comDet,assignment,wB,nw,vs),
                                    "{}{}_{}_{}_wb{}_ClustDist_{}w{}s.png".format(iDP,fn,comDet,assignment,wB,nw,vs),
                                    "{}{}_{}_{}_wb{}_InClustDist_{}w{}s.png".format(iDP,fn,comDet,assignment,wB,nw,vs),
                                    "{}{}_{}_{}_wb{}_workersPerCom_{}w{}s.png".format(iDP,fn,comDet,assignment,wB,nw,vs),
                                   ),
                        args=["filename","imgDirPath","nWorkers","voterSeed","comDet","assignment","withBoundary"],
                        outputs=["imgPath","imgCDPath","imgInCDPath","imgWpCPath"]
                    ),
                    Step(
                        evaluation.workerInClusterDistances,
                        args=["partition","workers","imgInCDPath"],
                        outputs=["workerComDistances"]
                    ),
                    Step(
                        evaluation.workerDistances,
                        args=["partition","workers","imgCDPath"],
                        outputs=["workerDistances"]
                    ),
                    Step(
                        evaluation.workersPerCommunity,
                        args=["partition","workers","imgWpCPath"],
                        outputs=["workersPerCommunity"]
                    ),
                    Step(
                        lambda p:p.graph,
                        args=["partition"],
                        outputs=["graph"],
                        keep_inputs=False
                    ),
                    Step(
                        community_detection.draw.emphasizeWorkers,
                        args=["graph","workers"],
                        outputs=["graph"]
                    ),
                    Step(
                        community_detection.draw.drawGraph,
                        args=["graph","imgPath"]
                    ),
                ],name="Workers metrics"),
                [
                    Pipeline([
                        Step(
                            util.add_params,
                            params={"distType":"close"},
                            outputs=["distType"]
                        ),
                        Step(
                            util.add_params,
                            params=PG({"minDist":MIN_DIST_CLOSE}),
                            outputs=["minDist"]
                        ),
                        Step(
                            lambda mD,nw,vs,comDet,aF,fn,iDP:"{}{}_{}-closecliquesGraph_{}w{}s{}aF{}comDet.png".format(
                                                    iDP,fn,mD,nw,vs,aF,comDet),
                            args=["minDist","nWorkers","voterSeed","comDet","assignment","filename","imgDirPath"],
                            outputs=["cliqueImgPath"]
                        ),
                        Step(
                            evaluation.closenessCliques,
                            args=["graph", "minDist", "workers"],#, "cliqueImgPath"],
                            outputs=["candidates"]
                        )
                    ],name="Close workers cliques"),
                    Pipeline([
                        Step(
                            util.add_params,
                            params={"distType":"far"},
                            outputs=["distType"]
                        ),
                        Step(
                            lambda d:list(range(6,d+1)),#MIN_DIST_FAR
                            args=["diameter"],
                            outputs=["minDist"],
                        ),
                        MetaStep(
                            meta.split,
                            params={"key":"minDist"}
                        ),
                        Step(
                            lambda mD,nw,vs,comDet,aF,fn,iDP:"{}{}_{}-farcliquesGraph_{}w{}s{}aF{}comDet.png".format(
                                                    iDP,fn,mD,nw,vs,aF,comDet),
                            args=["minDist","nWorkers","voterSeed","comDet","assignment","filename","imgDirPath"],
                            outputs=["cliqueImgPath"]
                        ),
                        Step(
                            evaluation.farnessCliques,
                            args=["graph", "minDist", "workers"],#, "cliqueImgPath"],
                            outputs=["candidates"]
                        )
                    ],name="Far workers cliques"),
                ],
                MetaStep(
                    meta.remove_params,
                    params={"keys":["graph"]}
                )
],name="MainWorkflow")