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

import numpy as np
import math
dirPath="graphs/formatted/"
filePathList=preprocessing.getFileNamesInDir(dirPath)
imgDirPath="graphs/img/"
pipeline=Pipeline([
                Step(
                    util.add_params,
                    params=PG({"filepath":filePathList}),
                    outputs=["filepath"]
                ),
                Step(
                    lambda fn:dirPath+fn,
                    args=["filepath"],
                    outputs=["filepath"]
                ),
                Step(
                    lambda fp:ig.Graph.Read(fp,format="ncol").as_undirected(),
                    args=["filepath"],
                    outputs=["graph"]
                ),
                Step(
                    community_detection.worker_selection.findCommunities,
                    args=["graph"],
                    outputs=["graph","partition"],
                    read_only_outputs=set(["graph"])
                ),
                Pipeline([
                    Step(
                        lambda fp: (
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_diameters.png",
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_radii.png",
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_nodesPerCom.png",
                                   ),
                        args=["filepath"],
                        outputs=["imgDiamPath","imgRadPath","imgNpCPath"]
                    ),
                    Step(
                        lambda g: g.diameter(),
                        args=["graph"],
                        outputs=["diameter"]
                    ),
                    Step(
                        evaluation.diameters,
                        args=["partition","imgDiamPath"],
                        outputs=["diameters"]
                    ),
                    Step(
                        evaluation.radii,
                        args=["partition","imgRadPath"],
                        outputs=["radii"]
                    ),
                    Step(
                        evaluation.optimalWorkerCountByDiameters,
                        args=["partition"],
                        outputs=["optimalNWorkers"]
                    ),
                    Step(
                        evaluation.nodesPerCommunity,
                        args=["graph","imgNpCPath"],
                        outputs=["nodesPerCommunity"]
                    )
                ],name="Graph & Community metrics"),
                Pipeline([
                    Step(
                        lambda g:np.logspace(1, math.log10(len(g.vs)//2), 5, endpoint=True,dtype=int),
                        args=["graph"],
                        outputs=["nWorkers"],
                        read_only_outputs=set("nWorkers")
                    ),
                    MetaStep(
                        meta.split,
                        params={"key":"nWorkers"}
                    )
                ],name="nWorkers spacing"),
                Step(
                    util.add_params,
                    params=PG({"nVoters":10,"voterSeed":range(1),
                               "numGenFunction":numbering.generateArrangementNumber
                              }),
                    outputs=["nVoters","voterSeed","numGenFunction"]
                ),
                Step(
                    numbering.orderBasedWorkerSelection,
                    args=["graph","nVoters","nWorkers","voterSeed","numGenFunction"],
                    outputs=["workers"],
                    read_only_outputs=set("workers")
                ),
                Pipeline([
                    Step(
                        lambda fp,nw,vs: (imgDirPath+fp.split(".")[0].split("/")[-1]+"_Graph_{}w{}s.png".format(nw,vs),
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_ClustDist_{}w{}s.png".format(nw,vs),
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_InClustDist_{}w{}s.png".format(nw,vs),
                                    imgDirPath+fp.split(".")[0].split("/")[-1]+"_workersPerCom_{}w{}s.png".format(nw,vs),
                                   ),
                        args=["filepath","nWorkers","voterSeed"],
                        outputs=["imgPath","imgCDPath","imgInCDPath","imgWpCPath"]
                    ),
                    #Step(
                    #    community_detection.draw.emphasizeWorkers,
                    #    args=["graph","workers"],
                    #    outputs=["graph"]
                    #),
                    #Step(
                    #    community_detection.draw.drawGraph,
                    #    args=["graph","imgPath"]
                    #),
                    Step(
                        evaluation.workerDistances,
                        args=["graph","workers","imgCDPath"],
                        outputs=["workerDistances"]
                    ),
                    Step(
                        evaluation.workerInClusterDistances,
                        args=["graph","partition","workers","imgInCDPath"],
                        outputs=["workerComDistances"]
                    ),
                    Step(
                        evaluation.workersPerCommunity,
                        args=["graph","workers","imgWpCPath"],
                        outputs=["workersPerCommunity"]
                    ),
                ],name="Workers metrics"),
                [
                    Pipeline([
                        Step(
                            util.add_params,
                            params=PG({"maxDist":range(1,4)}),
                            outputs=["maxDist"]
                        ),
                        Step(
                            lambda mD,nw,vs,fp:"{}{}_{}-closecliquesGraph_{}w{}s.png".format(
                                                    imgDirPath,fp.split(".")[0].split("/")[-1],mD,nw,vs),
                            args=["maxDist","nWorkers","voterSeed","filepath"],
                            outputs=["cliqueImgPath"]
                        ),
                        Step(
                            evaluation.closenessCliques,
                            args=["graph", "maxDist", "workers"],#, "cliqueImgPath"],
                            outputs=["candidates"]
                        )
                    ],name="Close workers cliques"),
                    Pipeline([
                        Step(
                            lambda d:list(range(4,d+1)),
                            args=["diameter"],
                            outputs=["minDist"],
                        ),
                        MetaStep(
                            meta.split,
                            params={"key":"minDist"}
                        ),
                        Step(
                            lambda mD,nw,vs,fp:"{}{}_{}-farcliquesGraph_{}w{}s.png".format(
                                                    imgDirPath,fp.split(".")[0].split("/")[-1],mD,nw,vs),
                            args=["minDist","nWorkers","voterSeed","filepath"],
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
                    params={"keys":["graph","partition"]}
                )
],name="MainWorkflow")