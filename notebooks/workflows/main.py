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
                    util.add_params,
                    params=PG({"nVoters":10,"nWorkers":10,"voterSeed":10,
                               "numGenFunction":[numbering.generateArrangementNumber]
                              }),
                    outputs=["nVoters","nWorkers","voterSeed","numGenFunction"]
                ),
                Step(
                    numbering.orderBasedWorkerSelection,
                    args=["graph","nVoters","nWorkers","voterSeed","numGenFunction"],
                    outputs=["workers"],
                ),
                Step(
                    print,
                    args=["workers"]
                ),
                Step(
                    community_detection.worker_selection.findCommunities,
                    args=["graph"],
                    outputs=["graph","partition"]
                ),
                Step(
                    lambda fp: (imgDirPath+fp.split(".")[0].split("/")[-1]+".png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_ClustDist.png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_InClustDist.png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_diameters.png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_radii.png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_nodesPerCom.png",
                                imgDirPath+fp.split(".")[0].split("/")[-1]+"_workersPerCom.png",
                               ),
                    args=["filepath"],
                    outputs=["imgPath","imgCDPath","imgInCDPath","imgDiamPath","imgRadPath","imgNpCPath","imgWpCPath"]
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
                Step(
                    evaluation.workerDistances,
                    args=["graph","workers","imgCDPath"]
                ),
                Step(
                    evaluation.diameters,
                    args=["partition","imgDiamPath"]
                ),
                Step(
                    evaluation.radii,
                    args=["partition","imgRadPath"]
                ),
                Step(
                    evaluation.workerInClusterDistances,
                    args=["graph","partition","workers","imgInCDPath"]
                ),
                Step(
                    evaluation.nodesPerCommunity,
                    args=["graph","imgNpCPath"]
                ),
                Step(
                    evaluation.workersPerCommunity,
                    args=["graph","workers","imgWpCPath"]
                ),
                Step(
                    evaluation.optimalWorkerCountByDiameters,
                    args=["partition"],
                    outputs=["optimalNWorkers"]
                ),
],name="MainWorkflow")