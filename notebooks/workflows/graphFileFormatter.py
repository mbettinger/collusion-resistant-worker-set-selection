from workflow_manager.pipeline import Pipeline
from workflow_manager.step import Step
from workflow_manager.parameter_grid import parameter_grid as PG
from workflow_manager.metastep import MetaStep
from workflow_manager import meta
from workflow_manager import util

from libs import community_detection
from libs import order_based_seed_gen
from libs import preprocessing
from itertools import permutations

dirPaths=[#"graphs/tab/",
          "graphs/csv/"
          #,"graphs/hcsv/"
          #,"graphs/ncol/"
         ]
destDirPath="graphs/formatted/"

hcsvPath="graphs/hcsv/"
pipeline=Pipeline([
                Step(
                    util.add_params,
                    params=PG({"dirPath":dirPaths}),
                    outputs=["dirPath"]
                ),
                Step(
                    preprocessing.getFileNamesInDir,
                    args=["dirPath"],
                    outputs=["fileNames"]
                ),
                Step(
                   lambda fmt:fmt==hcsvPath,
                   args=["dirPath"],
                   outputs=["header"]
                ),
                Step(
                    lambda dP,fnList:[dP+fn for fn in fnList],
                    args=["dirPath","fileNames"],
                    outputs=["filepaths"]
                ),
                Step(
                    lambda fnList:[destDirPath+fn for fn in fnList],
                    args=["fileNames"],
                    outputs=["destFilepaths"]
                ),
                Step(
                    print,
                    args=["filepaths", "destFilepaths"]
                ),
                Step(
                    lambda fpList, dfpList,h: [preprocessing.formatVertexIds(fp,dfpList[idx],h) for idx, fp in enumerate(fpList)],
                    args=["filepaths", "destFilepaths", "header"]
                )
],name="FileFormatter")