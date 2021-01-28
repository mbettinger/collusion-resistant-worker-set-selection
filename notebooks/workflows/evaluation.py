from workflow_manager.pipeline import Pipeline
from workflow_manager.step import Step
from workflow_manager.parameter_grid import parameter_grid as PG
from workflow_manager.metastep import MetaStep
from workflow_manager import meta
from workflow_manager import util

from libs import evaluation

import os

import numpy as np
import math

dirPath="outputs/"
imgDirPath="graphs/img/"        

pipeline=Pipeline([
                Step(
                    util.add_params,
                    params={"fsp":{"voterSeed":"NA","comDet":"community_label_propagation","withBoundary":"True"}},
                    outputs=["fileSelectionParams"]
                ),
                Step(
                    parametrizedFileSelection,
                    args=["fileSelectionParams"],
                    outputs=["dataFrame"]
                ),
],name="EvaluationWorkflow")