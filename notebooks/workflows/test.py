from workflow_manager.pipeline import Pipeline
from workflow_manager.step import Step
from workflow_manager.parameter_grid import parameter_grid as PG
from workflow_manager.metastep import MetaStep
from workflow_manager import meta
from workflow_manager import util

from libs import community_detection
from libs import order_based_seed_gen

#from workflows.neutralizer import pipeline as neutral_clsf_pipe


from itertools import permutations

superList=["a","b","c","d","e"]

pipeline=Pipeline([
                Step(
                    util.add_params,
                    params=PG({"sublist":permutations(superList,3),"superList":[superList]}),
                    outputs=["sublist", "superList"]
                ),
                Step(
                    order_based_seed_gen.numbering.generateArrangementNumber,
                    args=["sublist", "superList"], 
                    outputs=["number"]
                ),
                Step(
                    print,
                    args=["number","sublist","superList"]
                )
],name="test")