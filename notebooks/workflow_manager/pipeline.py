from workflow_manager.step import Step
from workflow_manager.metastep import MetaStep
from workflow_manager.data import Data
import copy
class Pipeline():
    """
    Represents a sequence of *steps* to be executed to obtain a result.
    """
    def __init__(self, steps=[], name="Pipeline"):
        super().__init__()

        self.name=name

        assert Pipeline.check_instance_types(steps)
        self.steps=steps

    def run(self,data_containers,n_jobs=-1):
        """
            Executes the sequence of steps/sub-pipelines and returns all data_containers of all execution variants (parallel steps/different parameters/...)
        """
        print(self.name)
        for step in self.steps:
            if type(step) is not list:
                step=[step]

            temp_containers=[Data.rel_deep_copy(container) if type(container) is not str else container for container in data_containers]

            # is_metastep=lambda substep,rec: type(substep) is MetaStep or (type(substep) is Pipeline and any([rec(st, rec) for st in substep.steps]))
            # if n_jobs == -1 or any([is_metastep(substep, is_metastep) for substep in step]):
            data_variants = [step_variant.run(temp_containers,n_jobs) for step_variant in step] # deepcopy for different subworkflow dataspaces
            # else:
            #     assert n_jobs>=1

            #     data_variants=[]
            #     while temp_containers!=[]:
            #         exec_variants,temp_containers = ([step_variant.run(temp_containers[0:min(n_jobs, len(temp_containers))],n_jobs) for step_variant in step],
            #                                         temp_containers[n_jobs:]) # deepcopy for different subworkflow dataspaces
            #         data_variants.extend(exec_variants)
                
            # Flatten containers lists created by variants
            data_containers=[]
            for data_variant in data_variants:
                data_containers.extend(data_variant)

        return data_containers

    @staticmethod
    def check_instance_types(step, recursive=True):
        return isinstance(step, Step) \
            or isinstance(step, MetaStep) \
            or isinstance(step, Pipeline) \
            or (type(step) is list \
                and all([Pipeline.check_instance_types(substep) for substep in step]))
