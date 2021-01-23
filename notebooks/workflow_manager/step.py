from workflow_manager.data import Data
from workflow_manager.temporizer import *
import inspect
import types
import copy
import datetime
import itertools
import progressbar
import os
class Step():
    """
    Represents indivisible actions in the workflow. 
    A single step instance can define different executions variants which will be executed in parallel and generate their own dataspace.
    """
    def __init__(self, function, args=None, nargs=None, outputs=None, read_only_outputs=set(), params=[{}], keep_inputs=True, name="Step", export_path=None):
        """
        function: a callable function or a list of callables. Each one will be executed in *run* and each one will generate a data variant;
        args: positional arguments of the function;
        nargs: named arguments of the function;
        outputs: key strings where to store function outputs.
        read_only_outputs:  key strings set (only keys present in *outputs*) which declares which outputs shan't be modified later in execution.
                            Prevents unnecessary copies of large read_only data;
        params: named function arguments which are defined in the workflow itself (not results from execution) and won't be stored in the execution data;
        keep_inputs: boolean defining whether to keep inputs in the data container after the step execution;
        name: a name for that step;
        export_path: a string defining a filepath to output the data. This string may contain variable field which will be resolved during execution. See *Data.to_file* for more infos.
        """
        
        super().__init__()

        assert function is not None and (callable(function) or type(function) is list)
        if type(function) is list:
            self.function = function
        else:
            self.function = [function]

        assert type(name) is str
        self.name=name

        self.args=[]
        if args is not None:
            if type(self.args) is str:
                self.args.append(args)
            else:
                assert type(self.args) is list or type(self.args) is tuple
                for input_key in args:
                    assert type(input_key) is str
                self.args.extend(args)

        self.nargs={}
        if nargs is not None:
            assert type(self.nargs) is set
            for input_key in nargs:
                assert type(input_key) is str
            self.args.extend(nargs)

        self.outputs=None
        if outputs is not None:
            self.outputs=[]
            if type(outputs) is str:
                self.outputs.append(outputs)
            else:
                assert type(outputs) is list or type(outputs) is tuple
                for input_key in outputs:
                    assert type(input_key) is str
                self.outputs.extend(outputs)

        assert type(read_only_outputs) is set and (read_only_outputs == set() or all([type(output) is str for output in read_only_outputs])) 
        self.read_only_outputs=read_only_outputs

        assert type(params) is dict or type(params) is list or isinstance(params, types.GeneratorType)
        if type(params) is dict:
            self.params=[params]
        else:
            self.params=list(params)
        assert type(keep_inputs) is bool
        self.keep_inputs=keep_inputs

        assert export_path is None or type(export_path) is str
        self.export_path = export_path

    def run(self, data_containers, n_jobs=-1):
        print(self)

        variants_containers=[]

        for param_variant in self.params:
            args=[]
            nargs={}
            for param_key in param_variant:
                nargs[param_key]=param_variant[param_key]

            # Running function and collecting outputs
            for funct in self.function:

                output = None
                if self.args == [] and self.nargs=={}:
                    output = funct(*args, **nargs)
                    if self.outputs is not None:
                        # Storing outputs in container
                        if type(output) is tuple:
                            assert len(output) == len(self.outputs)
                        elif output is not None:
                            assert len(self.outputs)==1
                            output=[output]
                # Collecting input values from container
                for data_container in progressbar.progressbar(data_containers):
                    if type(data_container) is str:
                        data_container=Data.from_file(data_container)

                    for input_key in self.nargs:
                        nargs[input_key]=data_container[input_key]

                    args=[]
                    for input_key in self.args:
                        args.append(exec_just_in_time(data_container[input_key]))

                    funct_container=Data.rel_deep_copy(data_container)

                    if self.args != [] or self.nargs!={}:
                        output = funct(*args, **nargs)

                    if not self.keep_inputs:
                        for input_key in self.args:
                            if input_key in funct_container.read_only:
                                funct_container.read_only.remove(input_key)
                            funct_container.pop(input_key)
                        for input_key in self.nargs:
                            if input_key in funct_container.read_only:
                                funct_container.read_only.remove(input_key)
                            funct_container.pop(input_key)

                    if self.outputs is not None:
                        if self.args != [] or self.nargs!={}:
                            # Storing outputs in container
                            if type(output) is tuple:
                                assert len(output) == len(self.outputs)
                            elif output is not None:
                                assert len(self.outputs)==1
                                output=[output]

                        for index, output_key  in enumerate(self.outputs):
                            funct_container[output_key]=output[index]
                        
                        funct_container.read_only.union(self.read_only_outputs)

                    # Log the specific step infos in Data container
                    mem_params=self.params
                    mem_functs=self.function
                    self.params=param_variant
                    if isinstance(funct, types.LambdaType) and funct.__name__ == "<lambda>":
                        self.function=["lambda"]
                    else:
                        self.function=[funct]
                    step_variant=copy.deepcopy(self)
                    self.params=mem_params
                    self.function=mem_functs
                    funct_container.append_step(step_variant)

                    if self.export_path is not None:
                        funct_container.to_file(self.export_path)

                    if n_jobs!=-1:
                        funct_container=funct_container.to_file("Data_"+funct.__name__.replace("<","").replace(">",""))

                    variants_containers.append(funct_container)
        [os.remove(data_container) for data_container in data_containers if type(data_container) is str]
        return variants_containers

    def __repr__(self):
        return "Step(function = "+str([funct.__name__ if callable(funct) else funct for funct in self.function])+", args = "+ str(self.args) + ", nargs = " + str(self.nargs) + ", outputs = "+ str(self.outputs) + ", params = " + str(self.params) + ", keep_inputs = "+str(self.keep_inputs)+", name = "+self.name+")"