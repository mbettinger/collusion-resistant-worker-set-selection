from workflow_manager.data import Data
import os
class MetaStep():
    """
    Enables to manipulate execution dataspaces.
    """
    def __init__(self, function, params=[{}], name="MetaStep"):
        """
        function: a callable function or a list of callables. Each one will be executed in *run* 
        and each one will be applied to the dataspaces in parallel (not sequentially: e.g. calling two functions may double the number of dataspaces);
        params: named function arguments which are defined in the workflow itself (not results from execution) and won't be stored in the execution data;
        name: a name for that step;
        """
        super().__init__()
        assert function is not None and (callable(function) or type(function) is list)
        if type(function) is list:
            self.function = function
        else:
            self.function = [function]

        assert type(name) is str
        self.name=name

        assert type(params) is dict or type(params) is list or isinstance(params, types.GeneratorType)
        if type(params) is dict:
            self.params=[params]
        else:
            self.params=params

    def run(self, data_containers, n_jobs=-1):
        print(self)
        variants_containers=[]

        for param_variant in self.params:
            nargs={}
            for param_key in param_variant:
                nargs[param_key]=param_variant[param_key]

            temp_containers=[Data.from_file(data_container) if type(data_container) is str else data_container for data_container in data_containers]

            # Running function and collecting outputs
            for funct in self.function:
                funct_container=funct(temp_containers, **nargs)
                if n_jobs!=-1:
                    funct_container=[container.to_file("Data_"+funct.__name__.replace("<","").replace(">","")+str(index)) for index, container in enumerate(funct_container)]
                variants_containers.extend(funct_container)
        [os.remove(data_container) for data_container in data_containers if type(data_container) is str]
        return variants_containers

    def __repr__(self):
        return "MetaStep(function = "+str([funct.__name__ for funct in self.function])+", params = "+str(self.params)+", name = "+self.name+")"