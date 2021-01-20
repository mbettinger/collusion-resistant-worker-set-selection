import json
import sys
from types import ModuleType
import copy
import importlib
import pickle

from workflow_manager.data import Data
def run_workflow(workflow="workflows.test", input_data_path=None, output_data_path=None, use_pickle=False, use_plain=False, use_dict=False, use_json=False, n_jobs=-1):
    """
        workflow: the import path to the desired workflow;
        input_data_path: filepath to a file containing data executions;
        output_data_path: filepath where resulting data executions will be written;
        use_pickle: whether to write output in Pickle format;
        use_plain: whether to write output as plain text;
        use_dict: whether to write output as a dictionary (for data which cannot be written as json)
        use_json: whether to write output in json format (data must be json compatible)
    """
    data=[Data({})]
    if input_data_path is not None:
        with open(input_data_path, "rb") as data_file:
            data = pickle.Unpickler(data_file).load()
        
    module = importlib.import_module(workflow)
    data=module.pipeline.run(data, n_jobs)
    data=[Data.from_file(container, rm=True) if type(container) is str else container for container in data]
    if output_data_path is not None:
        if use_pickle:
            with open(output_data_path+".pkl", "wb") as data_file:
                pickle.Pickler(data_file).dump(data)
        if use_plain:
            with open(output_data_path+".data", "w") as data_file:
                data_file.write(str(data))
        if use_dict:
            for index,data_container in enumerate(data):
                with open(output_data_path+str(index)+".dict", "w") as data_file:
                    data_file.write(str(data_container.to_dict()))
        if use_json:
            for index,data_container in enumerate(data):
                with open(output_data_path+str(index)+".json", "w") as data_file:
                    json.dump(data_container.to_dict(),data_file, ensure_ascii=False)
    return data

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-wf", "--workflow", dest="workflow", type=str, default="workflows.test",
                        help="import workflow with Python syntax", metavar="import.path")
    parser.add_argument("-i", "--input-data", dest="input_data_path", type=str,
                        help="filepath to read initial data", metavar="filepath")
    parser.add_argument("-o", "--output-data", dest="output_data_path", type=str,
                        help="filepath to write execution data", metavar="filepath")
    parser.add_argument("-p", "--pickle", dest="use_pickle", action="store_true",
                        help="writes output in pickle binary if True")
    parser.add_argument("-t", "--text", dest="use_plain", action="store_true",
                        help="writes output in plain text if True")
    parser.add_argument("-d", "--dict", dest="use_dict", action="store_true",
                        help="writes output data as a dictionary if True")
    parser.add_argument("-j", "--json", dest="use_json", action="store_true",
                        help="writes output data as a json if True")
    parser.add_argument("-n", "--n_jobs", dest="n_jobs", type=int, default=-1,
                        help="determines maximum number of parallel executions")
    args = parser.parse_args()
    run_workflow(args.workflow, args.input_data_path, args.output_data_path, args.use_pickle, args.use_plain, args.use_dict, args.use_json, args.n_jobs)