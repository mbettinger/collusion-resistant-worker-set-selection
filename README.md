# Collusion-Resistant Worker Set Selection for Transparent and Verifiable Blockchain-based Voting

This repository contains the source code used in [Collusion-Resistant Worker Set Selection for Transparent and Verifiable Blockchain-based Voting by M.BETTINGER & L.BARBERO et al.]() (review pending).

## Environment
This code can be run inside a provided Docker container with all necessary dependencies. 

If one wanted to run this code without Docker, one shall manually install all dependencies specified in the [Dockerfile](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/Dockerfile).

## Setup

The more straightforward way of running the code (on Windows) in its appropriate environment is to run the [jupyter.bat](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/jupyter.bat) script. This script executes the following tasks:

* Run [docker_command.bat](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/docker_command.bat), i.e build and run the Docker Container, as well as launching a Jupyter Notebook inside said container;
* Wait and retrieve the notebook's URL (stored afterwards in [jupyter_path.txt](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/jupyter_path.txt));
* Launch a web navigator on that URL.

Remarks:
* Should the URL polling timing be off, resulting in the navigator opening a default webpage, one can either execute jupyter.bat again (without stopping the container), or use the URL in [jupyter_path.txt](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/jupyter_path.txt) once it has been written there.
* Several paths depend on user-defined installation paths and shall be adapted accordingly: 
    * The web navigator's executable path ([jupyter.bat](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/jupyter.bat) line 7)
    * The project's root-path host-to-container correspondance ([docker_command.bat](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/docker_jupyter/docker_command.bat), line 2 after '-v')

Should the script be unusable as is (e.g due to OS incompatibilities), one shall reproduce the aforementioned steps manually or adapt said script.

## Contents and features
This repository is structured in the following manner:
* ./docker_jupyter: files used in the Setup section, to start the needed environment;
* ./notebooks: the code used in the paper's experiments:
    * libs: python modules used in the various experiments;
    * workflow_manager: the version of [workflow_manager](https://github.com/mbettinger/workflow-manager) by BETTINGER Matthieu as it was available at the time of ongoing experiments (for more information about that module, please refer to the [README.md](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/notebooks/workflow_manager/README.md) in that folder);
    * workflows: pipelines of experiments run using workflow_manager. Important workflows are:
        * graphFileFormatter.py: formats graphfiles in input to a ncol format;
        * main.py: end-to-end pipeline as defined in our paper's experimental protocol.
    * \_\_main__.py: entry-point for running workflows. See [main_exec_command.txt](https://github.com/mbettinger/collusion-resistant-worker-set-selection/blob/main/main_exec_command.txt) for a command example, which launches the main pipeline and outputs executions in json format.
    * experimentation_analysis.ipynb: Jupyter notebook corresponding to the cross-execution analyses phase in our experimental protocol.
    * graph_tests.ipynb & referendum_blockchain_simulation: Jupyter notebooks used during the prototyping phases, respectively corresponding to Worker Set Selection using the graph as knowledge and using a simulated Vote-Order-based Selection.
        
