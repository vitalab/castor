<div align="center">

# CASTOR

Welcome to the code repository for projects related to the *CArdiac SegmenTation with cOnstRaints* (CASTOR) project.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI: Code Format](https://github.com/nathanpainchaud/castor/actions/workflows/code-format.yml/badge.svg?branch=main)](https://github.com/nathanpainchaud/castor/actions/workflows/code-format.yml?query=branch%3Amain)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nathanpainchaud/castor/blob/dev/LICENSE)

## Publications

[![Journal](http://img.shields.io/badge/IEEE%20TMI-2022-4b44ce.svg)](https://doi.org/10.1109/TMI.2022.3173669)
[![Paper](http://img.shields.io/badge/paper-arxiv.2112.02102-B31B1B.svg)](https://arxiv.org/abs/2112.02102)

[![Journal](http://img.shields.io/badge/IEEE%20TMI-2020-4b44ce.svg)](https://doi.org/10.1109/TMI.2020.3003240)
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.08825-B31B1B.svg)](https://arxiv.org/abs/2006.08825)

[![Conference](http://img.shields.io/badge/MICCAI-2019-4b44ce.svg)](https://doi.org/10.1007/978-3-030-32245-8_70)
[![Paper](http://img.shields.io/badge/paper-arxiv.1907.02865-B31B1B.svg)](https://arxiv.org/abs/1907.02865)

</div>

## Description
This is a project that constrains the predictions of automatic cardiac segmentation *a posteriori* to guarantee useful
properties, i.e. anatomical validity and temporal consistency.

To help you follow along with the organization of the repository, here is a summary of each major package's purpose:

- [apps](castor/apps): interactive applications, either graphical or command line, that help to inspect data and/or
results.

- [metrics](castor/metrics): metrics specific to our cardiac images that are not part of the
traditional libraries. The metrics are first divided by datasets on which they apply (see
[ACDC](castor/metrics/acdc) or [CAMUS](castor/metrics/camus)). Afterwards,
the metrics are divided according to what they're computing (e.g. clinical or anatomical metrics).

- [results](castor/results): API and executable scripts for processing results during the
evaluation phase.

- [requirements](requirements): conda and pip requirement files, along with detailed instructions on how to setup a
working environment in different conditions (local, cluster, etc.)

- [vital](https://github.com/nathanpainchaud/vital/tree/dev/vital): a separate repository (included as a
[git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)), of generic PyTorch modules, losses and metrics
functions, and other tooling (e.g. image processing, parameter groups) that are commonly used. Also contains the code
for managing specialized medical imaging datasets, e.g. ACDC, CAMUS.


## How to Run

### Install
First, download the code and setup a virtual environment:
```shell script
# clone project
git clone --recurse-submodules https://github.com/nathanpainchaud/castor.git

# set-up project's environment and activate it
conda env create -f requirements/environment.yml
conda activate castor
```
To test that the project was installed successfully, you can try the following command from the Python REPL:
```python
# now you can do:
from castor import Whatever
```
NOTE: All following commands in this README (and other READMEs for specific packages), will assume you're working from
inside the Conda environment you've just setup.

### Data
Next, navigate to the data folder for either the
[ACDC](https://github.com/nathanpainchaud/vital/tree/dev/vital/data/acdc) or
[CAMUS](https://github.com/nathanpainchaud/vital/tree/dev/vital/data/camus) dataset and follow the instructions on how
to setup the datasets:
- [ACDC instructions](https://github.com/nathanpainchaud/vital/blob/dev/vital/data/acdc/README.md#dataset-generator)
- [CAMUS instructions](https://github.com/nathanpainchaud/vital/blob/dev/vital/data/camus/README.md#cross-validation)

### Configuring a Run
This project uses Hydra to handle the configuration of the
[`vital` runner script](https://github.com/nathanpainchaud/vital/tree/dev/vital/vital/runner.py). To understand how to
use Hydra's CLI, refer to its [documentation](https://hydra.cc/docs/intro/). For this particular project, preset
configurations for various parts of the `vital` runner pipeline are available in the [config package](castor/config).
These files are meant to be composed together by Hydra to produce a complete configuration for a run.

Below we provide examples of how to run some basic commands using the Hydra CLI:
```shell script
# list generic trainer options and datasets on which you can train
castor-runner -h

# select high-level options of task to run, and architecture and dataset to use
castor-runner task=<TASK> task.model=<MODEL> data=<DATASET>

# display the available configuration options for a specific combination of task/model/data (e.g Enet on CAMUS)
castor-runner task=segmentation task.model=enet data=camus -h

# train and test a specific system (e.g beta-VAE on CAMUS)
castor-runner task=autoencoder task.model=beta-vae data=camus data.dataset_path=<DATASET_PATH> [optional args]

# test a previously saved system (e.g. beta-VAE on CAMUS)
castor-runner task=autoencoder task.model=beta-vae data=camus data.dataset_path=<DATASET_PATH> \
  ckpt=<CHECKPOINT_PATH> train=False

# run one of the fully pre-configured 'experiment' from the `config/experiment` folder (e.g. Enet on CAMUS)
castor-runner +experiment=camus/enet
```

To create your own pre-configured experiments, like the one used in the last example, we refer you to Hydra's own
documentation on the subject.

### Tracking experiments
By default, Lightning logs runs locally in a format interpretable by
[Tensorboard](https://www.tensorflow.org/tensorboard/).

Another option is to use [Comet](https://www.comet.ml/) to log experiments, either online or offline. To enable the
tracking of experiments using Comet, simply use one of the pre-built Hydra configuration for Comet. The default
configuration is for Comet in `online` mode, but you can use it in `offline` mode by selecting the corresponding config
file when launching the [`vital` runner script](https://github.com/nathanpainchaud/vital/tree/dev/vital/vital/runner.py):
```bash
castor-runner logger=comet/offline ...
```
To configure the Comet API and experiment's metadata, Comet relies on either i) environment variables (which you can set
in a `.env` that will automatically be loaded using `python-dotenv`) or ii) a [`.comet.config`](.comet.config)` file. For
more information on how to configure Comet using environment variables or the config file, refer to
[Comet's configuration variables documentation](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).

An example of a `.comet.config` file, with the appropriate fields to track experiments online, can be found
[here](.comet.config). You can simply copy the file to the directory of your choice within your project (be sure
not to commit your Comet API key!!!) and fill the values with your own Comet credentials and workspace setup.

> NOTE: No change to the code is necessary to change how the `CometLogger` handles the configuration from the
> `.comet.config` file. The code simply reads the content of the `[comet]` section of the file and uses it to create a
> `CometLogger` instance. That way, you simply have to ensure that the fields present in your configuration match the
> behavior you want from the `CometLogger` integration in Lighting, and you're good to go!

## How to Contribute

### Environment Setup
When installing a python environment as [described above](#install), the resulting environment is already fully
configured to start contributing to the project. There's nothing to change to get coding!

### Version Control Hooks
Before first trying to commit to the project, it is important to setup the version control hooks, so that commits
respect the coding standards in place for the project. The [`.pre-commit-config.yaml`](.pre-commit-config.yaml) file
defines the pre-commit hooks that should be installed in any project contributing to the `vital` repository. To setup
the version control hooks, run the following command:
```shell script
pre-commit install
```

> NOTE: In case you want to copy the pre-commit hooks configuration to your own project, you're welcome to :)
> The configuration for each hook is located in the following files:
> - [isort](https://github.com/timothycrosley/isort): [`pyproject.toml`](./pyproject.toml), `[tool.isort]` section
> - [black](https://github.com/psf/black): [`pyproject.toml`](./pyproject.toml), `[tool.black]` section
> - [flake8](https://gitlab.com/pycqa/flake8): [`setup.cfg`](./setup.cfg), `[flake8]` section
>
> However, be advised that `isort` must be configured slightly differently in each project. The `src_paths` tag
> should thus reflect the package directory name of the current project, in place of `vital`.


## References
If you find this code useful, please consider citing the paper implemented in this repository relevant to you from the
list below:
```bibtex
@article{painchaud_echocardiography_2022,
    title = {Echocardiography {Segmentation} {with} {Enforced} {Temporal} {Consistency}},
    doi = {10.1109/TMI.2022.3173669},
	journal = {IEEE Transactions on Medical Imaging},
	author = {Painchaud, N. and Duchateau, N. and Bernard, O. and Jodoin, P.-M.},
	year = {2022},
}

@article{painchaud_cardiac_2020,
	title = {Cardiac {Segmentation} {With} {Strong} {Anatomical} {Guarantees}},
	volume = {39},
	copyright = {All rights reserved},
	issn = {1558-254X},
	doi = {10.1109/TMI.2020.3003240},
	number = {11},
	journal = {IEEE Transactions on Medical Imaging},
	author = {Painchaud, N. and Skandarani, Y. and Judge, T. and Bernard, O. and Lalande, A. and Jodoin, P.-M.},
	month = nov,
	year = {2020},
	pages = {3703--3713},
}
```
