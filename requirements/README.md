## Detailed Install Instructions

### Local Environment

To setup a local development environment for the project, simply create a new Conda environment using
[`environment.yml`](environment.yml). Creating a new Conda environment from a file can be done using the following
command (from the root folder of the project):
```shell script
conda env create -f requirements/environment.yml
conda activate castor
```
Unless you manually edit the first line (`name: castor`) of the file, the environment will be named `castor` by default.

### Compute Canada Environment

Here is a brief step by step description of how to setup a working environment for the project on Compute Canada's
servers. For more information on how to configure virtual environments on Compute Canada's servers,
please refer to their own
[documentation on the subject](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment).

On Compute Canada's servers, the recommended tool to manage virtual environments is Python's own virtualenv, rather
than Conda. It is also recommended to use packages compiled specially by Compute Canada for the servers' architectures,
instead of generic packages automatically downloaded by pip. This last recommendation explains why some of the
following commands add options that are not generally seen when configuring local virtual environments.

When using virtualenv, it is necessary to first create an environment like below. Note that it is important to first
load the Python module to ensure that the virtual environment's base Python version is the appropriate one.
```shell script
module load python/3.9
virtualenv --no-download {path to virtual env}
```

After the virtual environment is created, it is necessary to activate it and update the base environment.
```shell script
source {path to virtual env}/bin/activate
pip install --upgrade --no-index pip setuptools wheel
```

Afterwards, the environment's packages can be installed from the custom requirements file.
```shell script
pip install -r requirements/computecanada.txt
```

Do not forget to also install the project's packages, which will allow to import modules from the project, using the
following commands:
```shell script
pip install -e ./vital -e .
```
> NOTE: You should be the one to decide whether to install the project's packages in editable mode (`-e`) or not,
> depending on whether you plan to reuse the environment with varying versions of the code (i.e. installing it to the
> shared filesystem) or use it only once (i.e. installing it directly on a compute node inside a job).
