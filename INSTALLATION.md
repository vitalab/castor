## Additional Install Instructions

### Local Environment

The installation instructions for setting up local environments for either [deployment](README.md#install) or
[development](README.md#environment-setup) are already provided in the main README file, and you should refer to them.

###  Digital Research Alliance of Canada Environment

Here is a brief step by step description of how to setup a working environment for the project on the clusters managed
by the Digital Research Alliance of Canada (DRAC). For more information on how to configure virtual environments on
DRAC's servers, please refer to their own [documentation on the subject](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment).

On clusters managed by the DRAC, the recommended tool to manage virtual environments is Python's own virtualenv. It is
also recommended to use packages compiled by them specially for the servers' architectures, instead of generic packages
automatically downloaded by pip. This last recommendation explains why some of the following commands add options that
are not generally seen when configuring local virtual environments.

When using virtualenv, it is necessary to first create an environment like below. Note that it is important to first
load the Python module to ensure that the virtual environment's base Python version is the appropriate one.
```shell script
module load python/3.10
virtualenv --no-download {path to virtual env}
```

After the virtual environment is created, it is necessary to activate it and update the base environment.
```shell script
source {path to virtual env}/bin/activate
pip install --no-index --upgrade pip
```

Afterwards, the environment's packages can be installed from the custom requirements file.
```shell script
pip install -r alliancecan-requirements.txt
```

Do not forget to also install the project's packages, which will allow to import modules from the project, using the
following commands:
```shell script
pip install -e ./vital --no-dependencies -e . --no-dependencies
```
> **Warning**
> You should be the one to decide whether to install the project's packages in editable mode (`-e`) or not, depending on
> whether you plan to reuse the environment with varying versions of the code (i.e. installing it to the shared
> filesystem) or use it only once (i.e. installing it directly on a compute node inside a job).
