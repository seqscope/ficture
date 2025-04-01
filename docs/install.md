# Installing FICTURE

## Installing from GitHub repository

To install the latest copy of FICTURE, you can clone the repository from GitHub.

```bash linenums="1"
git clone https://github.com/seqscope/ficture.git
```

It is recommended to create a [virtual environment](https://docs.python.org/3/library/venv.html) and install FICTURE and its dependencies.

FICTURE is tested with python 3.11 and 3.9. While there might be problems with different python versions, the required packages are fairly standard, so we expect it to work with other versions of python as well.

To follow our best practices, it is recommended to use python 3.11 and update pip to the newest version.

(`requirements.txt` is at the root directory of FICTURE)

Here are example commands to install FICTURE and its dependencies.

```bash linenums="1"
## Create a virtual environment
VENV=/path/to/venv/name   ## replace /path/to/venv/name with your desired path
python -m venv ${VENV}

## Activate the virtual environment
source ${VENV}/bin/activate

## Clone the GitHub repository
git clone https://github.com/seqscope/ficture.git
cd ficture

## Install the required packages
pip install -r requirements.txt

## Install FICTURE locally
pip install -e .
```

Installing FICTURE this way allows you to run the software while making minor changes to the codebase. It also allows you to access the example data and scripts that come with the repository.

## Installing from PyPI

If you want to install FICTURE from the pypi repository, you can do so with the following command.

```bash linenums="1"
pip install ficture
```

Note that the version on PyPI may not be the latest version of FICTURE. Sometimes the `dev` branch main contain some of the latest features that may not have been merged with the `main` branch. 


<!-- Or build locally (need to install `build`) from the foot of the cloned repository.
```
python -m build
pip install ./dist/ficture-0.0.2-py3-none-any.whl
``` -->
