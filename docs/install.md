# Install

```
https://github.com/seqscope/ficture.git
```

It is recommended to create a virtual environment and install FICTURE and its dependencies.

FICTURE is tested with python 3.11 and 3.9, there might be problems with different python versions, but the required packages are farely standard.

It is recommented to use python 3.11 and update pip to the newest version.

(`requirements.txt` is at the root directory of FICTURE)

```
python -m venv /path/to/venv/name
source /path/to/venv/name/bin/activate

pip install -r requirements.txt
```

Install from pypi
```
pip install ficture
```

Or build locally (need to install `build`) from the foot of the cloned repository.
```
python -m build
pip install ./dist/ficture-0.0.2-py3-none-any.whl
```
