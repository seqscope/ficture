# Install

```
git clone -b stable git@github.com:seqscope/ficture.git
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

For this branch (`stable`), we run FICTURE as individual python scripts, so no installation is needed.
