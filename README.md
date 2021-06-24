# Helsinki Machine Learning Project Template
> This template contains all necessary tools for a convenient ML DevOps pipeline, open source. You can use it as a template for your ML or analytics project


```python
#hide #
#from your_lib.core import *
#from ml-project-template.data import *
```

This file will become your README and also the index of your documentation.

## Installing the template

```
git clone [this environment]
git rm -r .git # figure out how to use templates more conveniently
git init
conda create --name [your project env name]`
conda activate [your project env name]
conda install pip
pip install -r requirements.txt
nbdev_install_git_hooks # see if this is necessary
python -m ipykernel install --user --name [your ipython kernel name] --display-name "Python [python version] ([your ipython kernel name])"

```

## How to use

T

Fill me in please! Don't forget code examples:

## Creating doc pages

1. Make sure everything runs smoothly

2. Make sure that settings.ini has correct information

3. You may have to manually edit repo name in `Makefile`, `docs/_config.yml` and `docs/_data/topnav.yml` to match your project

4. Run `nbdev_build_lib & nbdev_build_doc`

5. Push

6. Make your Git repository public

7. View the github pages https://city-of-helsinki.github.io/ml_project_template/ (modify according to your project)

