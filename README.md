# Helsinki Machine Learning Project Template
> Template for 


```python
#hide #
#from your_lib.core import *
#from ml-project-template.data import *
```

```python
# default_exp data
%load_ext lab_black
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

## Basic principles

Explorative coding: code, documentation and results as one

CD/CT/CI

Tidy

Notebook structure: index, data, model, loss, other

## Tools

Black

Sklearn, pandas, numpy, matplotlib, scipy, etc.

Notebooks

Nbdev



## How to use

edit the notebooks `data`, `model` and `loss` directly or replace them with empty templates found in `notebook_templates`

edit `settings.ini` according to your project details



## Creating doc pages

1. Make sure everything runs smoothly

2. Make sure that settings.ini has correct information

3. You may have to manually edit repo name in `Makefile`, `docs/_config.yml` and `docs/_data/topnav.yml` to match your project

4. Run `nbdev_build_lib & nbdev_build_doc`

5. Push

6. Make your Git repository public (github doc pages are only available for public projects. You can also try to build the doc pages locally with jekyll)

7. View the github pages https://city-of-helsinki.github.io/ml_project_template/ (modify according to your project)

