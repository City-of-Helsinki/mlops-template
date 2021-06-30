# Helsinki Machine Learning Project Template
> Template for open source ML and analytics projects.


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

## About This Template

This is a template for open source ML and analytics projects.
It is best used as part of MLOps practices, for continuously exploring with data, developing, training and deploying your models.
It ties together all components required for reproducible and explainable data science, CD/CT/CI tools, unit testing and more. 
All you need to do for starting to work on your data project is to install the template and hook it to your data source and application.

You are free to modify this teplate for the needs of your project.
It is meant to get you started, and you are free to modify it to your needs. All feedback is wellcome!

For general coding best practices, refer to [dev.hel.fi](https://dev.hel.fi/) where applicable.

In addition to those, the template follows four fundamental principles:

### 1. Exploratory Programming

We want to keep our code, documentation and results together, seamlessly.
That's why we use jypyter notebooks as the core of our development.

The notebooks are enhanced `nbdev` to export code to modules, create doc pages, run tests etc.
In addition, the notebooks can be parameterized with `papermill` and piped with `snakemake` for automated use.

Some reasoning for those who are not yet convinced:

- In data projects, the code efficiency is irrelevant. The thinking time is what matters.
- It is simply impractical to create poorly documented notebook. With notebook development, your code is always well documented.
- How many of you actually unit test your ML code? Clean, running notebooks **are** the tests. 
- Most data science projects involve multiple stakeholders with various backgrounds and skillsets.
Many of them do not understand code, and even those who do, can not if it is poorly documented, nor can they interpret the results alone.
- If you need to smash your algorithm into a tiny IoT device or ship it to a network of space shuttles,
the work is much easier to accomplish following a well documented demo - built on top of this template

With notebook development you get the right results much faster, and everyone involved can actually understand what is happening.

Read more on exploratory programming with notebooks from [this blog post](https://www.fast.ai/2019/12/02/nbdev/).

Read more on nbdev on their [project pages](https://nbdev.fast.ai/).

### 2. Ease of Reproducibility

Poor reproducibility is a major issue in data science projects, both in the industry and academia, but is often overlooked at.
For the city of Helsinki as a public sector operator, it is unacceptable, although we believe that everyone would benefit from it. 
Each state and decision of a ML model should be reproducible.
A theoretical possibility of recreating a particular result is not enough, if it takes unreasonable efforts to do it.
Good reproducibility equals to ease of reproducibility.

For ease of reproducibility we 

1. Document
2. Seed
3. Pipe
4. Version

everything (Data version control is a topic we are still working on).

### 3. Tidy Data & Tools

Tidy data is easy to handle and understand.

Data is tidy, when:

1. Every column is a variable (feature or label)
2. Every row is an observation (data point).
3. Every cell contains a single value

Read more on tidy data from [tidy data manifesto](https://vita.had.co.nz/papers/tidy-data.html).

Tidy tools makes handling data, programming and creating explainable ML much easier.

Tidy tools:

1. Reuse existing data structures
2. Compose simple functions with the pipe
3. Embrace functional programming
4. Are designed for humans

Read more on tidy tools from [tidy tools manifesto](https://cran.r-project.org/web/packages/tidyverse/vignettes/manifesto.html).

### 4. Data/Model/Loss - The three components of machine learning

The core of this template constitutes of three notebooks: data, model and loss.
These have a running number prefix to emphasize the running order and to improve readability.
Any data project can be resolved by defining these three things.

You might be used to doing all of them in a single script,
but separating makes development, explaining the results and debugging much easier. 

In the **data** notebook, the the data is loaded and cleaned, and a basic analysis may be carried out.
Related functions may also be defined for later use, such as functions for frequent ETL tasks.
You should also create a small toy dataset to develop and test your algorithm with - no,
trust me, your code won't work for the first n+1 times, and running it with the whole dataset will waste so much time!
This is also why we separate between the model and loss notebooks.

In the **model** notebook, the machine learning model (or analytics or simulation) is idealized, defined and tested.
You can begin with scripting, but based on the script you should develop real generalizable and tested code.
This part of the notebooks is the closest to traditional software development it gets: you create clean python code. 

In the **loss** notebook, you will finally fit your model to the whole dataset and evaluate it in action.
Some call might call this step *inference*, others *evaluation*. No matter the name, you evaluate the performance of your model to see if it is ready for production.
If the results are satisfactory, you can then use your code as is, or ship your code to it's destination
(this part depends a lot on the project, so we'll leave it to you to figure it out).

Currently, these notebooks will have to be run manually.
In the next release, we will include additional tools `papermill` and `snakemake`,
and a third notebook `pipe` for automatical excecution of the workflow.

Each notebook is also a basis for a python module, including tests and documentation.
The `nbdev` tool constructs a python module of each of these notebooks, under the folder `[your git project name]` (`ml_project_template` in this template).
This allows you to share code between your notebooks, or even publish a complete python module, while doing all your development in the notebooks. 

To demonstrate the different parts of the template, the template is build around a demo ML project on heart disease dataset.
This dataset was selected for the project, because it contained missing values and required cleanig before using it. 

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

