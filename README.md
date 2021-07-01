# Helsinki Machine Learning Project Template
> Template for open source ML and analytics projects.


This file will become your README and also the index of your documentation.

## About This Template

This is a template for open source ML and analytics projects.
It is best used as part of MLOps practices, for continuously exploring with data, developing, training and deploying your models.
It ties together all components required for reproducible and explainable data science, CD/CT/CI tools, unit testing and more. 

All you need to do for starting to work on your data project is to install the template and hook it to your data source and application.

You are free to modify this teplate for the needs of your project.

For general coding best practices, refer to [dev.hel.fi](https://dev.hel.fi/) where applicable.

In addition to those, the template follows four fundamental principles.
Remember, none of these are strict and you are free to deviate for achieving the best results for you.

##  Four Principles of the Template

### 1. Exploratory Programming

We want to keep our code, documentation and results together, seamlessly.
That's why we use jypyter notebooks as the core of our development.

Actually, even this page was generated from a notebook!

The notebooks are enhanced `nbdev` to export code to modules, create doc pages, run tests, handle notebook version control etc.
In addition, the notebooks can be parameterized with `papermill` and piped with `snakemake` for automated use.

Some reasoning for those who are not yet convinced:

- In data projects, the code efficiency is irrelevant. The thinking time is what matters.
- It is simply impractical to create poorly documented notebook. With notebook development, your code is always well documented.
- How many of you actually test your ML code? Clean, running notebooks are the tests, and with `nbdev` unit tests are easy to include. 
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
3. Pipeline / orchestrate
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
2. Compose simple functions with the pipe \*
3. Embrace functional programming \*\*
4. Are designed for humans

\* A pipe operator takes the expression before it and gives it as the first argument to the expression after it,
assuming the expression on the right is a function call. In addition, pipe functions should perform a transformation or a side-effect to their input,
but never both. This allows composition of simple functions, e.g.  `model >> fit >> predict`.
Python does not have a native pipe operator such as `%>%` in R tidyverse,
but Python class functions can be written in a pipe-like way.
More on this in the `model` notebook.

Read more on tidy tools from [tidy tools manifesto](https://cran.r-project.org/web/packages/tidyverse/vignettes/manifesto.html).

\** Python is not a functional programming language, but it can be written in functional style.

Read more on functional programming with python from this Stack Abuse [article](https://stackabuse.com/functional-programming-in-python).


### 4. Data, Model & Loss - The Three Components of Machine Learning

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

## Example Project

We wanted to make this template easy to approach.
That's why we included a demo, that it is built around.

If you'd like to skip the demo, and get right into action, you can replace the notebooks `data`, `model` and `loss` with clean copies under `notebook_templates`.

The demo is an example ML project on automating heart disease diagnosis with logistic regression [UCI heart disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).
The dataset contains missing values, and is thus great for demonstrating some light data wrangling.
The demo is only meant for showcasing how the template joins together different tools and steps.

## Installing the template

### On your GitHub homepage:

1. Sign into your GitHub account
2. In the top right corner of the homepage, click the '+'-button
3. Select 'Import repository'
4. Under 'Your old repository's clone URL' copy the clone url of this repository: `git@github.com:City-of-Helsinki/ml_project_template.git`
5. Select owner of the repo, if you want to use the template for your organization.
Also define your project publicity (you can change this later, in most cases you'll want to begin with a private repo).
6. Click 'Begin import'

This will create a new repository for you copying everything from this template, including the commit history.

### On your computing environment:



1. Clone the repository: `git clone git@github.com:City-of-Helsinki/ml_project_template.git`
2. (Optional) Clear the commit history:
```
git -rf .git
git init
git remote add origin git@github.com:<YOUR ACCOUNT>/<YOUR REPOS>.git
```
3. Create virtual environment (You may change this according to your system and preferences, designed for Helsinki developers):
```
conda create --name [your project env name]`
conda activate [your project env name]
conda install pip
pip install -r requirements.txt
nbdev_install_git_hooks
python -m ipykernel install --user --name [your ipython kernel name] --display-name "Python [python version] ([your ipython kernel name])"
```
4. Configure your git user name and email adress (one of those added to your git account) if you haven't done it already:
```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```
5. Make initial commit:
```
git add .
git commit -m "Initial commit"
```
6. Push (overwriting commit history if you deleted it): `git push -u --force origin master`



## How to use

1. Install this template as basis of your new project (see above)

2. Check out the notebooks, and play around a bit to see that your installation works and you understand the template structure

3. Edit `settings.ini`, `docs/_config.yml` and `docs/_data/topnav.yml` according to your project details

4. Edit the notebooks `data`, `model` and `loss` directly or replace them with empty notebooks clean of the code examples found in the folder `notebook_templates`.
This notebook (`index`) will become the README source of your project.

5. You may delete `ml_project_template`, `notebook_templates` folders and the extra notebook `plot` if you no longer need them. 

6. Save your notebooks and call `nbdev_build_lib` to build python modules of your notebooks - needed if you want to share code between notebooks or create a modules.
Remember to do this if you want to rerun your workflow after making changes to exportables. 

7. Save your notebooks and call `nbdev_build_docs` to create doc pages based on your notebooks (see below).
This will also create README.md file based on this notebook.
If you want to host your project pages on GitHub, you will have to make your project public.
You can also build the pages locally with `jekyll`.

8. Remember to track your changes with git! Some useful commands:

See which files have changes since the last commit: `git status` 

Add files to a commit: `git add [file names/paths separated by whitespace ' ']`

Create commit: `git commit -m [short description of the changes you made]`

To use git with remote repository, you must create an ssh key and upload it to your git profile settings.
See [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) how to do it.
Then, you can push the commits to remote repository, where they are safe and allow collaborative work on the project:

Push commit to remote repository (GitHub server): `git push origin -u` 

In addition, there are many fancy features for git that enable collaborative work, debugging, automated testing and other crazy things, but there are other sources for that.

**Now, begin working with your ML project!**



## Contributing
