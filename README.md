# Helsinki Machine Learning Project Template
> Template for open source ML and analytics projects.


```python
%load_ext lab_black
# nb_black if using jupyter
```

## About

This is a git repository\* template for Python-based open source ML and analytics projects.

---
** INFO BOX: Git Repository ** 
\*A git repository is a folder that contains source code and documentation for a software project.
Git is a software for version control and collaborative coding work. 
Git is commonly used from from command prompt (Windows), shell (Linux / Ubuntu) or terminal (MacOS).
It keeps track of all changes you make to your software in a `.gitfile`,
so that you can try out different things without making messy manual back ups.
To learn more about Git, their [homepage](https://git-scm.com/) is a great place to start.
Git is often used with GitHub, a free online service for storing and sharing repositories.
GitHub allows collaborative work, automated testing, hosting project doc pages and other fancy features.
Read more on GitHub on their [homepage](https://github.com/City-of-Helsinki/ml_project_template).

---

This template helps you to develop, test, share and update ML applications, in a collaborative way.
It defines steps and tools of a ML project and helps you to tie them together for easy to use, explainable and reproducible workflow.
You can just replace the examples with your own code, or start from scratch with clean notebooks.
Instructions for using the template are given below, so just keep reading!

ML requires team effort. When working with data, we have three kinds of people.
We havepeople who know code, people who know data, and people who know a problem to solve.
Those who know two of these are rare, not to mention owning all three.
Working with the template enables joint work of application field specialists 
(e.g. healthcare, city planning, finance), researchers, data analysts, engineers & scientists, programmers and other stakeholders.
In essence, the template is a tool for teamwork - *a single person does not have to, and most likely does not even know how to complete all of the steps defined!*.
With code, documentation and results as one, each team member can understand what is going on, and contribute on their behalf.

The template is completely open source and environment agnostic.
It contains examples and instructions to help you through the whole project.
However, we try to keep it light and easy to maintain, and thus the documentation is not exhaustive.
We have added references to original sources so that they are easy to find.

If you see a tool or concept you are not familiar with, don't be scared.
Just follow along and you'll get started with ease, for sure.

If you have a question, the internet most probably already has the answer. Just search engine it!
If you can't find the answer, you can post your question to the [discussion forum](https://github.com/City-of-Helsinki/ml_project_template/discussions),
so the maintainers can help. [Stack Overflow](https://stackoverflow.com/) is the recommended forum for questions that are not specific to this template.

All you need to do for starting to work on your data project is to install the template following the instructions below.


## Contents

The repository contains the following files and folders:

    ## EDITABLES:
    data/               # Folder for storing data files. Contents are ignored by git.
    results/            # Runned notebooks, figures, trained models etc. Ignored by git.
    notebook_templates/ # Notebook templates with usage instructions, but without code examples
    .gitignore          # Filetypes, files and folders to be ignored by git.
    00_data.ipynb       # Data loading, cleaning and preprocessing with examples
    01_model.ipynb      # ML Model scripting, class creation and testing with examples
    02_loss.ipynb       # ML Evaluation with examples
    04_workflow.ipynb   # ML workflow definition with examples
    requirements.txt    # Required python packages and versions
    CONTRIBUTING.md     # Instructions for contributing
    settings.ini        # Project specific settings. Build instructions for lib and docs.

    ## AUTOMATICALLY GENERATED: (Do not edit unless otherwise specified!)
    docs/               # Project documentation (html)
    [your_module]/      # Python module built from the notebooks. The name of the module is the name of the repository (after installation).
    Makefile
    README.md 

    ## STATIC NON-EDITABLES: (Do not edit unless you really know what you're doing!)
    LISENCE
    MANIFEST.in
    docker-compose.yml   
    setup.py                                     

## Guiding Principles

The template followsiveur guiding principles.
Remember, none of these are strict and you are free to deviate for achieving the best results for you.
Also, it is better to get started with the work than to perfect it from the beginning.
You may return to these concepts when you want to improve your project or get stuck, iteratively!

For general coding best practices, refer to [dev.hel.fi](https://dev.hel.fi/) where applicable.

### 1. Prefer Standard Tools

It is recommended to use widely used standard tools with stable community support over niche optimal solutions, if possible.
This helps in ensuring that the methods used are reviewed, stable, accurate and maintained.
This is not a hard rule, but in general a viable stable solution is far better than one that is optimized but trivial.

This template installs the following Python standard tools for data science:

    NumPy          # matrix computation, linear algebra, vectorized operations
    Pandas         # complex data structures, useful functions for handling and plotting data, based on numpy
    Matplotlib     # create visualizations
    SciPy          # math and statistics
    Scikit-learn   # data mining, preprocessing, machine learning models
    ipython        # jupyter notebooks
    nbdev          # efficient software development with notebooks, code, docs and results as one 
    papermill      # parameterize notebooks
    Snakemake      # create reproducible workflows
    
These come with many dependencies, see `requirements.txt` for complete documentation.

Here are a few examples of preferred standard tools not included in this template that might help you get started when the template default tools lack features:

    statsmodels         # statistical models and tests (based on scipy)
    Keras / TensorFlow  # Neural networks (PyTorch is another alternative)
    Seaborn             # enhanced visualizations (based on matplotlib)
    Scrapy              # scrape data from online sources that do not have their own API
    BeautifulSoup       # parse data from web pages (simpler than Scrapy)
    NLTK                # natural language processing tools
    NetworkX            # network analysis algorithms
    DEAP                # genetic algorithms

The list is not, and never will be complete. You are free to use the tools that best suite you.
However, regardless of what your problem is, there are likely many different implementations for solving it.
Try to select a tool that is well documented and has steady userbase, frequent updates and many contributors.

### 2. Exploratory Programming - Use Jupyter Notebooks

We want to keep our code, documentation and results together, seamlessly.
We also want to see what's going on as we create the software, immediately.
Create code where we first need it, without the need of copy-pasting it around.
That's why we use jypyter notebooks as the core of our development.

Actually, even this page was generated from a notebook!

The notebooks are enhanced with `nbdev` tool to export code to modules, create doc pages, run tests, handle notebook version control etc.
Read more on nbdev on their [project pages](https://nbdev.fast.ai/).

---
**INFO BOX: How nbdev exports code from notebooks?**

![Exporting code from notebooks with nbdev](visuals/nbdev_build_lib.png)

---

Some reasoning for those who are not yet convinced:

- In data projects, the code efficiency is irrelevant. The thinking time is what matters.
- It is simply impractical to create poorly documented notebook. With notebook development, your code is always well documented.
- How many of you actually test your ML code? Clean, running notebooks are the tests, and with `nbdev` unit tests are easy to include. 
- Most data science projects involve multiple stakeholders with various backgrounds and skillsets.
Many of them do not understand code, and even those who do, can not if it is poorly documented, nor can they interpret the results alone.
Notebook development can be used to improve explainability.
- If you are building an armada of spaceships, tiny IoT devices or otherwise feel that this template does not fulfill your requirements for production pipeline,
you can still use this for planning and creating documentation. Clean code is easier to achieve following a well documented demo!

With notebook development you get the right results much faster, and everyone involved can actually understand what is happening.

Read more on exploratory programming with notebooks from [this blog post](https://www.fast.ai/2019/12/02/nbdev/).

### 3. Ease of Reproducibility

Poor reproducibility is a major issue in data science projects, both in the industry and academia, but is often overlooked at.
We at the city of Helsinki, as a public sector operators, value it highly, and believe that everyone will benefit from it.
Our goal is, that each state and decision of our ML models are reproducible.
A theoretical possibility of recreating a particular result is not enough, if it takes unreasonable efforts to do it.
Good reproducibility equals to ease of reproducibility.

For ease of reproducibility we 

1. Document
2. Seed
3. Orchestrate (pipeline)
4. Version control everything.

--- 
** INFO BOX: Documentation, Seeding, Orchestration and Version control **

*Documentation*

Documentation means, that everything in a ML project is explained in a text (up to a reasonable level).
This includes commenting code, but also adding relevant references, explaining the maths if needed, and introducing the logic and reasoning between every step.
To help you with documentation, you can ask yourself "what am I doing and why?" when coding,
and "what does this mean?" every time you get results, be it an intermediate step in data cleansing or the final results of your workflow.
Then, write the answers down, and ask your *non-tech-savvy* colleague to explain the process and results to you based on your documentation.
Iterate this, until you are happy with their answer, and you'll have great documentation!
With great documentation you can ensure that someone else could actually reproduce the same results you came up with.

*Seed*

Seeding means, that random processes are initialized with a *seed*, also known as *random state*.
Creating random numbers is a difficult task in computer science. Each random number you get from a random number generator,
such as the `np.random`, is actually a *pseudo random* number - number taken from a number sequence.
Bunch of numbers taken from this sequence have properties similar to some taking them from true random distribution.
The sequence is defined by the initial number, the seed, and so if you use the same seed for a random number generator,
you can reproduce the results.

*Orchestration*

Orchestration or *pipeline* means automated workflow control.
The goal is, that with a single command you can run all steps of your workflow,
instead of trying to rerun individual cells or notebooks.
It means, that with the same code and same data,
you can always reproduce the same results, even if your code isn't all in a single script.
It helps you to automate the training of ML models in production,
but also when testing your model in development.
An orchestrated workflow is excecuted on a trigger event.
They can either be static or dynamic. A static workflow executes all steps on the trigger event.
Most applications have static workflows.
This is ok, if you have a static data source (the data can change) and your processing steps are computationally light.
Dynamic workflows only execute the steps that are required, i.e. the steps,
that are affected by the changes that happened since the last trigger event.
This change can either be in the code or in the data.
For example if you may have a varying number of input sources to read data from at each training round of the algorithm.
Depending on your ML application, you should consider if you want to use static or dynamic orchestration.
We will add examples of both in the `workflow` notebook.

*Version control*

Version control means that you keep track of all changes in your system,
in a reversable way that allows you to step back to a previous version, or make branches to try out options.
Version control allows you to refer to a specific version of your system, making these snapshots reproducible.
We use Git for version control of code. Data version control is a topic we are still working on.

---

Read more on reproducible code and more tips for scientific computing on [Code Refinery](https://coderefinery.org/).
Code Refinery is an academic project focused on teaching and sharing knowledge on scientific computing practices.
They organize workshops and have many great free resources and self-learn [lessons](https://coderefinery.org/lessons/) that you can check out to improve your data science skills.

### 4. Tidy Data & Tools

Tidy principles are guidelines for clean and efficiend data utilization.
They can be appied to different programming languages.
Common packages, like `numpy`, `pandas` and `sklearn` have been developed so that these concepts are easy to apply.
Tidy data is easy to handle and understand. Tidy tools makes handling data, programming and creating explainable ML much easier.

**Data is tidy, when:**

1. **Every column is a variable** (either a feature or a label)
2. **Every row is an observation** (a data point).
3. **Every cell contains a single numerical value** (int, float, bool, str*)
> *strings should be converted to numerical format before applying ML

Read more on tidy data from [tidy data manifesto](https://vita.had.co.nz/papers/tidy-data.html).

**Tidy Tools:**

1. **Reuse existing data structures**

Favour the default data types of the tools used over custom data types.
Avoid unnecessary conversions: once you have converted your data to tidy format, keep it tidy.

2. **Compose simple functions with the pipe**

A pipe operator takes the expression before it and gives it as the first argument to the expression after it,
assuming the expression on the right is a function call. In addition, pipe functions should do one thing, and do it well.
They either perform a transformation or a side-effect, but never both.

In a transformation a modified copy of the input is returned.
In a side effect a reference to the directly modified input is returned.

This allows composition of simple functions. In addition, you can easily determine what a pipeable function does just from its name.
In pseudocode, it looks something like this

```
model() >> init(X_train, y_train) >> fit(hyperparam) >> predict(X_test) >> mean()
```

instead of multiple or nested lines of code

```
m = model()
m>init(X_train, y_train)
m>fit(hyperparam)
mean_values = mean(m>predict(X_test))
```
although you can use pipeable functions in either way, or as a composition.

Piped code is easy to read: you see that a model class is initialized, fitted with certain hyperparameters, a prediction is made and aggregated to a mean.
Python does not have a native pipe operator such as `%>%` in R tidyverse,
but Python class functions can be written in a pipe-like way.
More on this in the `model` notebook.

As an excercise, you can take a look at function definitions of your favourite `sklearn` model.
Which of the functions perform transformations and which side-effects?
Can you find a function that does both?

3. **Embrace functional programming**

Python is not a functional programming language, but it can be written in functional style.
Many of the concepts can be used to write cleaner data code with Python. 
The key concepts of functional programming include pure functions, immutability and higher order functions.

* Pure Functions do not have side effects, meaning that they do not change the state of the program. Output of a pure function depends only on it's input.
* Immutability means that data cannot be changed after creation. Changes can only be made to a copy of the data.
* Higher order functions are functions that may accept other functions as parameters or return new functions. This allows high level of abstractation.

Read more on functional programming with Python from [this Stack Abuse article](https://stackabuse.com/functional-programming-in-python).

4. **Are designed for humans**

Create your code in a way that it's easy to use, read and debug.
In addition to clean structure and documentation, consider the naming of your classes, functions and variables.
Clean code is easy to understand, and actually eases your work with the documentation. 
Function name should describe what it does.
A lengthy informative names is better than short, uninformative ones.
Having a common prefix with similar functions and autocomplete makes make even lengthy names descriptions easy to use. 

Read more on tidy tools from [tidy tools manifesto](https://cran.r-project.org/web/packages/tidyverse/vignettes/manifesto.html).


### 4. Data, Model & Loss: the core components of a ML workflow

The core of this template constitutes of three notebooks: data, model and loss.
The notebooks running number prefix (`00_data.ipynb` etc.) to emphasize the running order and to improve readability.
Any data project can be resolved by defining these three steps. Together they form a basis for a complete ML workflow from data to trained deployable model.

You might be used to doing all the steps in a single script or notebook,
but separating makes development, explaining the results and debugging much more efficient. 

Each notebook is also a basis for a python module, including tests and documentation.
The `nbdev` tool constructs a python module of each of these notebooks, under the folder `[your_repository]/[your_module]/`
(`ml_project_template/ml_project_template/` in this template).
The name of the module becomes the name of your repository after installing the template following the instructions below
This allows you to share code between your notebooks, or even publish a complete python module, while doing all your development in the notebooks.

**Data**

In the data notebook, the the data is loaded and cleaned, and a basic analysis may be carried out.
With nbdev you can also export data handling functions to be used in other notebooks.
You should also create a small toy dataset to develop and test your algorithm with - no,
trust me, your code won't work for the first n+1 times, and running it with the whole dataset will waste so much time!
This is also why we separate between the model and loss notebooks.

**Model**

In the model notebook, the machine learning model (or analytics or simulation) is explored, defined and tested.
You can begin with scripting, but based on the script you should develop real generalizable and tested code.
This part of the notebooks is the closest to traditional software development it gets: the output is a clean Python module. 


**Loss**

In the loss notebook, you will finally train your model with the whole dataset and evaluate it in action.
Some might call this step *inference*, others *evaluation*.
No matter the name, you evaluate the performance of your model to see if it is ready for production.
If the results are satisfactory, you can ship your code to it's destination.
For example Azure SDK allows you to define your code in Python and then excecute it in the cloud, seamlessly.
However, this part depends a lot on the project, so we'll leave it to you to figure it out.
If your are doing research, having the results in the notebooks might be enough for you.

**Workflow**

We added the fourth notebook to enable automatic workflow control of the notebooks.
With the help of this notebook, you can create workflows to automatically update, train and deploy your ML model.
The workflows can either be static or dynamic, and you can even parameterize them to automatically update your workflow definition.
Usually this notebook is the last one you need to touch, so you don't need to mind it in the beginning.
However, if you have multiple or varying number of data sources, or a very complex workflow,
you might want to define a dynamic workflow already in the early development phase.

You can also create new notebooks to your liking. For example, if you want to create and create many algorithms that are inheritantly different
it might be better to separate them in their own notebooks `02a_model1.ipynb` and `02b_model2.ipynb`.

## Example Project

We wanted to make this template easy to approach.
That's why we included a demo, that it is built around.

The demo is an example ML project on automating heart disease diagnosis with logistic regression on [UCI heart disease open dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).
The dataset contains missing values, and is thus great for demonstrating some light data wrangling.
The demo is only meant for showcasing how the template joins together different tools and steps.

**If you'd like to skip the demo**, and get right into action, you can replace the notebooks `index`, `data`, `model`, `loss` and `workflow` with clean copies under the folder `notebook_templates/`.

The `index` notebook (this notebook or the empty copy) will become the `README` of your project and frontpage of your documentation, so edit it accordingly.
You should at least have a general description of the project,
instructions on how to install and use it,
and instructions for contributing.

## Installing the Template
{% include note.html content='if you are doing a project on personal or sensitive data for the City of Helsinki, contact the data and analytics team of the city before proceeding!' %}
### On your GitHub homepage:

0. (Create [GitHub account](https://github.com/) if you do not have one already. 
1. Sign into your GitHub homepage
2. In the top right corner of the homepage, click the '+'-button
3. Select 'Import repository'
4. Under 'Your old repository's clone URL' copy the clone url of this repository: `https://github.com/City-of-Helsinki/ml_project_template`
5. Give your project a name. Do not use the dash symbol '-', but rather the underscore '_', because the name of the repo will become the name of your Python module.
6. If you are creating a project for your organization, change owner of the repo from the drop down bar (it's you by default).
You need to be included as a team member to the GitHub of the organization.
7. Define your project publicity (you can change this later, in most cases you'll want to begin with a private repo).
8. Click 'Begin import'

This will create a new repository for you copying everything from this template, including the commit history.

### On your computing environment:

**Put all the highlited ** `commands` ** to shell one ate a time and press enter**
**(replace the parts with square brackets with your own information '[replace this with your info]')**
(remove the brackets)

0. Create an SSH key and add it to your github profile. SSH is a protocol for secure communication over the internet. 
    A ssh key is unique to a computing unit, and you must recreate this step every time you are using a new unit,
    be it a personal computer, server or a cloud computing instance. You can read more on SSH from [Wikipedia](https://fi.wikipedia.org/wiki/SSH) or 
    from [GitHub docs](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
    * Create SSH key with `ssh-keygen -t ed25519 -C "[your_email]"`
    * You can leave the name empty (just press enter), but **always create keys with a secure password that you remember**.
    This password can not be reset. You have to create new key if you forget it.
    * Now among other lines, there should be a text displayed saying `Your public key has been saved in [public_key_path]/id_ed25519.pub`.
    * Copy the public key adress and call `cat [public_key_path]/id_ed25519.pub]`
    * Now the key has been copied to clipboard and displayed on your shell. It begins with 'ssh' and ends with your email.
    Depending on your system, you may also have to manually copy it from the shell output.
    * Go to your GitHub homepage > progile picture in top right corner > settings > SSH and GPG keys > new ssh key
    * Paste the public key to the key part, and give the key a name that describes the computing environment it belongs to.
    * If you permanently stop using this computing environment, remove the public key from your github profile.
    
1. In your shell, move to the folder you want to work in: `cd [path to your programming projects folder]`.
(If you get lost, `cd ..` moves you one folder towards root, and `cd` gets you to root.)
2. Clone the repository you just imported: `git clone git@github.com:[repository_owner]/[your_repository]`.
If the repository is private, you'll be asked the password of the ssh key you just generated. 
This will copy all the files and folders that you imported to your new repository in the github website, to your computing environment.
3. Go inside the repository folder: `cd [your_repository]`
4. Create virtual environment.  Virtual environments allow you to install project specific python versions and track dependencies.
Read more on virtual environments from [this blog post](https://realpython.com/python-virtual-environments-a-primer/).


    # Using conda (Azure ML only supports conda virtual environments):

    conda create --name [environment_name] python=3.8
    conda activate [environment_name] # every time you start working
    conda install pip

    conda deactivate # when you stop working

    # Using virtualenv (preferred way if not working in Azure):

    pip install virtualenv
    python3.8 -m virtualenv [environment_name]
    source [environment_name]/bin/activate # every time you start working
    
    deactivate # when you stop working

5. Install dependencies (versions of python packages that work well together):


    pip install -r requirements.txt # install required versions of python packages with pip
    nbdev_install_git_hooks # install nbdev git additions

6. Create an ipython kernel for running the notebooks. Good practice is to name the kernel with your virtual environment name.


    python -m ipykernel install --user --name [ipython_kernel_name] --display-name "Python 3.8 ([ipython_kernel_name])"

7. With your team, decide which notebook editor are you using. There are two common editors: Jupyter and JupyterLab, but both run the same notebooks.
Depending on the selection, you'll have to edit the top cell of each notebook where black formatter extension is activated for the notebook cells.
You can change this later, but it is convenient to only develop with one type of an editor.
Black is a code formatting tool used to unify code style regardless of who is writing it.
You may notice, that the structure of your code changes a bit from what you have written after you run the cells of a notebook.
This is the formatter restructuring your code.
There are other formats and tools, and even more opinions on them, but black is used in the city of Helsinki projects.
So, after deciding which editor you are working with (Azure ML default notebook view is based on JupyterLab), edit the top cell of all notebooks:
    
    
    # if using Jupyter:
    %load_ext nb_black
    
    # if using JupyterLab:
    %load_ext lab_black

    # do not add comment to same line with a magic command:
    %load_ext nb_black #this comment breaks the magic command

8. Check that you can run the notebooks `00_data.ipynb`, `01_model.ipynb` and `02_loss.ipynb`.
You may have to change the kernel your notebook interpreter is using to the one you just created.
This can be done drop down bar in top of the notebook editor.

9. Edit `settings.ini`, `docs/_config.yml` and `docs/_data/topnav.yml` according to your project details.
The files contain instructions for minimum required edits.
You can continue editing them in the future, so no need to worry about getting it right the first time.
These are used for building the python modules and docs based on your notebooks.
If you get errors when building a module or docs, take a look again at these files.
10. Configure your git user name and email adress (one of those added to your git account) if you haven't done it already:


    git config --global user.name "FIRST_NAME LAST_NAME"
    git config --global user.email "MY_NAME@example.com"

11. Make initial commit (snapshot of the code as it is when you begin the work):


    git add .
    git commit -m "Initial commit"

12. Push (save changes to remote repository): `git push -u origin master`. You will be asked to log in with your SSH key and password, again.


## How to use

1. Install this template as basis of your new project (see above).

2. Remember to always activate your virtual environment before you start working: `conda activate [environment name]` with anaconda or `source [environment name]/bin/activate` with virtualenv

2. Check out the notebooks, and play around a bit to see that your installation works (notebooks run smoothly) and you understand the template structure

3. Replace the notebooks `index`, `data`, `model`, `loss` and `workflow` with copies without the code examples (there is also additional empty notebook template `_XX_empty_notebook_template.ipynb` if you want to deviate from basic template structure):


    git rm index.ipynb 00_data.ipynb 01_model.ipynb 02_loss.ipynb 03_workflow.ipynb
    git mv notebook_templates/_index.ipynb ./index.ipynb
    git mv notebook_templates/_00_data.ipynb ./00_data.ipynb
    git mv notebook_templates/_01_model.ipynb ./01_model.ipynb
    git mv notebook_templates/_02_loss.ipynb ./02_loss.ipynb
    git mv notebook_templates/_03_workflow.ipynb ./03_workflow.ipynb

4. You may delete the folders `ml_project_template` and `notebook_templates`.


    git rm -r ml_project_template notebook_templates

5. Save your notebooks and call `nbdev_build_lib` to build python modules of your notebooks - needed if you want to share code between notebooks or create a modules.
Remember to do this if you want to rerun your workflow after making changes to exportables. 

6. Save your notebooks and call `nbdev_build_docs` to create doc pages based on your notebooks (see below).
This will also create README.md file based on this notebook.
If you want to host your project pages on GitHub, you will have to make your project public.
You can also build the pages locally with jekyll.

7. You can install new packages with `pip install [package name]`.
Check out what packages are installed with the template from `requirements.txt`, or check if a specific package is installed with `pip show [package_name]`.
If you install new packages, remember to update the requirements for dependency management: `pip freeze > requirements.txt`.

8. Before you publish your project, edit LISENCE and the copywright information in the `index.ipynb` according to your project details.
Please mind that some dependencies of your project might have more restrictive licenses than the Apace-2.0 this template is distributed under. 

9. Remember to track your changes with git! 

---

**INFO BOX: Some useful commands with git**


    git status #See which files have changes since the last commit: `git status` 

    git add [filenames separated by whitespace ' '] # Add files to a commit. * will add all files, and . will add everythin in a directory 

    git commit -m "[short description of the changes you made]" # Create commit and explain what you changed

    git push origin -u # Push commit to remote repository (GitHub server) 

*A good rule of thumb is to commit every change you make, and push at the end of the day when you stop working!*


    git pull # Load changes that someone else has made

If you are working with a team of people, you will most likely run into conflicts when pushing or pulling code.
This means, that there are overlapping changes in the code. Read more from [Stack Overflow](https://stackoverflow.com/questions/161813/how-to-resolve-merge-conflicts-in-a-git-repository)
or [GitHub docs](https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github)
on how to resolve conflicts.


    git rm [file name] # Remove files so that git will also stop tracking them  (`git rm -r [folder]` for folders)

To ignore files or folders from being tracked by git, add them to `.gitignore` file.
In this template the `data` and `results` folders have been added to the `.gitignore`.
We do not want to version them with git, as it will explode the size of the repository. 

Branches are like alternative timelines in your commit history.
They allow you to test out things that radically change the code.


    git branch # Check out current branch

    git branch [branch_name] # Make a new branch

    git checkout [branch name] # Change to another branch


In addition, there are many fancy features for git that enable comparing differences, collaborative work, debugging, automated testing and other crazy things.
However, there are better sources for learning all that stuff, like this [free ebook](https://git-scm.com/book/en/v2).

---



## Contributing

See `CONTRIBUTING.md` on how to contribute.


## Copyright

Copyright 2021 City-of-Helsinki. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.

This template was built using [nbdev](https://nbdev.fast.ai/) on top of the fast.ai [nbdev template](https://github.com/fastai/nbdev_template).

## Now you are all set up and ready to begin you ML project!
