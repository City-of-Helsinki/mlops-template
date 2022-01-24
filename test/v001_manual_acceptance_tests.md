# Helsinki ML Project Template, Release v0.0.1, Acceptance tests

The purpose of these tests is to validate the usability of the template to the end users.
The tests focus on the template instructions - can the user with no prior experience with the template understand it, install it and begin work based on the instructions?

After you have carried out a test, add PASS / FAIL + YOUR NAME + DATE after the test. If the test FAILS, make an issue where you describe the problem in detail. Add the issue number to the list after the test. Unclear instructions are a valid reason for an issue.

## 1. Review documentation on GitHub: https://github.com/Datahel/ml_project_template/tree/master
1.1 Read through README.md. Check for broken formatting, typos and mark down if there are any parts you don't understand. Check that all links in README work, i.e. lead to a sensible page or hyperlink to some part of the template. Acceptance criteria: nothing to fix (format breakdowns may have to be accepted because README / html conversions).
1.2 Check out the notebooks (in GitHub) (index, 00_data, 01_model, 02_loss, 03_workflow, 04_api). Read through the notebooks and again look out for broken formatting, typos, and mark down parts that you don't understand. Check that all links work, i.e. lead to sensible page or hyperref to some part of the template. Acceptance criteria: nothing to fix (format breakdowns may have to be accepted because README / html conversions).

## 2. Revew documentation on GitHub pages: https://city-of-helsinki.github.io/ml_project_template/
2.1 Check out all the tabs (index, 00_data, 01_model, 02_loss, 03_workflow, 04_api). Read through the notebooks and again look out for broken formatting, typos, and mark down parts that you don't understand. Check that all links work, i.e. lead to sensible page or hyperref to some part of the template.

## 3. Review the install instructions: https://github.com/Datahel/ml_project_template/tree/master/README.md
3.1 Install with Codespaces: follow the How to install: 1, 2a, 3. Create the repo as inside repo in Datahel and name your repo ml_template_test_codespaces. Acceptance criteria: you can run 03_workflow notebook and data, model and loss notebooks are created by papermill under results/notebooks.
3.2 Local install with Docker: follow the How to install: 1, 2b, 3. Create the repo as inside repo in Datahel and name your repo ml_template_test_docker. Acceptance criteria: you can run 03_workflow notebook and data, model and loss notebooks are created by papermill under results/notebooks.
3.3. Local manual install: follow the How to install: 1, 2c, 3. Create the repo as inside repo in Datahel and name your repo ml_template_test_local. Acceptance criteria: you can run 03_workflow notebook and data, model and loss notebooks are created by papermill under results/notebooks.
3.4. Local manual install: follow the How to install: 1, 2c, 3. Create the repo as inside repo in Datahel and name your repo ml_template_test_local. Acceptance criteria: you can run 03_workflow notebook and data, model and loss notebooks are created by papermill under results/notebooks.

## 4. Review the How to use instructions: https://github.com/Datahel/ml_project_template/tree/master/README.md
4.1 Take the repo created ad step 3.1. Follow the How to use & Installing and updating project requirements, add and install `seaborn` to the project requirements. Acceptance criteria: seaborn is automatically installed when a new codespace is created from the project.
4.2 Take the repo created ad step 3.1. Follow the How to use instructions and publish github hosted doc pages for your test repo. Acceptance criteria: https://Datahel.github.io/ml_project_template_test_codespaces/ can be accessed.
