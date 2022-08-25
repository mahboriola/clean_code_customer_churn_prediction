# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Your project description here.
This project is part of the Udacity nanodegree [Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). In this project, the goal is to reorganize a Jupyter Notebook (`churn_notebook.ipynb`) into Python Script (`churn_library.py`) applying the clean code concepts and following the PEP8 coding convetions.

Furthermore, in this project is required testing and logging.

## Files and data description


```
.
├── data
├── images
│   ├── eda
│   └── results
├── logs
├── models
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── README.md
└── requirements.txt
```

Above there are the organization of this project.

The folder `data` contains the data files used in data visualization and model training.
The folder `images` has two more folders where the plots from EDA and model results are stored.
The logs generated during the tests are located at the folder `logs`.
In the folder `models` are stored the models trained.

The files [churn_library.py](churn_library.py) and [churn_script_logging_and_tests.py](churn_script_logging_and_tests.py) are responsible to run the main code and test the main code respectively.

## Running Files
How do you run your files? What should happen when you run your files?

Before running this project, first you need to install the required libs, so just run:
```
pip install -r requirements.txt
```


To run this project you should execute the following command in your terminal:
```
$ ipython churn_library.py
```

By the way, to run tests on the `churn_library.py`, just run:
```
ipython churn_script_logging_and_tests.py
```

After running the above commands, you can check for their outputs in the folders mentioneds in the previous section.

In the `master` branch you can find the output files in their respective folders and some others files available for this project, but if you want to run in a clean setup just checkout to the `clean` branch.
