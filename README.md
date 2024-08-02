# datavizRun
A small dataviz project about gps traces.

# Setup working environment

Create a Python virtual environment.
```
virtualven venv
```

Then activate the virtual python environment:
```
python venv/bin/activate
```

And finally add the required modules for this project:
```
pip install -r requirements.txt
```

Start the notebook server
```
(venv) ... jupyter notebook
```

# How to start?

You can look at the notebook in the folder *notebook*:
+ **./notebook/preparingDatGPS.ipynb** is about pre-processing the data collected
+ **./notebook/datavizdatGPS.ipynb** is about displaying data, answering questions you may have about the data

# What will you find in that project?

You will find:
+ some operations about data cleaning, from .gpx files to .csv files.
+ exemples of **Pandas** library usage
+ exemples of GPS data visualisation
+ exemples of attempt to answer questions about those data
+ data visualisuation of gps traces

## About Panda

The usual, write and read csv files from Pandas.

Concatenation of Pandas DataFrame, change of index in order to use time reference for each data sample.

### About my data structure

I'm using a list of file names, a list of DataFrame and a DataFrame gathering information about data sample.

And a data sample correspond to a GPS trace as a .gpx file (more info in the notebooks).

# What will you not find in that project?

I'm working here with static data, even so I can add new data each week, it is not real time processing, even so the data cover several years.

# To do Next

I need to improve my dataviz, to experiment more with gps Python mddules, no need to re-invent the wheel.


