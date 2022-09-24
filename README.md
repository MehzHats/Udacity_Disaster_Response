# Udacity's Disaster Response Pipeline Project

## Project Description
The Disaster Response Pipeline is a part of the Nanodegree Program by Udacity, in collaboration with Appen (formerly Figure Eight). For this project we have built a Machine Learning model that classifies the emergency of the message from real-life disaster events, based on the information and needs communicated by the sender. This also includes a web app where a new message is classified in to 36 pre-defined categories.

## Project Components
The project is divided in three major parts

### 1. ETL Pipeline
Data Processing Pipeline, to load data from source, merge and clean the data and save it in a SQLite database.

### 2. ML Pipeline
Machine Learning Pipeline, to load preprocessed data from the SQLite database, split the dataset into training and test sets, build and train the model using GridSearchCV and export the final model as pkl file.

### 3. Flask Web App
Web App to show data visualisation of the trained data using Plotly. An interactive portal is available to use model in real time.

## Getting Started

### Dependencies

1. Python 3.10.4
2. Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
3. Natural Language Process Libraries: NLTK
4. SQLlite Database Libraqries: SQLalchemy
5. Web App and Data Visualization: Flask, Plotly

A requirements.txt file is created to install all the dependencies needed for this project. To install them run following command.

```pip install -r requirements.txt```

## Authors
* [MehzHats](https://github.com/MehzHats)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance.

![message_result](images/message_result.png)

2. Data visualisation for the least 10 categories.

![least_categories](images/least_categories.png)

3. Data visualisation for the message category distribution.

![least_categories](images/message_cat_dist.png)

4. Data visualisation for the words count distribution.

![least_categories](images/word_count_dist.png)

5. The f1 score, precision and recall for the test set output is shown for each category.

model_metrics_1             |  model_metrics_2
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_1.png )  |  ![](images/model_metrics/model_metrics_2.png)


model_metrics_3             |  model_metrics_4
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_3.png )  |  ![](images/model_metrics/model_metrics_4.png)


model_metrics_4             |  model_metrics_6
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_5.png )  |  ![](images/model_metrics/model_metrics_6.png)


model_metrics_7             |  model_metrics_8
:-------------------------:|:-------------------------:
![](images/model_metrics/model_metrics_3.png )  |  ![](images/model_metrics/model_metrics_4.png)

## Acknowledgements

* [Udacity](https://www.udacity.com/) Data Science Nanodegree Program
* [Appen](https://appen.com/) Data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.