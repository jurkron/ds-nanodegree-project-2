# Disaster Response Pipeline Project

### Project Motivation

Figure Eight is a company that receives a lot of disaster messages. These messages have to be classified in order to send the messages to an appropriate disaster relief agency.
This project creates a machine learning pipeline to categorize these events in order to reduce manual work and fasten the process.

### Requirements

This project needs the following requirement to run:

* [Python3](https://www.python.org)

further libraries are used and should be preinstalled 

* [pandas](https://pandas.pydata.org/) 
* [sqlalchemy](https://www.sqlalchemy.org/)
* [sklearn](https://scikit-learn.org/)

### Files

#### app

    run.py - script to run the app
    templates/.html - html templates

#### data

    disaster_messages.csv - contains messages 
    disaster_categories.csv - contains categories for the messages
    process_data.py - python script with the ETL pipeline 

#### models

    train_classifier.py - script with the nlp pipeline to build the machine learning model
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgements

The dataset used in this project was provided by [Figure Eight](https://www.figure-eight.com/) through [Udacity](https.//www.udacity.com). Thanks to both for providing this data.