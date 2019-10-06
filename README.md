# Disaster Response Pipeline Project

During a natural disaster, responsiveness to requests, alerts, and other messages is paramount to ensure the success of recovery mission, and in many cases can mean life-or-death for victims.

This project builds a Machine Learning pipeline that pre-processes & parses _Figure Eight_'s disaster message dataset to train a supervised learning model that can accurately categorize incoming messages to their respective categories (and dispatch them to the proper disaster response organization).

Ultimately, the model feeds into a simple, easy-to-use web app that will provide response leads with overviews to messages and instant classifications for incoming messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
