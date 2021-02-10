# Disaster Response Pipeline

Whenever a disaster occurs, millions of people will send messages asking for help. Different organizations will typically respond to different problems; however, it is also during these times that organizations have the least capacity to filter through messages. Using data provided by Figure Eight, this project first uses an ETL pipeline to save a cleaned dataset containing messages with disaster category classifications in a SQLite database. Then, it uses a ML pipeline to create a supervised ML model that classifies disaster messages. Finally, the model is used to create a web app that can classify new messages into the different categories.

# Requirements

- pandas
- SQLAlchemy
- re
- nltk
- scikit-learn
- pickle
- sys
- json
- Plotly
- Flask
- Joblib

# Files

- app
  - Wordcount.py - contains custom transformer WordCounter
  - templates
    - master.html - main page of web app
    - go.html - classification result page of web app
  - run.py - Flask file that runs app
- data
  - disaster_categories.csv - categories data to process
  - disaster_messages.csv - messages data to process
  - process_data.py - ETL pipeline
- models
  - Wordcount.py - contains custom transformer WordCounter
  - train_classifier.py - ML pipeline
  
To view the web app, execute the following commands in the project's root directory. First, run the ETL pipeline to save a SQLite database with the cleaned dataset.

```
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Then, run the ML pipeline to save a pickle file with the supervised ML model.

```
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Finally, run the Flask file to view the web app at http://0.0.0.0:3001/

```
$ python run.py
```

# Acknowledgements

Special thanks to Figure Eight for providing the data and the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) for providing ETL pipeline, ML pipeline, and web app templates.

# License

The contents of this repository are covered under the [GNU General Public License v3.0](https://github.com/evanchen13/disaster-response/blob/main/LICENSE).
