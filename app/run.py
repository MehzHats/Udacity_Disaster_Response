import json
import plotly
import joblib
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Line, Waterfall
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens_cleaned = []
    for tok in tokens:
        cleaned_token = lemmatizer.lemmatize(tok).lower().strip()
        tokens_cleaned.append(cleaned_token)

    return tokens_cleaned

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    least_category_count = df.iloc[:,4:].sum().sort_values(ascending=True)[0:10]
    least_category_names = list(least_category_count.index)

    Message_counts = df.drop(['id','message','original','genre', 'related'], axis=1).sum().sort_values(ascending=False)
    Message_names = list(Message_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # GRAPH 2 - distribution of categories
        {
            'data': [
                Bar(
                    x=least_category_names,
                    y=least_category_count
                )
            ],
            'layout': {
                'title': 'Least 10 Categories',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 35
                }
            }
        },
        # GRAPH 3 - category graph
        {
            'data': [
                Bar(
                    x=Message_names,
                    y=Message_counts
                )
            ],

            'layout': {
                'title': 'Message Category Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },
        # GRAPH 4 - distribution of word lengths
        {
            'data': [
                Line(
                    y=df['message'].apply(lambda s: len(s.split(' ')))
                )
            ],
            'layout': {
                'title': "Words Count Distribution",
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Number of words'
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()