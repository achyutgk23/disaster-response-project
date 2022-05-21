import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

# Function for text pre-processing
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # add some visualizations
    # messages classified according to genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data to plot top 10 and bottom 10 disaster response categories
    categories_counts = dict()
    for i in list(df.columns[4:]):
        categories_counts[i] = df[i].sum()

    categories_counts = dict(sorted(categories_counts.items(), key=lambda item: item[1]))
    top_10_category_names = list(dict(list(categories_counts.items())[-10:]).keys())
    top_10_category_values = list(dict(list(categories_counts.items())[-10:]).values())

    bottom_10_category_names = list(dict(list(categories_counts.items())[:10]).keys())
    bottom_10_category_values = list(dict(list(categories_counts.items())[:10]).values())

 # create visuals
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
        {
            'data':[
                Scatter(
                    x=top_10_category_names,
                    y=top_10_category_values
                )
            ],

            'layout': {
                'title': 'Top 10 Disaster Response Cateories',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title':'Category Names'
                }
            }

        },
        {
            'data':[
                Scatter(
                    x=bottom_10_category_names,
                    y=bottom_10_category_values
                )
            ],

            'layout': {
                'title': 'Bottom 10 Disaster Response Cateories',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title':'Category Names'
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
