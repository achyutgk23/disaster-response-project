import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt'])
nltk.download(['wordnet'])
nltk.download(['stopwords'])
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle



def load_data(database_filepath):
    """Load the data into df variable from the database file.

    Args:
    database_filename: str. Name of the database file in which
                       the data is stored

    Returns:
    X: ndarray. An ndarray and an input variable
       containing message column of the df datasets
    Y: DataFrame. A dataframe and a target variable
       containing all 36 categories columns
    category_names: list. A python list containing target variable category_names
    """

    # Load the data from the database into 'df' dataframe
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("DisasterResponse.db",con=engine)

    # Assign the message column from 'df' dataframe to 'X' variable.
    # X is an input variable.
    X = df.message.values

    # Assign all the 36 category columns from 'df' dataframe to 'Y' variables.
    # Y is the target variable
    Y = df[df.columns[4:]]

    # Extract all 36 category names
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """Apply some text preprocessing techniques like normalization,
    tokenization, stop word removal and lemmatization.

    Args:
    text: str. messages from people during a DisasterResponse

    Returns:
    clean_words: list. A python list of words which are left after preprocessing
    """

    # Normalize and convert the text into lower case
    text = re.sub(r"[^a-z A-Z 0-9]", " ", text).lower()

    # Convert each word of the text into a token
    tokens = word_tokenize(text)

    # Remove stop words from the words in the 'tokens' list
    words = [x for x in tokens if x not in stopwords.words('english')]

    # Instantiate the 'lemmatizer' function
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word from 'words' list, add those words into
    # 'clean_words' python list and return the 'clean_words' list.
    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word, pos='v')
        clean_words.append(clean_word)

    return clean_words


def build_model():
    """Construct a pipeline consisting of 'CountVectorizer' function,
    'TfidfTransformer' and 'Random RandomForestClassifier'

    Args:
    None

    Returns:
    cv: GridSearchCV object. A GridSearch object which includes the pipeline and
        a parameter grid for hyper-parameter tuning
    """

    # Construct a 'pipeline' conatining 'CountVectorizer' function,
    # 'TfidfTransformer' and 'RandomForestClassifier'.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_jobs=-1))
        ])

    # # Create a parameter grid for hyper-parameter tuning of the classifier.
    # param = [
    #     {'vect__ngram_range':[(1, 1), (1, 2)],
    #     'clf__n_estimators': [200, 230, 260],
    #     'clf__bootstrap': [True, False]}
    #     ]
    #
    # # Create a 'GridSearchCV' function, pass 'pipeline' and
    # # 'paramer grid' as arguments and return the function with the name 'cv'.
    # cv = GridSearchCV(pipeline, param_grid=param)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model performance by using the evaluation metric
    'classification_report'.

    Args:
    model: GridSearchCV object. Model obtained from the 'build_model' function
    X_test: numpy.ndarry. Messages dataset for model testing
    Y_test: numpy.ndarry. The dataset of target variables for model testing
    category_names: list. A python list containing target variable category_names

    Returns:
    prints the classification report of the model's performance on test set
    """

    # Make predictions on test set using trained model
    y_pred = model.predict(X_test)

    # Print the classification report of the model's performance
    print(classification_report(Y_test, y_pred, target_names=category_names))




def save_model(model, model_filepath):
    """Save the model with best parameters in 'pickle' format.

    Args:
    model: GridSearchCV object. Model obtained from the 'build_model' function
    model_filepath: str. The pathway in the directory for the model to be saved

    Returns:
    None
    """

    # Save the trained and hyper-parameter tuned model into the directory
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
