import sys
import re
import pickle
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger', 'omw-1.4'])

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load Data from Db

    Inputs:
        database_filepath -> To SQLite database destination path (i.e DisasterResponse.db).
        table_name -> table name in the database.

    Outputs:
        X -> Features of a df
        Y -> Labels of a df
        category_names -> List of categories(not sure?)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Messages', engine)

    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns[4:])

    Y = Y.astype(bool)

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text

    Input:
        text -> message to tokenise

    Outputs:
        tokens_cleaned -> tokens cleaned from text messages

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_detected = re.findall(url_regex, text)
    for url in urls_detected:
        text = text.replace(url, "urlplaceholder")


    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens_cleaned = []
    for token in tokens:
        cleaned_token = lemmatizer.lemmatize(token).lower().strip()
        tokens_cleaned.append(cleaned_token)

    return tokens_cleaned

def build_model():
    """
    Build a pipeline.

    Output: processing text messages and adding a classifier.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters = {
    #             # 'vect__min_df': [],
    # #             'tfidf__use_idf': [],
    #              'clf__estimator__n_estimators': [25,50],
    #              'clf__estimator__min_samples_split': [2,4]
    # }
    # model = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model
    Input:
        model -> model to be evaluated.
        X_test -> Input test data
        Y_test -> label test data
    This tests the model and shows the accuracy of the prediction.

    """
    Y_test_predict = model.predict(X_test)

    Y_test_predict = pd.DataFrame(Y_test_predict, columns=Y_test.columns)
    for category in Y_test.columns:
        print('-----{}------'.format(category.upper()))
        print(classification_report(Y_test[category], Y_test_predict[category]))


def save_model(model, model_filepath):
    """
    Save the model

    Save trained model as pickle file.

    Arguments:
        model -> Scikit ML Pipeline and GridSearchCV
        model_filepath -> destination path to save .pkl file

    """
    with open(model_filepath, 'wb') as mod_pkl_file:
        pickle.dump(model, mod_pkl_file)


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
        evaluate_model(model, X_test, Y_test)

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