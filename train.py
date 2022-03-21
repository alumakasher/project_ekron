from pprint import pprint
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.pipeline import Pipeline

from spacy.lang.en import English
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time
import joblib
from utils import predictors, get_tfidf_vector

# parameters for SVM:
parameters = {
    'classifier__C': [1, 10, 100, 1000],
    'classifier__gamma': [0.001, 0.0001]
}

'''
#parameters for logistic regression:

parameters = {
    "vectorizer__max_df": (0.5, 0.75, 1.0),
    # 'vectorizer__max_features': (None, 5000, 10000, 50000),
    "vectorizer__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    "classifier__max_iter": (20,),
    #"classifier__alpha": (0.00001, 0.000001),
    "classifier__penalty": ("l2", "elasticnet"),
    # 'classifier__max_iter': (10, 50, 80),
}

#parameters for random forest:
parameters = {
    "classifier__max_depth": [3, 5,10], #[3, None]
    "classifier__max_features": [1, 3, 10],
    "classifier__min_samples_split": [2, 5, 10], #[1, 3, 10]
    "classifier__min_samples_leaf": [2, 5, 10], #[1, 3, 10]
    # "bootstrap": [True, False],
    "classifier__criterion": ["gini", "entropy"]
}
'''

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)  # To see all rows
    pd.set_option('display.max_columns', 500)  # To see all columns
    pd.set_option('display.width', 1000)

    df_facebook = pd.read_csv("faceebook_data_recent.csv", sep=",")
    print(df_facebook.head())
    print(df_facebook.shape)

    # View data information
    # print(df_facebook.info())

    print(df_facebook.label.value_counts())

    # Load English tokenizer, tagger, parser, Named entity recognition (NER) and word vectors
    parser = English()

    # removing emojies:
    df_facebook['text'] = df_facebook['text'].apply(lambda text: re.sub('\\<.*?\>', ' ', text))
    df_facebook['text'] = df_facebook['text'].apply(lambda x: x.strip())
    df_facebook = df_facebook.loc[df_facebook['text'] != '', :]
    X = df_facebook['text']  # the features we want to analyze
    ylabels = df_facebook['label']  # the labels, or answers, we want to test against

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2)

    rows_test = list(set(X.index) - set(X_train.index))
    df_test_indices = pd.DataFrame(data=rows_test, columns=["index_test"])
    df_test_indices.to_csv('test_indices.csv', index = False)

    print("y_train: ", y_train.value_counts())
    print("y_test: ", y_test.value_counts())

    list_of_model_types = ['svm', 'logistic_regression', 'random_forest']

    # for model_name in list_of_model_types:
    # classifier = LogisticRegression()
    classifier = svm.SVC(kernel='linear')  # Linear Kernel
    # classifier = RandomForestClassifier(n_estimators=100)

    # Create pipeline using TFIDF
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', get_tfidf_vector()),
                     ('classifier', classifier)])

    grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("Done in %0.3f Seconds" % (time() - t0))

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # model generation
    print("Generating model")
    X_train = list(X_train)
    y_train = np.array(y_train)

    print("Model prediction")
    predicted = grid_search.predict(X_test)

    elements_count = np.bincount(predicted)
    elements = np.nonzero(elements_count)[0]
    print(dict(zip(elements, elements_count[elements])))

    print("Confusion matrix: \n", metrics.confusion_matrix(y_test, predicted))

    joblib.dump(grid_search, 'svm.pkl')
    #joblib.dump(grid_search, 'logistic_regression.pkl')
    #joblib.dump(grid_search, 'random_forest.pkl')

    # Model Accuracy
    print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted))
    print("SVM Precision:", metrics.precision_score(y_test, predicted))
    print("SVM Recall:", metrics.recall_score(y_test, predicted))


    # Score board:
    # 1) Logistic Regression:
    #    Logistic Regression Accuracy:0.9475524475524476
    #    Logistic Regression Precision: 0.75
    #    Logistic Regression Recall: 0.09375
    # 2) SVM :
    # SVM accuracy: 0.972027972027972
    # SVM Precision: 0.9230769230769231
    # SVM Recall: 0.4444444444444444
    # 3) Random Forest :
    # Random Forest accuracy: 0.9702797202797203
    # Random Forest Precision: 1.0
    # Random Forest Recall: 0.2608695652173913
    
    # Using gridSearch:
    # SVM accuracy: 0.9825174825174825
    # SVM Precision: 0.8888888888888888
    # SVM Recall: 0.6666666666666666
    # Logistic Regression  accuracy: 0.9615384615384616
    # Logistic Regression  Precision: 1.0
    # Logistic Regression  Recall: 0.12
    # Random Forest accuracy: 0.9458041958041958
    # Random Forest Precision:  0.0
    # Random Forest Recall: 0.06666666666666667
