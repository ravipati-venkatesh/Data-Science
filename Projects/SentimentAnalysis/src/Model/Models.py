#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_test_data_split(df):
    # Split data into training and testing sets
    X = df['vector'].to_list()
    y = df['sentiment_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)


def fit_models(X_train, y_train, model_name, vect_name):
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier()
    }

    # Dictionary to hold best models

    # Train models and find best models using GridSearchCV
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [100, 200, 300]
        }
    elif model_name == 'Multinomial Naive Bayes':
        param_grid = {
            'alpha': [0.1, 0.5, 1.0]
        }
    elif model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'SVC':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    elif model_name == 'KNN':
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    else:
        # Gaussian Naive Bayes does not have parameters to tune
        pass

#     # Perform GridSearchCV
#     grid_search = GridSearchCV(models[model_name], param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)

#     # Get best model
#     best_model = grid_search.best_estimator_
    best_model = models[model_name]
    best_model.fit(X_train, y_train)
    
    # Save best model to file
    model_filename = f'./Model/BestModels/best_{model_name.lower().replace(" ", "_")}_{vect_name}_model.pkl'
    joblib.dump(best_model, model_filename)

    print(f"Saved best {model_name} model to {model_filename}")




def predict_models_summary(X_test, y_test, model_name, vect_name):

    # evaluate models
    model_filename = f'./Model/BestModels/best_{model_name.lower().replace(" ", "_")}_{vect_name}_model.pkl'
    model = joblib.load(model_filename)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

