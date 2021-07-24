import dvc.api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix,plot_confusion_matrix
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc, \
    roc_auc_score
from xgboost import plot_tree
import mlflow
import mlflow.sklearn

import os
import warnings
import sys
import pathlib
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("./data").resolve()

path = DATA_PATH.joinpath("response.csv")
repo = 'D:/Stella/Documents/10_Academy/Week-2/abtest-mlops'
version = 'v2'

data_url = dvc.api.get_url(path=path,
                           repo=repo,
                           rev=version)

mlflow.set_experiment("abtest")

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    response = pd.read_csv(data_url, sep=",")
    lb_make = LabelEncoder()
    # encode the categorical variables using label encoding
    response['browser'] = lb_make.fit_transform(response['browser'])
    if 'browser' in response.columns:
        response['device_make'] = lb_make.fit_transform(response['device_make'])
    else:
        response=response
    # split the data into train, validation and test
    train, validate, test = np.split(response.sample(frac=1, random_state=42),
                                     [int(.7 * len(response)), int(.9 * len(response))])
    # separate the response variable from the dataset
    y_train, y_validation, y_test = (train['awareness'], validate['awareness'], test['awareness'])
    x_train, x_validation, x_test = (train.drop(['awareness'], axis=1), validate.drop(['awareness'], axis=1),
                                     test.drop(['awareness'], axis=1))

    C = 1

    # Run ElasticNet
    clf_log = LogisticRegression(C=C, random_state=0)
    log_result = clf_log.fit(x_train, y_train)
    predictions_log = clf_log.predict(x_validation)
    (accuracy, precision, recall) = eval_metrics(y_validation, predictions_log)

    # Print out ElasticNet model metrics
    print("Logistic Regression model (C=%f):" % (C))
    print("  Accuracy: %s" % accuracy)
    print("  Precision: %s" % precision)
    print("  Recall: %s" % recall)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("C", C)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(clf_log, "model")

    log_cnm = confusion_matrix(y_validation, predictions_log)

    # plot confusion matrix
    plt.figure(figsize=(6, 5))
    # create heatmap
    plot_confusion_matrix(clf_log, x_validation, y_validation)
    # plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    plt.close()

    # plot roc curve
    y_valid_proba = clf_log.predict_proba(x_validation)[::, 1]
    fpr, tpr, _ = roc_curve(y_validation, y_valid_proba)
    auc = roc_auc_score(y_validation, y_valid_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.savefig("roc.png")
    mlflow.log_artifact("roc.png")
    plt.close()

    #plot feature importance
    importances = clf_log.coef_[0]
    labels = response.columns
    feature_df = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importance"])
    feature_df = feature_df.sort_values(by='importance', ascending=False, )
    # image formatting
    axis_fs = 18  # fontsize
    title_fs = 22  # fontsize
    sns.set(style="whitegrid")

    ax = sns.barplot(x="importance", y="feature", data=feature_df)
    ax.set_xlabel('Importance', fontsize=axis_fs)
    ax.set_ylabel('Feature', fontsize=axis_fs)  # ylabel
    ax.set_title('Logistic feature importance', fontsize=title_fs)

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    mlflow.log_artifact("feature_importance.png")
    plt.close()
