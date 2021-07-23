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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,RepeatedKFold,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,classification_report,roc_curve,auc,roc_auc_score
from xgboost import plot_tree
import mlflow
import mlflow.sklearn

path='data/response.csv'
repo='D:/Stella/Documents/10_Academy/Week-2/abtest-mlops'
version='v1'

data_url=dvc.api.get_url(path=path,
                         repo=repo,
                         rev=version)


if __name__ == "__main__":
    warnings.filterwarningd("ignore")
    np.random.seed(40)

    response=pd.read_csv(data_url,sep=",")
    lb_make = LabelEncoder()
    # encode the categorical variables using label encoding
    response['device_make'] = lb_make.fit_transform(response['device_make'])
    response['browser'] = lb_make.fit_transform(response['browser'])
    # split the data into train, validation and test
    train, validate, test = np.split(response.sample(frac=1, random_state=42),
                                     [int(.7 * len(response)), int(.9 * len(response))])
    # separate the response variable from the dataset
    y_train, y_validation, y_test = (train['awareness'], validate['awareness'], test['awareness'])
    x_train, x_validation, x_test = (train.drop(['awareness'], axis=1), validate.drop(['awareness'], axis=1),
                                     test.drop(['awareness'], axis=1))
