import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import collections
#-----------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#-----------------------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer,confusion_matrix
from sklearn.metrics import plot_roc_curve,plot_precision_recall_curve,average_precision_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold
from sklearn.model_selection import cross_val_score
#-----------------------------------------------------------------------------------------------------------------
import datetime
import pickle
from copy import deepcopy
#-----------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")



def XGBC(X_train,y_train,X_test,y_test,scale_pos_weight):  
    classifier = XGBClassifier(learning_rate=0.1,n_estimators=100,scale_pos_weight=scale_pos_weight)
    classifier.fit(X_train, y_train)
    y_pred_prob=classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,shuffle=True)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision= precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'XGBoost',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score
    
def GradientBoostingC(X_train,y_train,X_test,y_test):  
    classifier = GradientBoostingClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred_prob=classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,shuffle=True)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision= precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'GradientBoostingClassifier',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score


def LogisticReg(X_train,y_train,X_test,y_test):  
    classifier = LogisticRegression(random_state = 42)
    classifier.fit(X_train, y_train)
    y_pred_prob=classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'LogisticRegression',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score


def RandomForest(X_train,y_train,X_test,y_test):  
    classifier = RandomForestClassifier(random_state = 42)
    classifier.fit(X_train, y_train)
    y_pred_prob=classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'RandomForestClassification',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score

def KNN(X_train,y_train,X_test,y_test):  
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_pred_prob=classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'KNNClassifier',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score

def SVM_C(X_train,y_train,X_test,y_test):  
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'SVC',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score
    
    
def NaiveBayes(X_train,y_train,X_test,y_test):  
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'NaiveBayes',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score
    
    
def LGBM(X_train,y_train,X_test,y_test):  
    classifier = lgb.LGBMClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    val_accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    kfold = StratifiedKFold(n_splits=5,random_state=1881)
    mdl_cross_val_score = np.mean(cross_val_score(classifier, X_train, y_train, cv=kfold))
    f1 =  f1_score(y_test, y_pred, average='weighted')
    plt.style.use('seaborn')
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    plot_roc_curve(classifier,X_test,y_test,ax=ax[0])
    plot_precision_recall_curve(classifier,X_test,y_test,ax=ax[1])
    ax[0].set_title("ROC Curve")
    ax[1].set_title("Precision vs Recall Curve")
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    plt.style.use('default')
    plot_confusion_matrix(classifier,X_test,y_test,cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    score={"Classifier":'LightGBM',
            "Accuracy":val_accuracy,
           "Cross_val_score":mdl_cross_val_score,
           "f1-score":f1,
          "precision":precision,
          "recall":recall}
    return classifier,score

    
def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names

    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')