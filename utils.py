

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from constants import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,plot_confusion_matrix
from sklearn.metrics import roc_auc_score,recall_score,precision_score,cohen_kappa_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from pycaret.classification import *
#from Lead_scoring_training_pipeline.constants import *

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,plot_confusion_matrix
from sklearn.metrics import roc_auc_score,recall_score,precision_score,cohen_kappa_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
import mlflow
import mlflow.sklearn

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES =['first_platform_c_Level10', 'first_platform_c_Level11','first_platform_c_Level12', 'first_platform_c_Level13',
'first_platform_c_Level14', 'first_platform_c_Level15','first_platform_c_Level17', 'first_platform_c_Level18',
'first_platform_c_Level19', 'first_platform_c_Level2','first_platform_c_Level20', 'first_platform_c_Level21',
'first_platform_c_Level22', 'first_platform_c_Level26', 'city_tier_2.0','first_platform_c_Level3', 'city_tier_3.0', 'first_platform_c_Level1',
'first_platform_c_Level28', 'first_platform_c_Level29', 'first_platform_c_Level27',
'first_platform_c_Level4', 'first_platform_c_Level25','first_platform_c_Level24', 'first_platform_c_Level40',
'first_platform_c_Level39', 'first_platform_c_Level38','first_platform_c_Level32', 'first_platform_c_Level23',
'first_platform_c_Level34', 'first_platform_c_Level43','first_platform_c_Level42', 'first_platform_c_Level36',
'first_platform_c_Level16', 'first_platform_c_Level33','first_platform_c_Level41', 'first_platform_c_Level35',
'first_platform_c_Level37', 'first_platform_c_Level30','first_platform_c_Level31']

# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['city_tier', 'first_platform_c', 'first_utm_medium_c','first_utm_source_c']


def create_sqlit_connection(DB_PATH,DB_FILE_NAME):
    """ create a database connection to a SQLite database """
    conn = None
    # opening the conncetion for creating the sqlite db
    try:
        conn = sqlite3.connect(DB_PATH+DB_PATH)
        print(sqlite3.version)
    # return an error if connection not established
    except Error as e:
        print(e)
    # closing the connection once the database is created
    finally:
        if conn:
            conn.close()

###############################################################################
# Define the function to encode features
# ##############################################################################

def encode_features(DB_PATH,DB_FILE_NAME):
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df=pd.read_sql('Select * from cleaned_data',conn)
    df['city_tier']=df['city_tier'].astype('object')
    index_names = df[df['total_leads_dropped'] == 'others' ].index
  
    # drop these row indexes
    # from dataFrame
    df.drop(index_names, inplace = True)
    df=pd.get_dummies(df, columns=['city_tier', 'first_platform_c', 'first_utm_medium_c','first_utm_source_c'] ,drop_first=True)
    df_target_features=df.pop('app_complete_flag')
    df_predictor_features=df
    #Connect the database
    
    print("Connection Successful",conn)
    df_predictor_features.to_sql('features',con=conn,if_exists='replace',index=False)
    df_target_features.to_sql('target',con=conn,if_exists='replace',index=False)
    
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''
    

###############################################################################
# Define the function to train the model
# ##############################################################################
    
          
def get_train_model(DB_PATH,DB_FILE_NAME,ml_flow_path):
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    X=pd.read_sql('Select * from features',conn)
    y=pd.read_sql('Select * from target',conn)

    # using the train test split function
    X_train,X_test,y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.25,shuffle=True)
    
    #ONE_HOT_ENCODED_FEATURES =
    X_train=X_train[ONE_HOT_ENCODED_FEATURES]

    X_test=X_test[ONE_HOT_ENCODED_FEATURES]
    mlflow.set_tracking_uri("http://0.0.0.0:6007")
    
    
    model_config = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'device':'gpu',
        'importance_type': 'split' ,
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'num_leaves': 31,
        'objective': None,
        'random_state': 42,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'silent': 'warn',
        'subsample': 1.0,
        'subsample_for_bin': 200000 ,
        'subsample_freq': 0
        }


    #Model Training
    with mlflow.start_run(run_name='run_LightGB') as mlrun:
       # evaluate the model
        Lgbmc = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                       device='gpu', importance_type='split', learning_rate=0.1,
                       max_depth=-1, min_child_samples=20, min_child_weight=0.001,
                       min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=31,
                       objective=None, random_state=42, reg_alpha=0.0, reg_lambda=0.0,
                       silent='warn', subsample=1.0, subsample_for_bin=200000,
                       subsample_freq=0)
        model=Lgbmc.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(sk_model=Lgbmc,artifact_path="models", registered_model_name='LightGBM')
        mlflow.log_params(model_config) 
         #Log metrics
        y_pred =model.predict(X_test)
        acc=accuracy_score(y_pred, y_test)
        conf_mat = confusion_matrix(y_pred, y_test)
        precision = precision_score(y_pred, y_test,average= 'macro')
        recall = recall_score(y_pred, y_test, average= 'macro')
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label='0')
        class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label='1')
        f1_score =model.best_score_

        mlflow.log_metric('test_accuracy', acc)
        #mlflow.log_metric("f1", f1_score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision_0", class_zero[0])
        mlflow.log_metric("Precision_1", class_one[0])
        mlflow.log_metric("Recall_0", class_zero[1])
        mlflow.log_metric("Recall_1", class_one[1])
        mlflow.log_metric("f1_0", class_zero[2])
        mlflow.log_metric("f1_1", class_one[2])
        mlflow.log_metric("False Negative", fn)
        mlflow.log_metric("True Negative", tn)
       
    
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(ml_flow_path)
        # Predict on a Pandas DataFrame.
        X_test=X_test[ONE_HOT_ENCODED_FEATURES]
        predictions = loaded_model.predict_proba(pd.DataFrame(X_test))
        print (pd.DataFrame(predictions,columns=["Prob of Not Churn","Prob of Churn"]).head()) 
        pd.DataFrame(predictions,columns=["Prob of Not Churn","Prob of Churn"]).to_sql(name='Final_Predictions', con=conn,if_exists='replace',index=False)
        
        
        
        def print_score(clf, X_train, y_train, X_test, y_test, train=True):
            '''
            print the accuracy score, classification report and confusion matrix of classifier
            '''
            if train:
                '''
                training performance
                '''
                print("Train Result:\n")
                print('Plot:Confusion Matrix\n {}\n'.format(plot_confusion_matrix(clf,X_train,y_train,values_format='d',display_labels= 
                                                                                  ['Not','Yes'])))
                print("accuracy score: {0:.5f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
                print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train),digits=5)))
                print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



            elif train==False:
                '''
                test performance
                '''
                print("Test Result:\n")
                print('Plot:Confusion Matrix\n {}\n'.format(plot_confusion_matrix(clf,X_test,y_test,values_format='d',display_labels=
                                                                                  ['Not','Yes'])))
                print("accuracy score: {0:.5f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
                print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test),digits=5)))
                print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test)))) 

        from sklearn.metrics import classification_report
        print_score(model, X_train, y_train, X_test, y_test, train=True)
        print_score(model, X_train, y_train, X_test, y_test, train=False)


