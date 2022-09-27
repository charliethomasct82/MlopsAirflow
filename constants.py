DB_PATH = '/home/dags/Lead_scoring_training_pipeline'
DB_FILE_NAME ='/lead_scoring_data_cleaning.db'

DB_MLFLOW ='/Lead_scoring_mlflow_production.db' 
DB_EXPERIMENT_TRACKING='/lead_scoring_model_experimentation.db'
ML_RUN_FOLDER='/mlruns'
TRACKING_URL ="Http://0.0.0.0:6007/" 
 
ml_flow_path ='/home/dags/Lead_scoring_training_pipeline/mlruns/0/fa10837487c34e4ab7c4a48c45bde971/artifacts/models'

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

MODEL_CONFIG = {
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