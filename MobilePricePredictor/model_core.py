"""Dataset-native mobile price model definitions."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT/'data'/'mobile_phones.csv'
MODEL_PATH=ROOT/'models'/'mobile_price_model.joblib'
METRICS_PATH=ROOT/'models'/'metrics.json'
DB_PATH=ROOT/'data'/'user_features.sqlite3'
TARGET='price_inr'
CATEGORICAL=['brand','model']
NUMERIC=['rating','ram_gb','storage_gb','screen_inch','main_camera_mp','front_camera_mp','battery_mah']
FEATURES=CATEGORICAL+NUMERIC
MODEL_NAMES=['ridge_regression','random_forest','gradient_boosting','extra_trees']
MODEL_LABELS={'ridge_regression':'Ridge Regression','random_forest':'Random Forest','gradient_boosting':'Gradient Boosting','extra_trees':'Extra Trees'}
RANDOM_STATE=42
FEATURE_LABELS={'brand':'Brand','model':'Mobile model','rating':'User rating','ram_gb':'RAM (GB)','storage_gb':'Storage (GB)','screen_inch':'Screen size (inches)','main_camera_mp':'Rear camera (MP)','front_camera_mp':'Selfie camera (MP)','battery_mah':'Battery (mAh)'}
NUMERIC_RANGES={'rating':(1,5),'ram_gb':(1,32),'storage_gb':(1,2048),'screen_inch':(3,10),'main_camera_mp':(1,300),'front_camera_mp':(1,100),'battery_mah':(500,20000)}
def make_pipeline(name):
    prep=ColumnTransformer([('cats',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL),('nums',StandardScaler(),NUMERIC)])
    if name=='ridge_regression': reg=Ridge(alpha=2.0)
    elif name=='random_forest': reg=RandomForestRegressor(n_estimators=220,min_samples_leaf=2,max_features=0.9,n_jobs=-1,random_state=RANDOM_STATE)
    elif name=='gradient_boosting': reg=GradientBoostingRegressor(n_estimators=180,max_depth=2,learning_rate=.04,loss='huber',random_state=RANDOM_STATE)
    elif name=='extra_trees': reg=ExtraTreesRegressor(n_estimators=220,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=RANDOM_STATE)
    else: raise ValueError('Unknown model '+name)
    return TransformedTargetRegressor(regressor=Pipeline([('prep',prep),('model',reg)]),func=np.log1p,inverse_func=np.expm1,check_inverse=False)
def price_category(p):
    if p<10000:return 'Budget'
    if p<25000:return 'Mid-range'
    if p<50000:return 'Premium'
    return 'Flagship'
