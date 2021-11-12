import os
from textwrap import indent
from scipy.sparse.construct import random
import argparse
import pandas as pd
from get_data import read_params
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import numpy as np
import json
import joblib

def evaluate_metrics(actual,pred):
    rmse= np.sqrt(mean_squared_error(actual,pred))
    mae= mean_absolute_error(actual,pred)
    r2= r2_score(actual,pred)

    return rmse, mae, r2


def train_and_test(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir=config["model_dir"]
    alpha = config["Estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["Estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target = config["base"]["target_col"]


    train_data = pd.read_csv(train_data_path,sep=',')
    test_data = pd.read_csv(test_data_path,)
    
    train_y = train_data[target]
    test_y = test_data[target]
    train_x = train_data.drop(target,axis=1)
    test_x = test_data.drop(target,axis=1)

    lr = ElasticNet(alpha = alpha ,l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x,train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse,mae,r2) = evaluate_metrics(test_y, predicted_qualities)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
#################################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    with open(scores_file,"w") as f:
        scores={
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        json.dump(scores,f,indent=4)
    with open(params_file,'w') as f:
        params={
            "l1_ratio":l1_ratio,
            "alpha":alpha
        }
        json.dump(params,f,indent=4)
    os.makedirs("saved_models",exist_ok=True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(lr,model_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_test(parsed_args.config)