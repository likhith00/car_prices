import os 
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import yaml
from get_data import read_params


def split_data(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path,sep=',')
    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state)
    train.to_csv(train_data_path,sep=',',index=False)
    test.to_csv(test_data_path,sep=',',index=False)



if __name__=="__main__":
    args= argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    split_data(parsed_args.config)