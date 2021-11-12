import os
import pandas as pd
import yaml
import argparse
from get_data import get_data,read_params
from sklearn.preprocessing import LabelEncoder


def load_and_save(config_path):
    config  = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ","_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    cols = ["CarName","fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"]
    le = LabelEncoder()
    df[cols] = df[cols].apply(le.fit_transform)

    df.to_csv(raw_data_path,sep=",",header=new_cols,index=False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(parsed_args.config)