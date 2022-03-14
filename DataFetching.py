# Imports
import requests
import json
import pandas as pd
import numpy as np

URL = "https://recruitment.aimtechnologies.co/ai-tasks"
raw_data = pd.read_csv("C:/Users/HP/AIM_Dialect_Prediction/dialect_dataset.csv")

raw_data = raw_data.astype(str)

def fetch(col, chunck_size):
    """
    """
    tweets = np.array([])
    lst = raw_data[col].tolist()
    chunks = [lst[i:i + 1000] for i in range(0, len(lst), chunck_size)]
    for chunck in chunks:
        json_chunck = json.dumps(chunck)
        r = requests.post(URL, data = json_chunck)
        tweets_chunck = json.loads(r.text)
        tweets_chunck_list = list(tweets_chunck.values())
        tweets = np.append(tweets, tweets_chunck_list)
    
    tweets = tweets.flatten()
    
    return tweets


tweets = fetch('id', 1000)

raw_data['Tweets'] = tweets

raw_data.to_pickle("C:/Users/HP/AIM_Dialect_Prediction/tweets_data.pkl")