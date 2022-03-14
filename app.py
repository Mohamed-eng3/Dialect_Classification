import uvicorn
from fastapi import FastAPI
from preprocessing import remove_URL, remove_punct, remove_stopwords, remove_username, remove_emoji, remove_extra_spaces
from joblib import load

app = FastAPI()

@app.get('/')
def index():
    return {"text": "Hello AIM"}

@app.get('/predict/{tweet}')
def predict(tweet):
    #preprocessing
    tweet_ = remove_URL(tweet)
    tweet_ = remove_punct(tweet_)
    tweet_ = remove_stopwords(tweet_)
    tweet_ = remove_username(tweet_)
    tweet_ = remove_emoji(tweet_)
    tweet_ = remove_extra_spaces(tweet_)
    #classification
    clf = load('C:/Users/HP/AIM_Dialect_Prediction/svm_pipe.joblib')
    dialect = clf.predict([tweet_])
    return {"Tweet": tweet, "Dialect":dialect[0]}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
