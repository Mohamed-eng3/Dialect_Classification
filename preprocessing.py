#Imports
import pandas as pd
import re
import string
from nltk.corpus import stopwords

data = pd.read_pickle("C:/Users/HP/AIM_Dialect_Prediction/tweets_data.pkl")

stopwords_list = stopwords.words('arabic')

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_punct(text):
    return re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)

def remove_stopwords(text):
    filtered_words = [word for word in text.split() if word not in stopwords_list]
    return " ".join(filtered_words)

def remove_username(text):
    return re.sub('[A-Za-z0-9_]+', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def remove_extra_spaces(text):
    return re.sub('\s+', ' ', text)


data["Tweets"] = data.Tweets.map(remove_URL) 
data["Tweets"] = data.Tweets.map(remove_punct)
data["Tweets"] = data.Tweets.map(remove_stopwords)
data["Tweets"] = data.Tweets.map(remove_username)
data["Tweets"] = data.Tweets.map(remove_emoji)
data["Tweets"] = data.Tweets.map(remove_extra_spaces)

data.to_pickle("C:/Users/HP/AIM_Dialect_Prediction/processed_data.pkl")