from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

stopwords = set(stopwords.words('english'))

regressor = pickle.load(open('model/model.sav', 'rb'))

# Tokenization is the process of tokenizing or splitting a string, text into a list of tokens
def tokenize(txt):
  return word_tokenize(txt)

# n_unique_tokens: Rate of unique words in the content
def n_unique_tokens(txt):
  txt = tokenize(txt)
  words = list(set(txt))   # set only stores unique values
  return len(words)/len(txt)

# average_token_length: Average length of the words in the content
def avg_token_length(txt):
    txt = tokenize(txt)
    length = []
    for i in txt:
      length.append(len(i))
    return np.average(length)


# n_non_stop_words: Rate of non-stop words in the content
# n_non_stop_unique_tokens: Rate of unique non-stop words in content
def n_nonstop_words(txt):
    txt = tokenize(txt)
    nonstop_words = [i for i in txt if not i in stopwords]
    n_nonstop_words = len(nonstop_words)/len(txt)
    nonstop_unique_words = list(set(nonstop_words))
    n_nonstop_unique_tokens = len(nonstop_unique_words)/len(txt)
    return n_nonstop_words, n_nonstop_unique_tokens

# Polar words
def polarity(txt):
  positive_words = []
  negative_words = []

  tokenize_txt = tokenize(txt)

  for i in tokenize_txt:
    blob = TextBlob(i)
    polarity = blob.sentiment.polarity
    if polarity > 0:
      positive_words.append(i)
    if polarity < 0:
      negative_words.append(i)

  return positive_words, negative_words

#Polarity_rates
def rates(txt):
  txt = polarity(txt)
  positive_words = txt[0]
  negative_words = txt[1]
  global_rate_positive_words = (len(positive_words)/len(txt))/100
  global_rate_negative_words = (len(negative_words)/len(txt))/100
  positive_polarity = []
  negative_polarity = []
  for i in positive_words:
    blob_a = TextBlob(i)
    positive_polarity.append(blob_a.sentiment.polarity)
  for j in negative_words:
    blob_b = TextBlob(j)
    negative_polarity.append(blob_b.sentiment.polarity)
  min_positive_polarity = min(positive_polarity, default=0)
  min_negative_polarity = min(negative_polarity, default=0)
  max_positive_polarity = max(positive_polarity, default=0)
  max_negative_polarity = max(negative_polarity, default=0)
  avg_positive_polarity = np.average(positive_polarity)
  if np.isnan(avg_positive_polarity):
    avg_positive_polarity = 0
  avg_negative_polarity = np.average(negative_polarity)
  return (global_rate_positive_words, global_rate_negative_words,
          avg_positive_polarity, min_positive_polarity,
          max_positive_polarity, avg_negative_polarity,
          min_negative_polarity, max_negative_polarity)


def processArticle(article):
  content = {}
  blob = TextBlob(article['text'])
  title_blob = TextBlob(article['title'])

  content['title'] = article['title']
  content['n_tokens_title'] = len(tokenize(article['title']))
  content['n_tokens_content'] = len(tokenize(article['text']))
  content['n_unique_tokens'] = n_unique_tokens(article['text'])
  content['n_non_stop_words'] = n_nonstop_words(article['text'])[0]
  content['n_non_stop_unique_tokens'] = n_nonstop_words(article['text'])[1]
  content['num_hrefs'] = article['num_links'] # article.html.count("https://www.newindianexpress.com")
  content['num_imgs'] = len(article['images'])
  content['num_videos'] = len(article['movies'])
  content['average_token_length'] = avg_token_length(article['text'])
  content['num_keywords'] = len(article['keywords'])

  if article['category'] == "lifestyle":
    content['data_channel_is_lifestyle'] = 1
  else:
    content['data_channel_is_lifestyle'] = 0

  if article['category'] == "entertainment":
    content['data_channel_is_entertainment'] = 1
  else:
    content['data_channel_is_entertainment'] = 0

  if article['category'] == "business":
    content['data_channel_is_bus'] = 1
  else:
    content['data_channel_is_bus'] = 0

  if "social media" or "facebook" or "whatsapp" in article['text'].lower():
    data_channel_is_socmed = 1
    data_channel_is_tech = 0
    data_channel_is_world = 0
  else:
    data_channel_is_socmed = 0
  if ("technology" or "tech" in article['text'].lower()) or ("tech" in article['category']):
    data_channel_is_tech = 1
    data_channel_is_socmed = 0
    data_channel_is_world = 0
  else:
    data_channel_is_tech = 0
  if "world" in article['keywords']:
    data_channel_is_world = 1
    data_channel_is_tech = 0
    data_channel_is_socmed = 0
  else:
    data_channel_is_world = 0

  content['data_channel_is_socmed'] = data_channel_is_socmed
  content['data_channel_is_tech'] = data_channel_is_tech
  content['data_channel_is_world'] = data_channel_is_world
  
  day = datetime.strptime(article['date'], '%a, %d %b %Y %H:%M:%S %Z').strftime("%A")
  if day == "Monday":
    content['weekday_is_monday'] = 1
  else:
    content['weekday_is_monday'] = 0
  if day == "Tuesday":
    content['weekday_is_tuesday'] = 1
  else:
    content['weekday_is_tuesday'] = 0
  if day == "Wednesday":
    content['weekday_is_wednesday'] = 1
  else:
    content['weekday_is_wednesday'] = 0
  if day == "Thursday":
    content['weekday_is_thursday'] = 1
  else:
    content['weekday_is_thursday'] = 0
  if day == "Friday":
    content['weekday_is_friday'] = 1
  else:
    content['weekday_is_friday'] = 0
  if day == "Saturday":
    content['weekday_is_saturday'] = 1
    content['is_weekend'] = 1
  else:
    content['weekday_is_saturday'] = 0
  if day == "Sunday":
    content['weekday_is_sunday'] = 1
    content['is_weekend'] = 1
  else:
    content['weekday_is_sunday'] = 0
    content['is_weekend'] = 0
        
  content['global_subjectivity'] = blob.sentiment.subjectivity
  content['global_sentiment_polarity'] = blob.sentiment.polarity

  text_rates = rates(article['text'])

  content['global_rate_positive_words'] = text_rates[0]
  content['global_rate_negative_words'] = text_rates[1]
  content['min_positive_polarity'] = text_rates[3]
  content['max_positive_polarity'] = text_rates[4]
  content['avg_positive_polarity'] = text_rates[2]
  content['avg_negative_polarity'] = text_rates[5]
  content['min_negative_polarity'] = text_rates[6]
  content['max_negative_polarity'] = text_rates[7]

  content['title_subjectivity'] = title_blob.sentiment.subjectivity
  content['title_sentiment_polarity'] = title_blob.sentiment.polarity

  return content

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/')
def root():
  return 'STATUS OK'

@app.route('/api/predict', methods = ['POST'])
def predict():
  payload = request.get_json()
  product = processArticle(payload)

  df = pd.DataFrame([product])
  test_df = df.drop(['title'], axis=1)
  result = regressor.predict(test_df)

  return jsonify({ 'status': 'ok', 'result': result[0] })

if __name__ == '__main__':
  app.run('0.0.0.0', 5000)
