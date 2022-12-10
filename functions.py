from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
import yfinance as yf
import plotly
import plotly.express as px
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries
import torch
import flair

#utilize M1 metal performance shaders for training
flair.device = torch.device("mps")

"""Code for fetching news articles for given Stock ticker"""
news_url = 'https://finviz.com/quote.ashx?t='

def get_html(ticker):
  url = news_url + ticker
  req = urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}))
  html = BeautifulSoup(req, 'html.parser')
  table = html.find(id = 'news-table')
  return table

"""Get List of Tickers from SP500"""
def get_tickers():
  req = urlopen(Request('https://www.slickcharts.com/sp500', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}))
  html = BeautifulSoup(req, 'html.parser').find_all('td')
  html = [html[x+2:x+4] for x in range(0,len(html),7)][:50]
  tickers = []
  for i in html:
    tickers.append(i[0].find('a').get_text())
  if 'BRK.B' in tickers:
    tickers[tickers.index('BRK.B')] = 'BRK-B'
  return tickers


"""Storing news headlines in pandas dataframe"""
def headlines_df(table):
  news = []
    
  for i in table.findAll('tr'):
      text = i.find('a')
      if text is not None:
        text = text.get_text()
      else:
        continue
      date_scrape = i.td.text.split()

      if len(date_scrape) == 1:
          time = date_scrape[0]
      else:
          date = date_scrape[0]
          time = date_scrape[1]
      
      news.append([date, time, text])

      news_df = pd.DataFrame(news, columns = ['date','time', 'news'])        

      news_df['datetime'] = pd.to_datetime(news_df['date'] + ' ' + news_df['time'])
      
  return news_df


"""Flair Financial Corpus Trained Text Classifier"""
classifier = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')
#classifier = TextClassifier.load('en-sentiment') #pretrained classifiers


"""Calculate sentiment for each headline and add to dataframe"""
def flair_sentiment_score(news_df):
  scores = []

  for s in news_df['news'].values.tolist():
    sentence = Sentence(s)
    classifier.predict(sentence)
    scores.append(sentence.to_dict())

  rawscores = []
  for s in scores:
    rawscores.append(int(s['all labels'][0]['value'])*s['all labels'][0]['confidence'])

  news_df['sentiment'] = rawscores

"""Generate dict of sentiments for each ticker"""

def generate_sentiment_all(tickers):
  scores = []
  for ticker in tickers:
    sent = headlines_df(get_html(ticker))
    flair_sentiment_score(sent)
    scores.append(sent['sentiment'].mean())
  return pd.DataFrame({'Sentiment':scores}, index = tickers)

def generate_ticker_info(tickers):
  prices,sectors,industries,cap,names = [],[],[],[],[]
  for ticker in tickers:
    info = yf.Ticker(ticker).info
    prices.append(info['regularMarketPrice'])
    sectors.append(info['sector'])
    industries.append(info['industry'])
    cap.append(info['marketCap'])
    names.append(info['shortName'])
  data = {'Price':prices,'Sector':sectors,'Industry':industries,'Market Cap':cap,'Name':names}
  return pd.DataFrame(data, index = tickers)

#generates dataframe with stock and sentiment info to be displayed on main page
def generate_main_df(tickers, num):
  if num > 49:
    num = 49
  avg_sentiments = generate_sentiment_all(tickers[:num])
  metadata_df = generate_ticker_info(tickers[:num])
  df = metadata_df.join(avg_sentiments)
  return df

"""Displays the tree graph for stocks in S&P500. Color corresponds to sentiment"""
def graph_tree(df):
  fig = px.treemap(df, path=[px.Constant("Sectors"), 'Sector', 'Industry', df.index], values='Market Cap',
                    color='Sentiment', hover_data=['Name','Price','Sentiment'],
                    color_continuous_scale=['#D10000', "#bbbbbb", '#009900'],
                    color_continuous_midpoint=0,
                    title=f'S&P500 Top {df.shape[0]} Stocks')

  fig.update_traces(textposition="middle center")
  fig.update_layout(margin = dict(t=65, l=25, r=25, b=25), font_size=20)
  return fig



"""Get Average Sentiment over Time Interval"""
def get_avg_sentiment(df, date):
  return df.loc[df['datetime'] < date]['sentiment'].mean()


"""Stock Data"""
ts = TimeSeries(key='J06KDLHK8T0N24N8',output_format='pandas') # api key
def generate_price_df(ticker):
  df = ts.get_intraday(symbol=ticker,interval='30min',outputsize='compact')[0]
  df['symbol'] = ticker
  return df

def join_df(price_data, sentiment_data):
  start_date = max(price_data.index.min(), sentiment_data['datetime'].min())
  end_date = min(price_data.index.max(), sentiment_data['datetime'].max())
  df = price_data
  df = df.loc[df.index >= start_date]
  df = df.loc[df.index <= end_date]
  
  s = []
  for t in df.index.values:
    s.append(get_avg_sentiment(sentiment_data, t))
  df['sentiment'] = s

  return df

def generate_training_data(symbols):
  dfs = []

  for s in symbols:
    sent = headlines_df(get_html(s))
    flair_sentiment_score(sent)
    price = generate_price_df(s)
    joined = join_df(price, sent)
    dfs.append(joined)

  return dfs[0]


def graph_hourly_sentiment(news_df, ticker):
  mean_scores = news_df.set_index('datetime').resample('H').mean() 
  fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment', title = ticker + ' Hourly Sentiment Scores')
  return fig

def graph_daily_sentiment(news_df, ticker):
  mean_scores = news_df.set_index('datetime').resample('D').mean() 
  fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment', title = ticker + ' Daily Sentiment Scores')
  return fig

def graph_stock(ticker):
  df = yf.Ticker(ticker)
  df = df.history(interval="15m",period="5d")
  fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
  fig.update_layout(title=ticker+' Stock Price')
  fig.update_xaxes(
        rangeslider_visible=True,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
        ]
    )
  return fig




