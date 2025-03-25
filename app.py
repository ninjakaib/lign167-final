from functions import *
from flask import Flask, render_template
import flask
import json

app = Flask(__name__)

tickers = get_tickers()
num_stocks = 20
df = generate_main_df(tickers, num_stocks)
fig = graph_tree(df)
fig.update_layout(width=1250, height=700)


@app.route('/')
def index():
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('index.html', graphJSON=graphJSON)


@app.route('/sentiment', methods=['POST'])
def sentiment():
	ticker = flask.request.form['ticker'].upper()
	sent_df = headlines_df(get_html(ticker))
	flair_sentiment_score(sent_df)
	sentiment = sent_df['sentiment'].mean()

	fig_hourly = graph_hourly_sentiment(sent_df, ticker)
	fig_daily = graph_daily_sentiment(sent_df, ticker)
	fig_stock = graph_stock(ticker)

	graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_stock = json.dumps(fig_stock, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template(
		'sentiment.html',
		graphJSON_hourly=graphJSON_hourly,
		graphJSON_daily=graphJSON_daily,
		graphJSON_stock=graphJSON_stock,
		table=sent_df.to_html(classes='data'),
		sentiment=sentiment,
	)
