import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title('Stock Price Prediction App')

# Define a function to fetch stock data
def get_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

# Define a function to retrieve news articles related to a stock symbol
def get_stock_news(symbol):
    newsapi = NewsApiClient(api_key='ef7280046c0d433bbf74a27166e796b5')  # Replace with your News API key
    news = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page=1, page_size=5)  # Limit to top 5 articles
    return news

# Define a function for sentiment analysis using VADER
def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

# Get a list of available stock symbols
available_symbols = yf.Tickers('AAPL MSFT GOOGL AMZN INFY WMT IBM MU BA AXP').tickers

# Sidebar - Input parameters
st.sidebar.header('Input Parameters')

# Create a multi-select dropdown for selecting multiple stock symbols
selected_symbols = st.sidebar.multiselect('Select Stock Symbols', available_symbols)

# Create date pickers for start and end dates
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2021-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2022-01-01'))

# Fetch stock data for selected symbols
data = pd.DataFrame()
for symbol in selected_symbols:
    symbol_data = get_data(symbol, start_date, end_date)
    data = pd.concat([data, symbol_data['Adj Close']], axis=1)
data.columns = selected_symbols

st.subheader('Raw Data')
st.write(data)

# Predict stock price
st.subheader('Stock Price Prediction')
n_years = st.number_input('Years of Prediction:', 1, 10, 1)
n_days = n_years * 365
df = data[selected_symbols]
df['Prediction'] = df[selected_symbols[0]].shift(-n_days)  # Assume the prediction is based on the first selected symbol

X = np.array(df.drop(['Prediction'], axis=1))
X = X[:-n_days]
y = np.array(df['Prediction'])
y = y[:-n_days]

# Check if there's enough data for the split
if len(X) > int(len(X) * 0.2):  # Check if data size is greater than 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    st.subheader('Model Evaluation')
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R2 Score: {r2}')

    # Predict the future stock price
    predict_df = df[selected_symbols].tail(n_days)
    prediction = model.predict(np.array(predict_df))
    st.subheader('Predicted Stock Price')
    st.write(predict_df)
    st.line_chart(predict_df)
else:
    st.write("Not enough data to perform the split. Please choose a larger date range.")

# Comparison of selected stocks
st.subheader('Comparison of Selected Stocks')
st.line_chart(data)

# News and Sentiment Analysis
st.subheader('News and Sentiment Analysis')

# Retrieve and display news articles for the selected symbols
for symbol in selected_symbols:
    st.subheader(f'News for {symbol}:')
    news = get_stock_news(symbol)
    for article in news['articles']:
        st.write(article['title'])
        st.write(f"Published at {article['publishedAt']}")
        st.write(article['description'])
        sentiment = perform_sentiment_analysis(article['title'])
        st.write(f"Sentiment Score: {sentiment['compound']}")

