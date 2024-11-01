import streamlit as st
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import requests
from textblob import TextBlob

st.title("Stock Dashboard")

#Stock Symbol Input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value = "AAPL")

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Fetch data from Yahoo Finance
#st.write(f"Fetching data for {stock_symbol} from Yahoo Finance...")
stock_data = yf.Ticker(stock_symbol)
df = stock_data.history(start = start_date, end = end_date)

# Display current price
st.write("### Latest Price")
st.write(df.tail())


# Plot the stock's close price history
st.subheader("Stock Close Price History")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"))
fig.update_layout(title="Closing Price Over Time", xaxis_title="Date", yaxis_title="Close Price (USD)")
st.plotly_chart(fig)

# Comparative Analysis
st.write("## Comparative Analysis")

# Comparative Analysis (if you want to compare multiple stocks)
st.sidebar.subheader("Comparative Analysis")
compare_symbol = st.sidebar.text_input("Enter another Stock Symbol to Compare", value="MSFT")

if compare_symbol:
    compare_data = yf.Ticker(compare_symbol)
    df_compare = compare_data.history(start=start_date, end=end_date)

    # Comparative plot
    st.subheader("Comparative Analysis with Another Stock")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f"{stock_symbol} Close Price"))
    fig_compare.add_trace(go.Scatter(x=df_compare.index, y=df_compare['Close'], mode='lines', name=f"{compare_symbol} Close Price"))
    fig_compare.update_layout(title=f"{stock_symbol} vs {compare_symbol} Close Price", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_compare)

# Moving Averages
st.subheader("Moving Averages")
df['MA_50'] = df['Close'].rolling(50).mean()
df['MA_200'] = df['Close'].rolling(200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], mode='lines', name="50-day MA"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], mode='lines', name="200-day MA"))
fig.update_layout(title="Stock Price with 50-day and 200-day Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig)


# Sentiment Analysis Section
st.subheader("Sentiment Analysis on News Headlines")

# User input for stock symbol
ticker = st.text_input("Enter Stock Symbol for Sentiment Analysis", "AAPL")

# Fetch news headlines related to the stock ticker
def fetch_headlines(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [article["title"] for article in articles]
    else:
        st.error("Error fetching news headlines. Please check your API key or try again later.")
        return []

# Display and analyze sentiment if a ticker is entered
if ticker:
    st.write(f"Recent News Sentiment for {ticker}")
    headlines = fetch_headlines(ticker)
    
    if headlines:
        sentiment_data = {"Headline": [], "Sentiment": []}
        
        for headline in headlines[:5]:  # Limit to top 5 headlines
            analysis = TextBlob(headline)
            sentiment_score = analysis.sentiment.polarity
            sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            sentiment_data["Headline"].append(headline)
            sentiment_data["Sentiment"].append(sentiment)
        
        # Create DataFrame for organized display
        sentiment_df = pd.DataFrame(sentiment_data)
        st.table(sentiment_df)
    else:
        st.write("No recent headlines found.")


# Interactive Filtering by Volume
st.sidebar.subheader("Interactive Filtering")
min_volume = st.sidebar.slider("Minimum Trading Volume", int(df['Volume'].min()), int(df['Volume'].max()), int(df['Volume'].min()))
filtered_df = df[df['Volume'] >= min_volume]

st.subheader("Filtered Data by Trading Volume")
st.write(filtered_df)

# Downloadable CSV
csv = df.to_csv().encode('utf-8')
st.sidebar.download_button(label="Download CSV", data=df.to_csv().encode("utf-8"), file_name=f"{stock_symbol}_data.csv", mime="text/csv")
 
