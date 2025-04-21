# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import feedparser
# import torch
# import joblib
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from datetime import datetime
# from transformers import BertTokenizer

# from model import HybridModel
# from predict import predict
# from dataset import tickers

# # ========== CONFIG ==========
# st.set_page_config(page_title="📈 Stock Movement Prediction", layout="wide")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== LOAD MODEL ==========
# model = HybridModel(num_numeric_features=6, num_classes=3)
# state_dict = torch.load("models/hybrid_sentiment_model.pth", map_location=device)
# new_state_dict = {k.replace("fc_numeric", "numeric_fc"): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# scaler = joblib.load("models/scaler.pkl")
# label_encoder = joblib.load("models/label_encoder.pkl")

# # ========== CLEAN HEADLINE ==========
# def clean(text):
#     return re.sub(r"[^\w\s]", "", text).strip()

# # ========== GET TECHNICAL INDICATORS ==========
# def get_numeric_features(symbol):
#     try:
#         data = yf.Ticker(symbol).history(period="15d")
#         if data.empty or len(data) < 11:
#             return None, None

#         data["MA5"] = data["Close"].rolling(window=5).mean()
#         data["MA10"] = data["Close"].rolling(window=10).mean()
#         data["Volatility"] = ((data["High"] - data["Low"]) / data["Open"]) * 100

#         today = data.iloc[-1]
#         yesterday = data.iloc[-2]
#         numeric = {
#             "PriceBefore": yesterday["Close"],
#             "PriceAfter": today["Close"],
#             "ChangePercent": ((today["Close"] - yesterday["Close"]) / yesterday["Close"]) * 100,
#             "MA5": today["MA5"],
#             "MA10": today["MA10"],
#             "Volatility": today["Volatility"]
#         }

#         return numeric, data
#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         return None, None

# # ========== FETCH HEADLINES ==========
# def fetch_headlines(company):
#     query = f"{company} stock"
#     url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
#     return feedparser.parse(url).entries[:5]

# # ========== APP INTERFACE ==========
# st.title("📊 Real-Time Stock Sentiment Predictor")
# col1, col2 = st.columns([2, 1])

# with col1:
#     selected_company = st.selectbox("Select Company", list(tickers.values()))
#     symbol = [k for k, v in tickers.items() if v == selected_company][0]

# with col2:
#     run_button = st.button("🔍 Analyze News & Predict")

# if run_button:
#     headlines = fetch_headlines(selected_company)
#     numeric_dict, hist_data = get_numeric_features(symbol)

#     if not headlines or not numeric_dict:
#         st.error("⚠️ Could not fetch news or stock data.")
#     else:
#         try:
#             numeric_arr = [numeric_dict[col] for col in scaler.feature_names_in_]
#             numeric_df = pd.DataFrame([numeric_arr], columns=scaler.feature_names_in_)
#             numeric_scaled = scaler.transform(numeric_df)[0]
#         except KeyError as e:
#             st.error(f"Missing feature: {e}")
#             st.stop()

#         st.subheader(f"📰 Headlines for {selected_company}")
#         results = []
#         for entry in headlines:
#             headline = clean(entry.title)
#             pub_date = getattr(entry, "published", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

#             prediction = predict(
#                 headline, numeric_scaled, model, tokenizer, scaler, label_encoder, device
#             )

#             results.append({
#                 "Date": pub_date.split(",")[-1].strip()[:16],
#                 "Company": selected_company,
#                 "Headline": headline,
#                 "Prediction": prediction,
#                 "Price": numeric_dict["PriceAfter"]
#             })

#         df = pd.DataFrame(results)

#         # ===== Display Table =====
#         st.dataframe(df[["Date", "Headline", "Prediction"]], use_container_width=True)

#         # ===== Show Technicals =====
#         st.subheader("📈 Real-Time Technical Features")
#         st.json(numeric_dict)

#         # ===== Charts =====
#         # 1. Scatter Plot
#         st.subheader("🧠 Prediction vs Price")
#         fig1, ax1 = plt.subplots(figsize=(10, 4))
#         sns.scatterplot(data=df, x="Date", y="Price", hue="Prediction", palette="Set2", s=100, ax=ax1)
#         ax1.tick_params(axis='x', rotation=45)
#         st.pyplot(fig1)

#         # 2. Bar Chart - Sentiment
#         st.subheader("📊 Sentiment Distribution")
#         fig2, ax2 = plt.subplots()
#         sentiment_counts = df["Prediction"].value_counts()
#         sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax2)
#         ax2.set_ylabel("Count")
#         st.pyplot(fig2)

#         # 3. Line Chart - Price
#         st.subheader("📉 Last 15-Day Price Trend")
#         hist_data = hist_data.reset_index()
#         fig3, ax3 = plt.subplots(figsize=(10, 4))
#         sns.lineplot(data=hist_data, x="Date", y="Close", marker="o", ax=ax3)
#         ax3.tick_params(axis='x', rotation=45)
#         st.pyplot(fig3)

#         # 4. Pie Chart
#         st.subheader("🧩 Sentiment Share")
#         fig4, ax4 = plt.subplots()
#         ax4.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
#         st.pyplot(fig4)

#         # 5. Technical Indicators Line Chart
#         st.subheader("📌 Technical Indicators (MA5 & MA10)")
#         fig5, ax5 = plt.subplots(figsize=(10, 4))
#         sns.lineplot(data=hist_data, x="Date", y="Close", label="Close", ax=ax5)
#         sns.lineplot(data=hist_data, x="Date", y="MA5", label="MA5", ax=ax5)
#         sns.lineplot(data=hist_data, x="Date", y="MA10", label="MA10", ax=ax5)
#         ax5.tick_params(axis='x', rotation=45)
#         st.pyplot(fig5)

#         # 6. Volatility Bar Chart
#         st.subheader("🌪️ Daily Volatility")
#         fig6, ax6 = plt.subplots()
#         sns.barplot(data=hist_data, x="Date", y="Volatility", palette="Blues", ax=ax6)
#         ax6.tick_params(axis='x', rotation=45)
#         st.pyplot(fig6)

#         # 7. Price Before vs After
#         st.subheader("📊 Price Before vs After")
#         fig7, ax7 = plt.subplots()
#         sns.barplot(x=["Price Before", "Price After"], y=[numeric_dict["PriceBefore"], numeric_dict["PriceAfter"]], palette="Set2", ax=ax7)
#         st.pyplot(fig7)

#         # 8. Correlation Matrix
#         st.subheader("📘 Correlation Matrix")
#         numeric_features_df = pd.DataFrame([numeric_dict])
#         fig8, ax8 = plt.subplots()
#         sns.heatmap(numeric_features_df.corr(), annot=True, cmap="coolwarm", ax=ax8)
#         st.pyplot(fig8)

#         # 9. Candlestick Chart
#         st.subheader("🕯️ Candlestick Chart")
#         fig9 = go.Figure(data=[go.Candlestick(
#             x=hist_data['Date'],
#             open=hist_data['Open'],
#             high=hist_data['High'],
#             low=hist_data['Low'],
#             close=hist_data['Close']
#         )])
#         fig9.update_layout(xaxis_rangeslider_visible=False)
#         st.plotly_chart(fig9)

#         # ===== Export Option =====
#         st.download_button("💾 Download CSV", df.to_csv(index=False), file_name="predictions.csv")
#         st.success("✅ Analysis Complete")

# st.markdown("---")
# st.caption("Made with ❤️ using Streamlit, BERT & yFinance")
   





# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import feedparser
# import torch
# import joblib
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from datetime import datetime
# from transformers import BertTokenizer
# from wordcloud import WordCloud

# from model import HybridModel
# from predict import predict
# from dataset import tickers

# # ========== CONFIG ==========
# st.set_page_config(page_title="📈 Stock Movement Prediction", layout="wide")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== LOAD MODEL ==========
# model = HybridModel(num_numeric_features=6, num_classes=3)
# state_dict = torch.load("models/hybrid_sentiment_model.pth", map_location=device)
# new_state_dict = {k.replace("fc_numeric", "numeric_fc"): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# scaler = joblib.load("models/scaler.pkl")
# label_encoder = joblib.load("models/label_encoder.pkl")

# # ========== CLEAN HEADLINE ==========
# def clean(text):
#     return re.sub(r"[^\w\s]", "", text).strip()

# # ========== GET TECHNICAL INDICATORS ==========
# def get_numeric_features(symbol):
#     try:
#         data = yf.Ticker(symbol).history(period="15d")
#         if data.empty or len(data) < 11:
#             return None, None

#         data["MA5"] = data["Close"].rolling(window=5).mean()
#         data["MA10"] = data["Close"].rolling(window=10).mean()
#         data["Volatility"] = ((data["High"] - data["Low"]) / data["Open"]) * 100

#         today = data.iloc[-1]
#         yesterday = data.iloc[-2]
#         numeric = {
#             "PriceBefore": yesterday["Close"],
#             "PriceAfter": today["Close"],
#             "ChangePercent": ((today["Close"] - yesterday["Close"]) / yesterday["Close"]) * 100,
#             "MA5": today["MA5"],
#             "MA10": today["MA10"],
#             "Volatility": today["Volatility"]
#         }

#         return numeric, data
#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         return None, None

# # ========== FETCH HEADLINES ==========
# def fetch_headlines(company):
#     query = f"{company} stock"
#     url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
#     return feedparser.parse(url).entries[:5]

# # ========== EMOJI MAP ==========
# emoji_map = {
#     "UP": "📈",
#     "DOWN": "📉",
#     "STABLE": "🔄"
# }

# # ========== APP INTERFACE ==========
# st.title("📊 Real-Time Stock Sentiment Predictor")

# col1, col2 = st.columns([2, 1])

# with col1:
#     selected_company = st.selectbox("Select Company", list(tickers.values()))
#     symbol = [k for k, v in tickers.items() if v == selected_company][0]

# with col2:
#     run_button = st.button("🔍 Analyze News & Predict")

# if run_button:
#     headlines = fetch_headlines(selected_company)
#     numeric_dict, hist_data = get_numeric_features(symbol)

#     if not headlines or not numeric_dict:
#         st.error("⚠️ Could not fetch news or stock data.")
#     else:
#         try:
#             numeric_arr = [numeric_dict[col] for col in scaler.feature_names_in_]
#             numeric_df = pd.DataFrame([numeric_arr], columns=scaler.feature_names_in_)
#             numeric_scaled = scaler.transform(numeric_df)[0]
#         except KeyError as e:
#             st.error(f"Missing feature: {e}")
#             st.stop()

#         results = []
#         wordcloud_text = []

#         for entry in headlines:
#             headline = clean(entry.title)
#             pub_date = getattr(entry, "published", datetime.now().strftime('%a, %d %b %Y %H:%M:%S'))
#             pred_label, confidence = predict(headline, numeric_scaled, model, tokenizer, scaler, label_encoder, device)

#             results.append({
#                 "Date": pub_date.split(",")[-1].strip()[:16],
#                 "Company": selected_company,
#                 "Headline": headline,
#                 "Prediction": f"{emoji_map.get(pred_label)} {pred_label} ({confidence}%)",
#                 "RawLabel": pred_label,
#                 "Confidence": confidence,
#                 "Price": numeric_dict["PriceAfter"]
#             })
#             wordcloud_text.append(headline)

#         df = pd.DataFrame(results)

#         # ========== TABS ==========
#         tab1, tab2, tab3 = st.tabs(["📰 Predictions", "📊 Charts", "📈 Technicals"])

#         # ========== TAB 1 ==========
#         with tab1:
#             st.subheader(f"📰 Headlines for {selected_company}")
#             st.dataframe(df[["Date", "Headline", "Prediction"]], use_container_width=True)

#             with st.expander("🔬 Raw Prediction Probabilities"):
#                 st.dataframe(df[["Headline", "RawLabel", "Confidence"]])

#             st.subheader("☁️ Word Cloud")
#             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(wordcloud_text))
#             fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
#             ax_wc.imshow(wordcloud, interpolation='bilinear')
#             ax_wc.axis("off")
#             st.pyplot(fig_wc)

#         # ========== TAB 2 ==========
#         with tab2:
#             st.subheader("🧠 Prediction vs Price")
#             fig1, ax1 = plt.subplots(figsize=(10, 4))
#             sns.scatterplot(data=df, x="Date", y="Price", hue="RawLabel", palette="Set2", s=100, ax=ax1)
#             ax1.tick_params(axis='x', rotation=45)
#             st.pyplot(fig1)

#             st.subheader("📊 Sentiment Distribution")
#             fig2, ax2 = plt.subplots()
#             sentiment_counts = df["RawLabel"].value_counts()
#             sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax2)
#             ax2.set_ylabel("Count")
#             st.pyplot(fig2)

#             st.subheader("📉 Last 15-Day Price Trend")
#             hist_data = hist_data.reset_index()
#             fig3, ax3 = plt.subplots(figsize=(10, 4))
#             sns.lineplot(data=hist_data, x="Date", y="Close", marker="o", ax=ax3)
#             ax3.tick_params(axis='x', rotation=45)
#             st.pyplot(fig3)

#             st.subheader("🕯️ Candlestick Chart")
#             fig4 = go.Figure(data=[go.Candlestick(
#                 x=hist_data['Date'],
#                 open=hist_data['Open'],
#                 high=hist_data['High'],
#                 low=hist_data['Low'],
#                 close=hist_data['Close']
#             )])
#             fig4.update_layout(xaxis_rangeslider_visible=False)
#             st.plotly_chart(fig4)

#         # ========== TAB 3 ==========
#         with tab3:
#             st.subheader("📌 Technical Indicators (MA5 & MA10)")
#             fig5, ax5 = plt.subplots(figsize=(10, 4))
#             sns.lineplot(data=hist_data, x="Date", y="Close", label="Close", ax=ax5)
#             sns.lineplot(data=hist_data, x="Date", y="MA5", label="MA5", ax=ax5)
#             sns.lineplot(data=hist_data, x="Date", y="MA10", label="MA10", ax=ax5)
#             ax5.tick_params(axis='x', rotation=45)
#             st.pyplot(fig5)

#             st.subheader("🌪️ Daily Volatility")
#             fig6, ax6 = plt.subplots()
#             sns.barplot(data=hist_data, x="Date", y="Volatility", palette="Blues", ax=ax6)
#             ax6.tick_params(axis='x', rotation=45)
#             st.pyplot(fig6)

#             st.subheader("📊 Price Before vs After")
#             fig7, ax7 = plt.subplots()
#             sns.barplot(x=["Price Before", "Price After"],
#                         y=[numeric_dict["PriceBefore"], numeric_dict["PriceAfter"]],
#                         palette="Set2", ax=ax7)
#             st.pyplot(fig7)

#             st.subheader("📘 Correlation Matrix")
#             numeric_features_df = pd.DataFrame([numeric_dict])
#             fig8, ax8 = plt.subplots()
#             sns.heatmap(numeric_features_df.corr(), annot=True, cmap="coolwarm", ax=ax8)
#             st.pyplot(fig8)

#         st.download_button("💾 Download CSV", df.to_csv(index=False), file_name="predictions.csv")
#         st.success("✅ Analysis Complete")

# st.markdown("---")
# st.caption("Made with ❤️ using Streamlit, BERT & yFinance")







# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import feedparser
# import torch
# import joblib
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from datetime import datetime
# from transformers import BertTokenizer
# from wordcloud import WordCloud

# from model import HybridModel
# from predict import predict
# from dataset import tickers

# # ========== CONFIG ==========
# st.set_page_config(page_title="📈 Stock Movement Prediction", layout="wide")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== LOAD MODEL ==========
# model = HybridModel(num_numeric_features=6, num_classes=3)
# state_dict = torch.load("models/hybrid_sentiment_model.pth", map_location=device)
# new_state_dict = {k.replace("fc_numeric", "numeric_fc"): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# scaler = joblib.load("models/scaler.pkl")
# label_encoder = joblib.load("models/label_encoder.pkl")

# # ========== CLEAN HEADLINE ==========
# def clean(text):
#     return re.sub(r"[^\w\s]", "", text).strip()

# # ========== GET TECHNICAL INDICATORS ==========
# def get_numeric_features(symbol):
#     try:
#         data = yf.Ticker(symbol).history(period="15d")
#         if data.empty or len(data) < 11:
#             return None, None

#         data["MA5"] = data["Close"].rolling(window=5).mean()
#         data["MA10"] = data["Close"].rolling(window=10).mean()
#         data["Volatility"] = ((data["High"] - data["Low"]) / data["Open"]) * 100

#         today = data.iloc[-1]
#         yesterday = data.iloc[-2]
#         numeric = {
#             "PriceBefore": yesterday["Close"],
#             "PriceAfter": today["Close"],
#             "ChangePercent": ((today["Close"] - yesterday["Close"]) / yesterday["Close"]) * 100,
#             "MA5": today["MA5"],
#             "MA10": today["MA10"],
#             "Volatility": today["Volatility"]
#         }

#         return numeric, data
#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         return None, None

# # ========== FETCH HEADLINES ==========
# def fetch_headlines(company):
#     query = f"{company} stock"
#     url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
#     return feedparser.parse(url).entries[:5]

# # ========== EMOJI MAP ==========
# emoji_map = {
#     "UP": "📈",
#     "DOWN": "📉",
#     "STABLE": "🔄"
# }

# # ========== APP INTERFACE ==========
# st.title("📊 Real-Time Stock Sentiment Predictor")

# col1, col2 = st.columns([2, 1])

# with col1:
#     selected_company = st.selectbox("Select Company", list(tickers.values()))
#     symbol = [k for k, v in tickers.items() if v == selected_company][0]

# with col2:
#     run_button = st.button("🔍 Analyze News & Predict")

# if run_button:
#     headlines = fetch_headlines(selected_company)
#     numeric_dict, hist_data = get_numeric_features(symbol)

#     if not headlines or not numeric_dict:
#         st.error("⚠️ Could not fetch news or stock data.")
#     else:
#         try:
#             numeric_arr = [numeric_dict[col] for col in scaler.feature_names_in_]
#             numeric_df = pd.DataFrame([numeric_arr], columns=scaler.feature_names_in_)
#             numeric_scaled = scaler.transform(numeric_df)[0]
#         except KeyError as e:
#             st.error(f"Missing feature: {e}")
#             st.stop()

#         results = []
#         wordcloud_text = []

#         for entry in headlines:
#             headline = clean(entry.title)
#             pub_date = getattr(entry, "published", datetime.now().strftime('%a, %d %b %Y %H:%M:%S'))
#             pred_label, confidence = predict(headline, numeric_scaled, model, tokenizer, scaler, label_encoder, device)

#             results.append({
#                 "Date": pub_date.split(",")[-1].strip()[:16],
#                 "Company": selected_company,
#                 "Headline": headline,
#                 "Prediction": f"{emoji_map.get(pred_label)} {pred_label} ({confidence}%)",
#                 "RawLabel": pred_label,
#                 "Confidence": confidence,
#                 "Price": numeric_dict["PriceAfter"]
#             })
#             wordcloud_text.append(headline)

#         df = pd.DataFrame(results)

#         # ========== TABS ==========
#         tab1, tab2, tab3 = st.tabs(["📰 Predictions", "📊 Charts", "📈 Technicals"])

#         # ========== TAB 1 ==========
#         with tab1:
#             st.subheader(f"📰 Headlines for {selected_company}")
#             st.dataframe(df[["Date", "Headline", "Prediction"]], use_container_width=True)

#             with st.expander("🔬 Raw Prediction Probabilities"):
#                 st.dataframe(df[["Headline", "RawLabel", "Confidence"]])

#             st.subheader("☁️ Word Cloud")
#             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(wordcloud_text))
#             fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
#             ax_wc.imshow(wordcloud, interpolation='bilinear')
#             ax_wc.axis("off")
#             st.pyplot(fig_wc)

#         # ========== TAB 2 ==========
#         with tab2:
#             st.subheader("🧠 Prediction vs Price")
#             fig1, ax1 = plt.subplots(figsize=(12, 5))
#             sns.scatterplot(data=df, x="Date", y="Price", hue="RawLabel", palette="Set2", s=100, ax=ax1)
#             ax1.tick_params(axis='x', rotation=45)
#             st.pyplot(fig1)

#             st.subheader("📊 Sentiment Distribution")
#             fig2, ax2 = plt.subplots(figsize=(10, 4))
#             sentiment_counts = df["RawLabel"].value_counts()
#             sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax2)
#             ax2.set_ylabel("Count")
#             st.pyplot(fig2)

#             st.subheader("📉 Last 15-Day Price Trend")
#             hist_data = hist_data.reset_index()
#             fig3, ax3 = plt.subplots(figsize=(12, 5))
#             sns.lineplot(data=hist_data, x="Date", y="Close", marker="o", ax=ax3)
#             ax3.tick_params(axis='x', rotation=45)
#             st.pyplot(fig3)

#             st.subheader("🕯️ Candlestick Chart")
#             fig4 = go.Figure(data=[go.Candlestick(
#                 x=hist_data['Date'],
#                 open=hist_data['Open'],
#                 high=hist_data['High'],
#                 low=hist_data['Low'],
#                 close=hist_data['Close']
#             )])
#             fig4.update_layout(xaxis_rangeslider_visible=False)
#             st.plotly_chart(fig4)

#         # ========== TAB 3 ==========
#         with tab3:
#             st.subheader("📌 Technical Indicators (MA5 & MA10)")
#             fig5, ax5 = plt.subplots(figsize=(12, 5))
#             sns.lineplot(data=hist_data, x="Date", y="Close", label="Close", ax=ax5, color="#4c72b0", marker='o')
#             sns.lineplot(data=hist_data, x="Date", y="MA5", label="MA5", ax=ax5, color="#55a868", marker='o')
#             sns.lineplot(data=hist_data, x="Date", y="MA10", label="MA10", ax=ax5, color="#c44e52", marker='o')
#             ax5.set_title("Moving Averages & Close Price", fontsize=14, fontweight="bold")
#             ax5.tick_params(axis='x', rotation=45)
#             ax5.set_xlabel("Date")
#             ax5.set_ylabel("Price")
#             fig5.tight_layout()
#             st.pyplot(fig5)

#             st.subheader("🌪️ Daily Volatility")
#             fig6, ax6 = plt.subplots(figsize=(10, 4))
#             sns.barplot(data=hist_data, x="Date", y="Volatility", palette="coolwarm", ax=ax6, edgecolor="black")
#             ax6.set_title("Volatility per Day", fontsize=14, fontweight="bold")
#             ax6.tick_params(axis='x', rotation=45)
#             fig6.tight_layout()
#             st.pyplot(fig6)

#             st.subheader("📊 Price Before vs After")
#             fig7, ax7 = plt.subplots(figsize=(6, 4))
#             sns.barplot(x=["Price Before", "Price After"],
#                         y=[numeric_dict["PriceBefore"], numeric_dict["PriceAfter"]],
#                         palette="Set2", ax=ax7, edgecolor="black")
#             ax7.set_title("Price Movement", fontsize=14, fontweight="bold")
#             fig7.tight_layout()
#             st.pyplot(fig7)

#             # Ensure the numeric data is valid for correlation matrix
#             numeric_features_df = pd.DataFrame([numeric_dict])

#             if numeric_features_df.empty:
#                 st.error("⚠️ No valid numeric data to display correlation matrix.")
#             else:
#                 st.subheader("📘 Correlation Matrix")
#                 # Display correlation matrix if the data is valid
#                 fig8, ax8 = plt.subplots(figsize=(6, 5))
#                 sns.heatmap(numeric_features_df.corr(), annot=True, cmap="coolwarm", ax=ax8, cbar=True)
#                 ax8.set_title("Correlation of Technical Features", fontsize=14, fontweight="bold")
#                 fig8.tight_layout()
#                 st.pyplot(fig8)

#         st.download_button("💾 Download CSV", df.to_csv(index=False), file_name="predictions.csv")
#         st.success("✅ Analysis Complete")

# st.markdown("---")
# st.caption("Made with ❤️ using Streamlit, BERT & yFinance")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import feedparser
# import torch
# import joblib
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from datetime import datetime
# from transformers import BertTokenizer
# from wordcloud import WordCloud

# from model import HybridModel
# from predict import predict
# from dataset import tickers

# # ========== CONFIG ==========
# st.set_page_config(page_title="📈 Stock Movement Prediction", layout="wide")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== LOAD MODEL ==========
# model = HybridModel(num_numeric_features=6, num_classes=3)
# state_dict = torch.load("models/hybrid_sentiment_model.pth", map_location=device)
# new_state_dict = {k.replace("fc_numeric", "numeric_fc"): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# scaler = joblib.load("models/scaler.pkl")
# label_encoder = joblib.load("models/label_encoder.pkl")

# # ========== CLEAN HEADLINE ==========
# def clean(text):
#     return re.sub(r"[^\w\s]", "", text).strip()

# # ========== GET TECHNICAL INDICATORS ==========
# def get_numeric_features(symbol):
#     try:
#         data = yf.Ticker(symbol).history(period="15d")
#         if data.empty or len(data) < 11:
#             return None, None

#         data["MA5"] = data["Close"].rolling(window=5).mean()
#         data["MA10"] = data["Close"].rolling(window=10).mean()
#         data["Volatility"] = ((data["High"] - data["Low"]) / data["Open"]) * 100

#         today = data.iloc[-1]
#         yesterday = data.iloc[-2]
#         numeric = {
#             "PriceBefore": yesterday["Close"],
#             "PriceAfter": today["Close"],
#             "ChangePercent": ((today["Close"] - yesterday["Close"]) / yesterday["Close"]) * 100,
#             "MA5": today["MA5"],
#             "MA10": today["MA10"],
#             "Volatility": today["Volatility"]
#         }

#         return numeric, data
#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         return None, None

# # ========== FETCH HEADLINES ==========
# def fetch_headlines(company):
#     query = f"{company} stock"
#     url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
#     return feedparser.parse(url).entries[:5]

# # ========== EMOJI MAP ==========
# emoji_map = {
#     "UP": "📈",
#     "DOWN": "📉",
#     "STABLE": "🔄"
# }

# # Function to safely retrieve emoji from emoji_map
# def get_prediction_emoji(pred_label):
#     # If the label is not found, return a fallback emoji
#     return emoji_map.get(pred_label, "❓")  # Default emoji if label is not found

# # ========== APP INTERFACE ==========
# st.title("📊 Real-Time Stock Sentiment Predictor")

# col1, col2 = st.columns([2, 1])

# with col1:
#     selected_company = st.selectbox("Select Company", list(tickers.values()))
#     symbol = [k for k, v in tickers.items() if v == selected_company][0]

# with col2:
#     run_button = st.button("🔍 Analyze News & Predict")

# if run_button:
#     headlines = fetch_headlines(selected_company)
#     numeric_dict, hist_data = get_numeric_features(symbol)

#     if not headlines or not numeric_dict:
#         st.error("⚠️ Could not fetch news or stock data.")
#     else:
#         try:
#             numeric_arr = [numeric_dict[col] for col in scaler.feature_names_in_]
#             numeric_df = pd.DataFrame([numeric_arr], columns=scaler.feature_names_in_)
#             numeric_scaled = scaler.transform(numeric_df)[0]
#         except KeyError as e:
#             st.error(f"Missing feature: {e}")
#             st.stop()

#         results = []
#         wordcloud_text = []

#         for entry in headlines:
#             headline = clean(entry.title)
#             pub_date = getattr(entry, "published", datetime.now().strftime('%a, %d %b %Y %H:%M:%S'))
#             pred_label, confidence = predict(headline, numeric_scaled, model, tokenizer, scaler, label_encoder, device)

#             # Get the emoji for the prediction label
#             prediction_emoji = get_prediction_emoji(pred_label)

#             results.append({
#                 "Date": pub_date.split(",")[-1].strip()[:16],
#                 "Company": selected_company,
#                 "Headline": headline,
#                 "Prediction": f"{prediction_emoji} {pred_label} ({confidence}%)",
#                 "RawLabel": pred_label,
#                 "Confidence": confidence,
#                 "Price": numeric_dict["PriceAfter"]
#             })
#             wordcloud_text.append(headline)

#         df = pd.DataFrame(results)

#         # ========== TABS ==========
#         tab1, tab2, tab3 = st.tabs(["📰 Predictions", "📊 Charts", "📈 Technicals"])

#         # ========== TAB 1 ==========
#         with tab1:
#             st.subheader(f"📰 Headlines for {selected_company}")
#             st.dataframe(df[["Date", "Headline", "Prediction"]], use_container_width=True)

#             with st.expander("🔬 Raw Prediction Probabilities"):
#                 st.dataframe(df[["Headline", "RawLabel", "Confidence"]])

#             st.subheader("☁️ Word Cloud")
#             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(wordcloud_text))
#             fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
#             ax_wc.imshow(wordcloud, interpolation='bilinear')
#             ax_wc.axis("off")
#             st.pyplot(fig_wc)

#         # ========== TAB 2 ==========
#         with tab2:
#             st.subheader("🧠 Prediction vs Price")
#             fig1, ax1 = plt.subplots(figsize=(12, 5))
#             sns.scatterplot(data=df, x="Date", y="Price", hue="RawLabel", palette="Set2", s=100, ax=ax1)
#             ax1.tick_params(axis='x', rotation=45)
#             st.pyplot(fig1)

#             st.subheader("📊 Sentiment Distribution")
#             fig2, ax2 = plt.subplots(figsize=(10, 4))
#             sentiment_counts = df["RawLabel"].value_counts()
#             sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax2)
#             ax2.set_ylabel("Count")
#             st.pyplot(fig2)

#             st.subheader("📉 Last 15-Day Price Trend")
#             hist_data = hist_data.reset_index()
#             fig3, ax3 = plt.subplots(figsize=(12, 5))
#             sns.lineplot(data=hist_data, x="Date", y="Close", marker="o", ax=ax3)
#             ax3.tick_params(axis='x', rotation=45)
#             st.pyplot(fig3)

#             st.subheader("🕯️ Candlestick Chart")
#             fig4 = go.Figure(data=[go.Candlestick(
#                 x=hist_data['Date'],
#                 open=hist_data['Open'],
#                 high=hist_data['High'],
#                 low=hist_data['Low'],
#                 close=hist_data['Close']
#             )])
#             fig4.update_layout(xaxis_rangeslider_visible=False)
#             st.plotly_chart(fig4)

#         # ========== TAB 3 ==========
#         with tab3:
#             st.subheader("📌 Technical Indicators (MA5 & MA10)")
#             fig5, ax5 = plt.subplots(figsize=(12, 5))
#             sns.lineplot(data=hist_data, x="Date", y="Close", label="Close", ax=ax5, color="#4c72b0", marker='o')
#             sns.lineplot(data=hist_data, x="Date", y="MA5", label="MA5", ax=ax5, color="#55a868", marker='o')
#             sns.lineplot(data=hist_data, x="Date", y="MA10", label="MA10", ax=ax5, color="#c44e52", marker='o')
#             ax5.set_title("Moving Averages & Close Price", fontsize=14, fontweight="bold")
#             ax5.tick_params(axis='x', rotation=45)
#             ax5.set_xlabel("Date")
#             ax5.set_ylabel("Price")
#             fig5.tight_layout()
#             st.pyplot(fig5)

#             st.subheader("🌪️ Daily Volatility")
#             fig6, ax6 = plt.subplots(figsize=(10, 4))
#             sns.barplot(data=hist_data, x="Date", y="Volatility", palette="coolwarm", ax=ax6, edgecolor="black")
#             ax6.set_title("Volatility per Day", fontsize=14, fontweight="bold")
#             ax6.tick_params(axis='x', rotation=45)
#             fig6.tight_layout()
#             st.pyplot(fig6)

#             st.subheader("📊 Price Before vs After")
#             fig7, ax7 = plt.subplots(figsize=(6, 4))
#             sns.barplot(x=["Price Before", "Price After"],
#                         y=[numeric_dict["PriceBefore"], numeric_dict["PriceAfter"]],
#                         palette="Set2", ax=ax7, edgecolor="black")
#             ax7.set_title("Price Movement", fontsize=14, fontweight="bold")
#             fig7.tight_layout()
#             st.pyplot(fig7)

#             # Ensure the numeric data is valid for correlation matrix
#             numeric_features_df = pd.DataFrame([numeric_dict])

#             if numeric_features_df.empty:
#                 st.error("⚠️ No valid numeric data to display correlation matrix.")
#             else:
#                 st.subheader("📘 Correlation Matrix")
#                 # Display correlation matrix if the data is valid
#                 fig8, ax8 = plt.subplots(figsize=(6, 5))
#                 sns.heatmap(numeric_features_df.corr(), annot=True, cmap="coolwarm", ax=ax8, cbar=True)
#                 ax8.set_title("Correlation of Technical Features", fontsize=14, fontweight="bold")
#                 fig8.tight_layout()
#                 st.pyplot(fig8)

#         st.download_button("💾 Download CSV", df.to_csv(index=False), file_name="predictions.csv")
#         st.success("✅ Analysis Complete")

# st.markdown("---")
# st.caption("Made with ❤️ using Streamlit, BERT & yFinance")
 






import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import torch
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
from transformers import BertTokenizer
from wordcloud import WordCloud

from model import HybridModel
from predict import predict
from dataset import tickers

# ========== CONFIG ========== 
st.set_page_config(page_title="📈 Stock Movement Prediction", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD MODEL ========== 
model = HybridModel(num_numeric_features=6, num_classes=3)
state_dict = torch.load("models/hybrid_sentiment_model.pth", map_location=device)
new_state_dict = {k.replace("fc_numeric", "numeric_fc"): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ========== CLEAN HEADLINE ========== 
def clean(text):
    return re.sub(r"[^\w\s]", "", text).strip()

# ========== GET TECHNICAL INDICATORS ========== 
def get_numeric_features(symbol):
    try:
        data = yf.Ticker(symbol).history(period="15d")
        if data.empty or len(data) < 11:
            return None, None

        data["MA5"] = data["Close"].rolling(window=5).mean()
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["Volatility"] = ((data["High"] - data["Low"]) / data["Open"]) * 100

        today = data.iloc[-1]
        yesterday = data.iloc[-2]
        numeric = {
            "PriceBefore": yesterday["Close"],
            "PriceAfter": today["Close"],
            "ChangePercent": ((today["Close"] - yesterday["Close"]) / yesterday["Close"]) * 100,
            "MA5": today["MA5"],
            "MA10": today["MA10"],
            "Volatility": today["Volatility"]
        }

        return numeric, data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

# ========== FETCH HEADLINES ========== 
def fetch_headlines(company):
    query = f"{company} stock"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    return feedparser.parse(url).entries[:5]

# ========== APP INTERFACE ========== 
st.title("📊 Real-Time Stock Sentiment Predictor")

col1, col2 = st.columns([2, 1])

with col1:
    selected_company = st.selectbox("Select Company", list(tickers.values()))
    symbol = [k for k, v in tickers.items() if v == selected_company][0]

with col2:
    run_button = st.button("🔍 Analyze News & Predict")

if run_button:
    headlines = fetch_headlines(selected_company)
    numeric_dict, hist_data = get_numeric_features(symbol)

    if not headlines or not numeric_dict:
        st.error("⚠️ Could not fetch news or stock data.")
    else:
        try:
            numeric_arr = [numeric_dict[col] for col in scaler.feature_names_in_]
            numeric_df = pd.DataFrame([numeric_arr], columns=scaler.feature_names_in_)
            numeric_scaled = scaler.transform(numeric_df)[0]
        except KeyError as e:
            st.error(f"Missing feature: {e}")
            st.stop()

        results = []
        wordcloud_text = []

        for entry in headlines:
            headline = clean(entry.title)
            pub_date = getattr(entry, "published", datetime.now().strftime('%a, %d %b %Y %H:%M:%S'))
            pred_label, confidence = predict(headline, numeric_scaled, model, tokenizer, scaler, label_encoder, device)

            results.append({
                "Date": pub_date.split(",")[-1].strip()[:16],
                "Company": selected_company,
                "Headline": headline,
                "Prediction": f"{pred_label} ({confidence}%)",  # No emoji
                "RawLabel": pred_label,
                "Confidence": confidence,
                "Price": numeric_dict["PriceAfter"]
            })
            wordcloud_text.append(headline)

        df = pd.DataFrame(results)

        # ========== TABS ========== 
        tab1, tab2, tab3 = st.tabs(["📰 Predictions", "📊 Charts", "📈 Technicals"])

        # ========== TAB 1 ========== 
        with tab1:
            st.subheader(f"📰 Headlines for {selected_company}")
            st.dataframe(df[["Date", "Headline", "Prediction"]], use_container_width=True)

            with st.expander("🔬 Raw Prediction Probabilities"):
                st.dataframe(df[["Headline", "RawLabel", "Confidence"]])

            st.subheader("☁️ Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(wordcloud_text))
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        # ========== TAB 2 ========== 
        with tab2:
            st.subheader("🧠 Prediction vs Price")
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            sns.scatterplot(data=df, x="Date", y="Price", hue="RawLabel", palette="Set2", s=100, ax=ax1)
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)

            st.subheader("📊 Sentiment Distribution")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sentiment_counts = df["RawLabel"].value_counts()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax2)
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            st.subheader("📉 Last 15-Day Price Trend")
            hist_data = hist_data.reset_index()
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=hist_data, x="Date", y="Close", marker="o", ax=ax3)
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)

            st.subheader("🕯️ Candlestick Chart")
            fig4 = go.Figure(data=[go.Candlestick(
                x=hist_data['Date'],
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close']
            )])
            fig4.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig4)

        # ========== TAB 3 ========== 
        with tab3:
            st.subheader("📌 Technical Indicators (MA5 & MA10)")
            fig5, ax5 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=hist_data, x="Date", y="Close", label="Close", ax=ax5, color="#4c72b0", marker='o')
            sns.lineplot(data=hist_data, x="Date", y="MA5", label="MA5", ax=ax5, color="#55a868", marker='o')
            sns.lineplot(data=hist_data, x="Date", y="MA10", label="MA10", ax=ax5, color="#c44e52", marker='o')
            ax5.set_title("Moving Averages & Close Price", fontsize=14, fontweight="bold")
            ax5.tick_params(axis='x', rotation=45)
            ax5.set_xlabel("Date")
            ax5.set_ylabel("Price")
            fig5.tight_layout()
            st.pyplot(fig5)

            st.subheader("🌪️ Daily Volatility")
            fig6, ax6 = plt.subplots(figsize=(10, 4))
            sns.barplot(data=hist_data, x="Date", y="Volatility", palette="coolwarm", ax=ax6, edgecolor="black")
            ax6.set_title("Volatility per Day", fontsize=14, fontweight="bold")
            ax6.tick_params(axis='x', rotation=45)
            fig6.tight_layout()
            st.pyplot(fig6)

            st.subheader("📊 Price Before vs After")
            fig7, ax7 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=["Price Before", "Price After"],
                        y=[numeric_dict["PriceBefore"], numeric_dict["PriceAfter"]],
                        palette="Set2", ax=ax7, edgecolor="black")
            ax7.set_title("Price Movement", fontsize=14, fontweight="bold")
            fig7.tight_layout()
            st.pyplot(fig7)

            # Ensure the numeric data is valid for correlation matrix
            numeric_features_df = pd.DataFrame([numeric_dict])

            if numeric_features_df.empty:
                st.error("⚠️ No valid numeric data to display correlation matrix.")
            else:
                st.subheader("📘 Correlation Matrix")
                # Display correlation matrix if the data is valid
                fig8, ax8 = plt.subplots(figsize=(6, 5))
                sns.heatmap(numeric_features_df.corr(), annot=True, cmap="coolwarm", ax=ax8, cbar=True)
                ax8.set_title("Correlation of Technical Features", fontsize=14, fontweight="bold")
                fig8.tight_layout()
                st.pyplot(fig8)

        st.download_button("💾 Download CSV", df.to_csv(index=False), file_name="predictions.csv")
        st.success("✅ Analysis Complete")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, BERT & yFinance")
