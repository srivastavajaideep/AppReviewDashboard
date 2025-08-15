# import urllib3

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import streamlit as st
import nltk
nltk.download('punkt')        # standard sentence and word tokenizers
nltk.download('punkt_tab')    # ensures related resources are also downloaded
nltk.download('vader_lexicon')
nltk.download('stopwords')


import plotly.graph_objects as go

import asyncio

import pandas as pd

import base64

import io

import asyncio

import pandas as pd

# import aiohttp

import qrcode

from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="WU Customer Sentiment Analyzer!!!", page_icon=":sparkles:",layout="wide")

# st.title(" :sparkles: WU App Review DashBoard")

st.markdown('<style>div.block-container{padding-top:1rem;text-align: center}</style>',unsafe_allow_html=True)

import plotly.express as px

import pandas as pd

import numpy as np

from datetime import date, timedelta

from pprint import pprint

import json

#import asyncio

import time

from fpdf import FPDF

from sklearn.cluster import KMeans

import warnings

# from bertopic import BERTopic

import requests

import datetime

from languages import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentence_transformers import SentenceTransformer


from deep_translator import GoogleTranslator

import plotly.express as px

from google_play_scraper import reviews, Sort

import os

from google_play_scraper import Sort, reviews_all

# from app_store_scraper import AppStore

import seaborn as sns

import pycountry

from wordcloud import WordCloud, STOPWORDS

from langdetect import detect

import matplotlib.pyplot as plt

from nltk.util import ngrams

from PIL import Image

from collections import Counter

from googletrans import Translator

from languages import *

warnings.filterwarnings('ignore')


import ssl

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

import certifi

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from streamlit_plotly_events import plotly_events

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import snscrape.modules.twitter as sntwitter

from langdetect import detect

import pandas as pd

import warnings

# import pyLDAvis

# import pyLDAvis.sklearn

import streamlit.components.v1 as components

from textblob import TextBlob

import matplotlib.pyplot as plt

from transformers import pipeline

from nltk.corpus import stopwords

from streamlit_autorefresh import st_autorefresh

 



 

app_url = "streamlit.app"

 

# Generate the QR code

qr = qrcode.QRCode(

    version=1,

    box_size=10,

    border=5

)

qr.add_data(app_url)

qr.make(fit=True)

 

# Create an image

img = qr.make_image(fill="black", back_color="white")

 

# Save to file

img.save("app_qr_code.png")

 

count = st_autorefresh(interval=3600000, key="fizzbuzzcounter")

sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

# st.cache_data.clear()

print(st.__version__)

translator = Translator()

 

# wu_mask = np.array(Image.open('wul.png'))

dir = os.path.dirname(__file__)

filename = os.path.join(dir, 'WUNEW.png')

image = Image.open(filename)

left_co, cent_co,last_co = st.columns(3)

with cent_co:

    st.image(image, caption='',width=150)

 

BEARER_TOKEN = ""

# Setup VADER Sentiment Analyzer

sid = SentimentIntensityAnalyzer()

 

#persist=True

@st.cache_data(ttl=86400,show_spinner=False)

def load_android_data(app_id, country, app_name):

    reviews = reviews_all(

        app_id,

        sleep_milliseconds=0,

        lang='en',

        country=country,

        sort=Sort.NEWEST,

    )

 

    df = pd.DataFrame(np.array(reviews), columns=['review'])

    df = df.join(pd.DataFrame(df.pop('review').tolist()))

    columns_to_drop = ['reviewId', 'thumbsUpCount', 'reviewCreatedVersion', 'repliedAt', 'userImage']

    df = df.drop(columns=columns_to_drop)

    df['AppName'] = app_name

    df['Country'] = country.lower()

    #translator=Translator(service_urls=['translate.googleapis.com'])

    try:

    # df['translated_text'] = df['review'].apply(lambda x: translator.translate(x, dest='English').text)

     df['translated_text'] = df['review'].apply(lambda x: translator.translate(x, dest='Exnglish').text)

   

    except:

     print("An exception occurred")

    df.rename(columns={

        'content': 'review',

        'userName': 'UserName',

        'score': 'rating',

        'at': 'TimeStamp',

        'replyContent': 'WU_Response'

    }, inplace=True)

    return df

 

app_details = [

    ('com.westernunion.android.mtapp', 'us', 'Android'),

    ('com.westernunion.moneytransferr3app.eu','fr','Android'),  

    ('com.westernunion.moneytransferr3app.au', 'au', 'Android'),

    # ('com.westernunion.moneytransferr3app.eu','gb','Android'),

    ('com.westernunion.moneytransferr3app.eu','de','Android'),

    ('com.westernunion.moneytransferr3app.ca', 'ca', 'Android'),  

    ('com.westernunion.moneytransferr3app.eu','it','Android'),

    ('com.westernunion.moneytransferr3app.eu3','se','Android'),

    #   ('com.westernunion.moneytransferr3app.eu3','se','Android'),

    ('com.westernunion.moneytransferr3app.nz', 'nz', 'Android'),

    ('com.westernunion.android.mtapp', 'co', 'Android'),

    ('com.westernunion.moneytransferr3app.nl','nl','Android'),

    ('com.westernunion.moneytransferr3app.acs3','br','Android'),

    ('com.westernunion.moneytransferr3app.eu2','be','Android'),

    ('com.westernunion.moneytransferr3app.eu3','no','Android'),

    ('com.westernunion.moneytransferr3app.eu','at','Android'),    

    ('com.westernunion.moneytransferr3app.eu2','ch','Android'),

    ('com.westernunion.moneytransferr3app.sg','sg','Android'),

    ('com.westernunion.moneytransferr3app.eu3','dk','Android'),

    ('com.westernunion.moneytransferr3app.eu','ie','Android'),

    ('com.westernunion.moneytransferr3app.pt','pt','Android'),

    ('com.westernunion.moneytransferr3app.eu4','po','Android'),

    ('com.westernunion.moneytransferr3app.eu3','po','Android'),

    ('com.westernunion.moneytransferr3app.apac','my','Android'),

    ('com.westernunion.moneytransferr3app.hk','hk','Android'),

    ('com.westernunion.moneytransferr3app.ae', 'ae', 'Android'),

    ('com.westernunion.moneytransferr3app.bh', 'bh', 'Android'),    

    ('com.westernunion.moneytransferr3app.kw', 'kw', 'Android'),

    ('com.westernunion.moneytransferr3app.qa', 'qa', 'Android'),

    ('com.westernunion.moneytransferr3app.sa', 'sa', 'Android'),

    ('com.westernunion.moneytransferr3app.in', 'in', 'Android'),

    ('com.westernunion.moneytransferr3app.th', 'th', 'Android')  

 

]

 

frames = []

for app_id, country, app_name in app_details:

    try:

       

        with st.spinner("⏳ Loading Android Reviews..."):

 

            frames.append(load_android_data(app_id, country, app_name))

    except KeyError:

        frames.append(pd.DataFrame())

 

finaldfandroid = pd.concat(frames)

 

#########Working code

# APP_ID = "424716908"

# @st.cache_data(ttl=3600)

# def fetch_reviews_rss(app_id, country_code, pages=10):

#     reviews = []

#     for p in range(1, pages + 1):

#         url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/page={p}/id={app_id}/sortBy=mostRecent/json"

#         #st.write(url)

#         resp = requests.get(url)

#         if resp.status_code != 200:

#             continue

#         data = resp.json()

#         entries = data.get("feed", {}).get("entry", [])

#         for e in entries[1:]:

#             reviews.append({

               

#                 "rating": int(e["im:rating"]["label"]),

#                 # "title": e["title"]["label"],

#                 "date": e["updated"]["label"],

#                 "review": e["content"]["label"],

#                 "translated_text":"",

#                 "WU_Response":"",

#                 "UserName": e["author"]["name"]["label"],

#                 "AppName":'iOS',

#                 "Country": country_code,

#                 "appVersion": (e["im:version"]["label"])

               

#             })

#     return pd.DataFrame(reviews)

   

 

# finaldfios = fetch_reviews_rss(APP_ID, "us", pages=1)

# # print(finaldfios.dtypes)

# finaldfios['date'] = pd.to_datetime(finaldfios['date'], errors='coerce')

# finaldfios['TimeStamp'] = finaldfios['date'].dt.date

# finaldfios["TimeStamp"] = pd.to_datetime(finaldfios["TimeStamp"])

################ Working code ends

 

app_country_list = [

    ("424716908", "us"), # Western Union US

    ("1045347175","fr"), #France

    ("1122288720", "au"), #Australia

    # ("1045347175","gb"),   #UK

    ("1045347175", "de"), #  Germany

    ("1110191056","ca"),  #canada

    ("1045347175","it"),   #Italy  

    ("1152860407","se"),   #Sweden  

    ("1268771757","es"), #Spain

    ("1226778839","nz"), #New Zealand

    ("1199782520","nl"),   #Netherland  

    ("1148514737","br"),  #Brazil

    ("1110240507","be"), #Belgium

    ("1152860407","no"),   #Norway  

    ("1045347175","at"),   #Austria

    ("1110240507","ch"),   #Switzerland

    ("1451754888","ch"),   #Singapore

    ("1152860407","dk"),   #Denmark  

    ("1045347175","ie"),   #ireland  

    ("1229307854","pt"),   #Portugal  

    ("1168530510","pl"),   #Poland

    ("1152860407","fi"),   #finland

    ("1165109779","hk"),   # hongkong    

 

 

    ("1171330611","ae"), # UAE    

    ("1329774999","co"), # columbia    

    ("1314010624","bh"), # Bahrain

    ("1304223498","cl"), #chile

    ("1459023219","jo"), #Jordan  

    ("1173794098","kw"), #kuwait

    ("1483742169","mv"), #Maldives      

    ("1459024696","sa"), #saudi arabia    

    ("1459226729","th"), #thailand

    ("1173792939","qa"), #qatar      

]

 

@st.cache_data(ttl=86400,show_spinner=False)

def fetch_ios_reviews(app_id, country_code, pages=5):

    reviews = []

    for p in range(1, pages + 1):

        url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/page={p}/id={app_id}/sortBy=mostRecent/json"

        try:

            resp = requests.get(url)

            if resp.status_code != 200:

                continue

            entries = resp.json().get("feed", {}).get("entry", [])[1:] # skip the app metadata

            for entry in entries:

                reviews.append({

                    "rating": entry.get("im:rating", {}).get("label"),

                    # "title": entry.get("title", {}).get("label"),

                    "date": entry.get("updated", {}).get("label"),

                    "review": entry.get("content", {}).get("label"),

                    "WU_Response": entry.get("im:developerResponse", {}).get("label", None),

                    "UserName": entry.get("author", {}).get("name", {}).get("label"),

                    "Platform": "iOS",

                    "AppName":'iOS',

                    "Country": country_code,

                    "AppID": app_id

                })

        except Exception as e:

            print(f"Error for {app_id}-{country_code}: {e}")

            continue

    return pd.DataFrame(reviews)

 

# Aggregate reviews from all countries & apps

df_list = []

 

for app_id, country_code in app_country_list:

   with st.spinner("⏳ Loading iOS Reviews..."):

    df = fetch_ios_reviews(app_id, country_code, pages=5)

    if not df.empty:

        # Convert date

        df["date"] = pd.to_datetime(df["date"], errors='coerce')

        df["TimeStamp"] = df["date"].dt.strftime('%Y-%m-%d')

        df_list.append(df)

 

       

 

# Final merged DataFrame

if df_list:

    finaldfios = pd.concat(df_list, ignore_index=True)

    # st.write(finaldfios)

else:

    st.write("No iOS reviews found.")

 

frames = [finaldfandroid,finaldfios]

 

finaldf = pd.concat(frames)

# st.write(finaldf)

 

try:

 finaldf['WU_Response'] = finaldf.apply(lambda x: json.loads(x['WU_Response'])['body'], axis = 1)

 print(finaldf['WU_Response'])

except:

 print('Some exception occured')  

 

# st.write(finaldf.loc[finaldf['WU_Response'].notnull()] )

#st.write(finaldf.head())

#st.write("Columns in finaldf:", finaldf.columns)

finaldf.columns = finaldf.columns.str.strip("'")

# ,'translated_text'

finaldf.sort_values(['TimeStamp', 'review','rating'])

finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]

col1, col2 = st.columns((2))

finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"])

 

# finaldf['Date'] = finaldf['TimeStamp'].dt.date

 

# Getting the min and max date

# startDate = pd.to_datetime(finaldf["TimeStamp"]).min()

# endDate = pd.to_datetime(finaldf["TimeStamp"]).max()

 

# with col1:

#     date1 = pd.to_datetime(st.date_input("Start Date", startDate))

 

# with col2:

#     date2 = pd.to_datetime(st.date_input("End Date", endDate))

 

# default_start = datetime.date(2025, 8, 1)

today=datetime.date.today()

default_start = today.replace(day=1)

default_end = datetime.date.today()

 

with col1:

    date1 = st.date_input("Start Date", value=default_start)

 

with col2:

    date2 = st.date_input("End Date", value=default_end)




def codechange(code):

    try:

        return pycountry.countries.get(alpha_2=code.upper()).alpha_3

    except:

        return None

 

date1 = pd.to_datetime(date1)

date2 = pd.to_datetime(date2)

 

# # Now the comparison will work:

# df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)].copy()

 

try:

 df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)].copy()

except KeyError:

 df = pd.DataFrame()

 

# st.sidebar.header("Choose your filter: ")

 

country = st.sidebar.multiselect("Select the Country", df["Country"].unique(), default=["us","in","fr","au","no"])

if not country:

    df1 = df.copy()

else:

    df1 = df[df["Country"].isin(country)]

 

region = st.sidebar.multiselect("Select the App Type", df["AppName"].unique())

if not region:

    df2 = df1.copy()

else:

    df2 = df1[df1["AppName"].isin(region)]

 

if not country and not region :

    filtered_df = df

else:

    filtered_df=df2




if 'rating' in filtered_df.columns:

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce') # convert rating to numeric (int/float)

    rating = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

    if rating:

        filtered_df = filtered_df[(filtered_df['rating'] >= rating[0]) & (filtered_df['rating'] <= rating[1])]

 

# Function to apply row-wise styling

def highlight_rating(row):

    try:

        rating = int(row['rating']) # Convert to int

    except:

        return [''] * len(row) # No styling if conversion fails

 

    if rating >= 4:

        return ['background-color: lightgreen'] * len(row)

    elif rating == 3:

        return ['background-color: yellow'] * len(row)

    else:

        return ['background-color: salmon'] * len(row)

 

df=df.reset_index(drop=True)

# styled_df = df.style.apply(highlight_rating, axis=1)

 

# st.write(styled_df)




if filtered_df.empty:

 st.warning("No records found within the specified date range")

else:

 #'Sentiment',

 filtered_df = filtered_df.reindex(['TimeStamp', 'review','rating','UserName','AppName','Country','appVersion'], axis=1)

#  ,'translated_text'

#  print(df)

 #

 filtered_df = filtered_df.reset_index(drop=True)

 filtered_df.index += 1  # Start index from 1

 filtered_df.index.name = "S.No."  # Give index column a name

 

 # filtered_df_display=filtered_df_display.head()

 filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

 filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

 filtered_df['TimeStamp']=pd.to_datetime(filtered_df["TimeStamp"]).dt.date

 

search_query = st.text_input("Search Reviews :")

 

# Filter the DataFrame in place based on the search query on the 'review' column

if search_query:

    filtered_df = filtered_df[filtered_df['review'].str.contains(search_query, case=False, na=False)]

 

# Now display the filtered_df in the original data grid (only one table shown)

st.dataframe(filtered_df, height=300, use_container_width=True)

 

# Download button just below the data grid

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")




#  st.dataframe(filtered_df,height=300,use_container_width=True)

# #  st.write(filtered_df)

#  csv = filtered_df.to_csv(index = False).encode('utf-8')

#  st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")  

 

 

 

source_text = st.text_area("Enter text to translate:", height=100)

default_language = 'English'  

default_index = languages.index(default_language) if default_language in languages else 0

 

# Set default selected language using index

target_language = st.selectbox("Select target language:", languages, index=default_index)

 

translate = st.button('Translate')

if translate:

    translator = Translator()

    out = translator.translate(source_text, dest=target_language)

    st.write(out.text)





# query = st.text_input("Search Query", value="Western Union lang:en")

 

# if st.button("Fetch Tweets"):

 

#         st.info("Fetching tweets...")

 

#         search_url = "https://api.twitter.com/2/tweets/search/recent"

#         headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

 

#         query_params = {

#             "query": query,

#             "max_results": 1,

#             "tweet.fields": "created_at,author_id,text",

#             "start_time": f"{date1}T00:00:00Z",

#             "end_time": f"{date2}T23:59:59Z",

#         }

 

#         response = requests.get(search_url, headers=headers, params=query_params)

 

#         if response.status_code == 200:

#             tweets = response.json().get("data", [])

 

#             if not tweets:

#                 st.warning("No tweets found for this period.")

#             else:

#                 filtered_df = pd.DataFrame(tweets)

#                 filtered_df["created_at"] = pd.to_datetime(filtered_df["created_at"])

 

#                 # --- Sentiment Analysis ---

#                 def analyze_sentiment(text):

#                     score = sid.polarity_scores(text)["compound"]

#                     if score >= 0.05:

#                         return "Positive"

#                     elif score <= -0.05:

#                         return "Negative"

#                     else:

#                         return "Neutral"

 

#                 df["sentiment"] = df["text"].apply(analyze_sentiment)

 

#                 # --- Show Raw Data ---

#                 st.success(f"Fetched {len(df)} tweets")

#                 st.dataframe(df[["created_at", "text", "sentiment"]])

 

#                 # --- Daily Tweet Count Plot ---

#                 daily_counts = df["created_at"].dt.date.value_counts().sort_index()

#                 fig, ax = plt.subplots()

#                 daily_counts.plot(kind="bar", ax=ax)

#                 ax.set_title("Tweet Volume by Day")

#                 ax.set_xlabel("Date")

#                 ax.set_ylabel("Number of Tweets")

#                 st.pyplot(fig)

 

#                 # --- Sentiment Breakdown ---

#                 st.subheader("📊 Sentiment Distribution")

#                 sentiment_counts = df["sentiment"].value_counts()

#                 st.bar_chart(sentiment_counts)

 

#         else:

#             st.error(f"Twitter API error: {response.status_code}")

#             st.json(response.json())




# if 'country' in df.columns:

#     country = st.sidebar.multiselect("Country", df['country'].unique())

#     if country:

#         df = df[df['country'].isin(country)]

 

# if 'apptype' in df.columns:

#     apptype = st.sidebar.multiselect("App Type", df['apptype'].unique())

#     if apptype:

#         df = df[df['apptype'].isin(apptype)]

 

   

 

# st.subheader("Review Data Overview")

# st.dataframe(filtered_df[['UserName', 'AppName', 'Country', 'review', 'rating']], use_container_width=True, height=250)

   

 

if not st.sidebar.checkbox("Keyword Analysis", True):

    # # Keyword and N-gram Analysis

    # st.subheader(" Keyword and N-gram Analysis")

    # all_text = " ".join(filtered_df['review'].astype(str).tolist()).lower()

    # tokens = [word for word in nltk.word_tokenize(all_text) if word.isalpha() and word not in stop_words]

    # unigram_freq = Counter(tokens)

    # bigram_freq = Counter(ngrams(tokens, 2))

    # trigram_freq = Counter(ngrams(tokens, 3))

 

    # top_unigrams = pd.DataFrame(unigram_freq.most_common(10), columns=['Unigram', 'Count'])

    # top_bigrams = pd.DataFrame(bigram_freq.most_common(10), columns=['Bigram', 'Count'])

    # top_trigrams = pd.DataFrame(trigram_freq.most_common(10), columns=['Trigram', 'Count'])

 

    # col1, col2, col3 = st.columns(3)

    # with col1:

    #     st.write("### Top Unigrams")

    #     st.dataframe(top_unigrams)

    # with col2:

    #     st.write("### Top Bigrams")

    #     st.dataframe(top_bigrams.astype(str))

    # with col3:

    #     st.write("### Top Trigrams")

    #     st.dataframe(top_trigrams.astype(str))

 

    st.subheader("Keyword and N-gram Analysis")

    filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

     

    selected_sentiment = st.selectbox("Filter by Sentiment for N-gram Analysis", filtered_df['sentiment_label'].unique().tolist() + ['All'])

    selected_apptype = st.selectbox("Filter by App Type for N-gram Analysis", filtered_df['AppName'].dropna().unique().tolist() + ['All'])

    ngram_count = st.slider("Number of Top N-grams to Display", 5, 30, 10)

 

    filtered_df = filtered_df.copy()

    if selected_sentiment != 'All':

        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]

    if selected_apptype != 'All':

        filtered_df = filtered_df[filtered_df['AppName'] == selected_apptype]

 

    all_text = " ".join(filtered_df['review'].astype(str).tolist()).lower()

    tokens = [word for word in nltk.word_tokenize(all_text) if word.isalpha() and word not in stop_words]

    unigram_freq = Counter(tokens)

    bigram_freq = Counter(ngrams(tokens, 2))

    trigram_freq = Counter(ngrams(tokens, 3))

 

    top_unigrams = pd.DataFrame(unigram_freq.most_common(ngram_count), columns=['Unigram', 'Count'])

    top_bigrams = pd.DataFrame(bigram_freq.most_common(ngram_count), columns=['Bigram', 'Count'])

    top_trigrams = pd.DataFrame(trigram_freq.most_common(ngram_count), columns=['Trigram', 'Count'])

 

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("### Top Unigrams")

        st.dataframe(top_unigrams)

        fig_uni = px.bar(top_unigrams, x='Unigram', y='Count', title='Top Unigrams')

        st.plotly_chart(fig_uni, use_container_width=True)

    with col2:

        st.write("### Top Bigrams")

        st.dataframe(top_bigrams.astype(str))

        top_bigrams['Bigram'] = top_bigrams['Bigram'].apply(lambda x: ' '.join(x))

        fig_bi = px.bar(top_bigrams, x='Bigram', y='Count', title='Top Bigrams')

        st.plotly_chart(fig_bi, use_container_width=True)

    with col3:

        st.write("### Top Trigrams")

        st.dataframe(top_trigrams.astype(str))

        top_trigrams['Trigram'] = top_trigrams['Trigram'].apply(lambda x: ' '.join(x))

        fig_tri = px.bar(top_trigrams, x='Trigram', y='Count', title='Top Trigrams')

        st.plotly_chart(fig_tri, use_container_width=True)




# if not st.sidebar.checkbox("Topic Modeling", True):

# # Topic Modeling (LDA)

#     st.subheader("Topic Modeling")

#     # vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)

#     # doc_term_matrix = vectorizer.fit_transform(filtered_df['review'].astype(str))

#     # lda = LatentDirichletAllocation(n_components=5, random_state=42)

#     # lda.fit(doc_term_matrix)

#     # words = vectorizer.get_feature_names_out()

#     # for i, topic in enumerate(lda.components_):

#     #     topic_words = [words[i] for i in topic.argsort()[-10:]]

#     #     st.write(f"**Topic {i+1}:** {', '.join(topic_words)}")

   

#     # Separate positive and negative reviews

#     positive_reviews = filtered_df[filtered_df['sentiment_label'] == 'Positive']['review']

#     negative_reviews = filtered_df[filtered_df['sentiment_label'] == 'Negative']['review']

 

#     # Function to extract topics and keywords

#     def extract_topics(reviews, n_topics=5, n_keywords=5):

#         vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)

#         doc_term_matrix = vectorizer.fit_transform(reviews)

#         lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

#         lda.fit(doc_term_matrix)

#         topics = []

#         for idx, topic in enumerate(lda.components_):

#             keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_keywords:]]

#             topics.append((f"Topic {idx+1}", keywords))

#         return topics, lda.transform(doc_term_matrix)

 

#         # Extract topics and keywords

#     positive_topics, positive_topic_distributions = extract_topics(positive_reviews)

#     negative_topics, negative_topic_distributions = extract_topics(negative_reviews)

 

#     # Function to get representative sentences

#     def get_representative_sentences(reviews, topic_distributions, n_sentences=2):

#         sentences = []

#         for topic_idx in range(topic_distributions.shape[1]):

#             topic_scores = topic_distributions[:, topic_idx]

#             top_indices = topic_scores.argsort()[-n_sentences:]

#             topic_sentences = reviews.iloc[top_indices].tolist()

#             sentences.append(topic_sentences)

#         return sentences

 

#     # Get representative sentences

#     positive_sentences = get_representative_sentences(positive_reviews, positive_topic_distributions)

#     negative_sentences = get_representative_sentences(negative_reviews, negative_topic_distributions)

 

#     # Format the output

#     def format_topic_summary(topics, sentences):

#         summaries = []

#         for i, (topic_name, keywords) in enumerate(topics):

#             summary = f"{topic_name}: Keywords - {', '.join(keywords)}\nRepresentative Sentences:\n"

#             for sent in sentences[i]:

#                 summary += f"- {sent}\n"

#             summaries.append(summary)

#         return summaries

 

#     # Generate summaries

#     positive_summaries = format_topic_summary(positive_topics, positive_sentences)

#     negative_summaries = format_topic_summary(negative_topics, negative_sentences)

 

#     # Save summaries to text files

#     with open("top_5_best_aspects.txt", "w", encoding="utf-8") as f:

#         for summary in positive_summaries:

#             f.write(summary + "\n\n")

 

#     with open("top_5_issues.txt", "w", encoding="utf-8") as f:

#         for summary in negative_summaries:

#             f.write(summary + "\n\n")

 

if not st.sidebar.checkbox("Topic Modeling", True):

    # Topic Modeling (LDA)

    st.subheader("Topic Modeling")

 

    # Separate positive and negative reviews

    # positive_reviews = filtered_df[filtered_df['sentiment_label'] == 'Positive']['review']

    # negative_reviews = filtered_df[filtered_df['sentiment_label'] == 'Negative']['review']

   

    positive_reviews = filtered_df[filtered_df['rating'] >= 4]['review']

    negative_reviews = filtered_df[filtered_df['rating'] <= 2]['review']

    neutral_reviews = filtered_df[filtered_df['rating'] == 3]['review']

 

   

# Function to extract topics and keywords

    def extract_topics(reviews, n_topics=5, n_keywords=5):

        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)

        doc_term_matrix = vectorizer.fit_transform(reviews)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

        lda.fit(doc_term_matrix)

        topics = []

        for idx, topic in enumerate(lda.components_):

            keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_keywords:]]

            topics.append((f"Topic {idx+1}", keywords))

        return topics, lda.transform(doc_term_matrix)

 

    # Function to get representative sentences

    def get_representative_sentences(reviews, topic_distributions, n_sentences=1):

        sentences = []

        for topic_idx in range(topic_distributions.shape[1]):

            topic_scores = topic_distributions[:, topic_idx]

            top_indices = topic_scores.argsort()[-n_sentences:]

            topic_sentences = reviews.iloc[top_indices].tolist()

            sentences.append(topic_sentences)

        return sentences

 

    # Function to display topic summaries using markdown

    def display_topic_summary(topics, sentences, section_title):

        st.markdown(f"### {section_title}")

        for i, (topic_name, keywords) in enumerate(topics):

            with st.expander(f"{topic_name}"):

                st.markdown(f"**Keywords:** {', '.join(keywords)}")

                st.markdown("**Representative Sentences:**")

                for sent in sentences[i]:

                    st.markdown(f"- {sent}")

 

    # Run topic modeling and display summaries

    if len(positive_reviews) > 0:

        positive_topics, positive_topic_distributions = extract_topics(positive_reviews)

        positive_sentences = get_representative_sentences(positive_reviews, positive_topic_distributions)

        display_topic_summary(positive_topics, positive_sentences, "Top 5 Best Aspects (Rating >= 4)")

 

    if len(negative_reviews) > 0:

        negative_topics, negative_topic_distributions = extract_topics(negative_reviews)

        negative_sentences = get_representative_sentences(negative_reviews, negative_topic_distributions)

        display_topic_summary(negative_topics, negative_sentences, "Top 5 Issues (Rating <= 2)")

 

    if len(neutral_reviews) > 0:

        neutral_topics, neutral_topic_distributions = extract_topics(neutral_reviews)

        neutral_sentences = get_representative_sentences(neutral_reviews, neutral_topic_distributions)

        display_topic_summary(neutral_topics, neutral_sentences, "Top 5 Neutral Topics (Rating = 3)")

 

   

    # Create a PDF document using FPDF

    pdf = FPDF()

    pdf.add_page()

    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", size=12)

 

    def add_topic_section_to_pdf(pdf, section_title, topics, sentences):

        pdf.set_font("Arial", 'B', 14)

        pdf.cell(200, 10, txt=section_title, ln=True)

        pdf.set_font("Arial", size=12)

        for i, (topic_name, keywords) in enumerate(topics):

            pdf.set_font("Arial", 'B', 12)

            pdf.cell(200, 10, txt=f"{topic_name}", ln=True)

            pdf.set_font("Arial", size=12)

            pdf.multi_cell(0, 10, txt=f"Keywords: {', '.join(keywords)}")

            pdf.multi_cell(0, 10, txt="Representative Sentences:")

            for sent in sentences[i]:

                clean_sent = ''.join(char if ord(char) < 256 else ' ' for char in sent)

                pdf.multi_cell(0, 10, txt=f"- {clean_sent}")

            pdf.ln(5)

 

    # Add sections to PDF

    add_topic_section_to_pdf(pdf, "Top 5 Best Aspects", positive_topics, positive_sentences)

    add_topic_section_to_pdf(pdf, "Top 5 Issues", negative_topics, negative_sentences)

 

    # Save the PDF

    pdf.output("topic_modeling_summary.pdf")

 

    # Streamlit download button

    with open("topic_modeling_summary.pdf", "rb") as f:

        st.download_button(

            label="📄 Download Topic Modeling Summary as PDF",

            data=f,

            file_name="topic_modeling_summary.pdf",

            mime="application/pdf"

        )

 

    # # Display in Streamlit

    # st.markdown("### Top 5 Best Aspects")

    # for summary in positive_summaries:

    #     st.text(summary)

 

    # st.markdown("### Top 5 Issues")

    # for summary in negative_summaries:

    #     st.text(summary)




# # Interactive LDA Visualization

# st.subheader("LDA Topic Visualization")

# lda_vis = pyLDAvis.sklearn.prepare(lda, doc_term_matrix, vectorizer)

# pyLDAvis.save_html(lda_vis, 'lda_vis.html')

# components.html(open("lda_vis.html", 'r', encoding='utf-8').read(), height=800)




# Clustering

# st.subheader("Review Clustering (Sentence-BERT + KMeans)")

# model = SentenceTransformer('all-MiniLM-L6-v2')

# embeddings = model.encode(df['review'].astype(str).tolist())

# kmeans = KMeans(n_clusters=5, random_state=42)

# labels = kmeans.fit_predict(embeddings)

# df['cluster'] = labels

# fig2 = px.histogram(df, x='cluster', title='Review Clusters')

# st.plotly_chart(fig2, use_container_width=True)

 

# Summarization

# st.subheader("Review Summarization")

# sample_reviews = " ".join(df['review'].astype(str).tolist()[:1000])

# summary = summarizer(sample_reviews[:1024])[0]['summary_text']

# st.write("**Summary:**")

# st.success(summary)

 

# BERTopic Visualization

# st.subheader("BERTopic Topic Visualization")

# bertopic_model = BERTopic()

# topics, probs = bertopic_model.fit_transform(df['review'].astype(str).tolist())

# fig_bert = bertopic_model.visualize_topics()

# components.html(fig_bert.to_html(), height=800)

 

# Anomaly Detection (basic)

# st.subheader("Anomaly Detection: Volume Spike")

# if 'date' in filtered_df.columns:

#     count_df = filtered_df.groupby(filtered_df['date'].dt.date).size()

#     fig3 = px.line(count_df, title='Review Volume Over Time')

#     st.plotly_chart(fig3, use_container_width=True)

 

# Heatmap: Rating vs Sentiment

# if not st.sidebar.checkbox("Rating vs Sentiment", True):

#     if 'rating' in filtered_df.columns:

#         st.subheader("Rating vs Sentiment Correlation")

#         filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

#         filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

#         heatmap_data = pd.crosstab(filtered_df['rating'], filtered_df['sentiment_label'])

#         fig4, ax4 = plt.subplots()

#         sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax4)

#         st.pyplot(fig4)




def plot_bar(subplot,filtered_df):

    plt.subplot(1,2,subplot)

    axNewest=sns.barplot(y='Country',x='rating',hue='AppName',data=filtered_df, color='slateblue')

    plt.title('Ratings vs country',fontsize=70)

    # plt.xlabel('Ratings vs Country',fontsize=50)

    plt.ylabel(None)

    # plt.xticks(fontsize=40)

    # plt.yticks(fontsize=40)

    # sns.despine(left=True)

    axNewest.grid(False)

    axNewest.tick_params(bottom=True,left=False)

    return None

 

#move to plotting

if not st.sidebar.checkbox("Visual Charts", True):

 

   

    fig1 = px.pie(

        filtered_df,

        names='sentiment_label',

        # title='Sentiment Distribution',

        width=600,   # Set desired width

        height=400,   # Set desired height

        color_discrete_map={

            'Positive': '#0CE73B',  

            'Negative': '#d62728',

            'Neutral': '#1f77b4'

 

        },

        hole=0.3

        )

 

    fig1.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig1,use_container_width=True)

 

    # Trend Over Time

    # if 'TimeStamp' in filtered_df.columns:

    #     filtered_df['TimeStamp'] = pd.to_datetime(filtered_df['TimeStamp'], errors='coerce')

    #     st.subheader("Sentiment Trend Over Time")

    #     trend_df = filtered_df.groupby([filtered_df['TimeStamp'].dt.to_period('W').dt.start_time, 'sentiment_label']).size().unstack().fillna(0)

    #     st.line_chart(trend_df)

   

    fig = plt.figure(figsize=(15, 5))

    ax = sns.countplot(x='rating',hue='rating', orient='h',dodge=False,data=filtered_df,palette='turbo')

    #ax.set(xlabel='Rating', ylabel='Ratings Count', title='Ratings vs Count')

 

 

    for label in ax.containers:

        ax.bar_label(label)

    st.pyplot(fig)

 

 

    # figNewer = plt.figure(figsize=(15, 5))

    # axar=sns.barplot(x='Country',y='rating',hue='Country',data=filtered_df,palette='Pastel1')

 

    # axar.set(xlabel='Country', ylabel='Ratings', title='Country vs Ratings')

    # for labelN in ax.containers:

    #     axar.bar_label(label)

    # st.pyplot(figNewer)  

    # Aggregate mean rating by country

    mean_ratings = filtered_df.groupby('Country')['rating'].mean().reset_index()

 

    figNewer = plt.figure(figsize=(15, 5))

    axar = sns.barplot(x='Country', y='rating', data=mean_ratings, palette='Pastel1')

 

    axar.set(xlabel='Country', ylabel='Average Rating', title='Average Rating by Country')

 

    # Add value labels on top of each bar with 2 decimal places

    for container in axar.containers:

        axar.bar_label(container, fmt='%.2f')

 

    st.pyplot(figNewer)

 

    # filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

    # filtered_df['year'] = filtered_df['TimeStamp'].dt.year

    # months = ["January", "February", "March", "April", "May", "June",

    #         "July", "August", "September", "October", "November", "December"]

    # filtered_df['month'] = filtered_df['TimeStamp'].dt.month.apply(lambda m: months[m-1])

    # filtered_df['month'] = pd.Categorical(filtered_df['month'], categories=months, ordered=True)

 

    # # Group by Country, AppName, year, month and compute average rating

    # monthly_avg = (filtered_df.groupby(['Country', 'AppName', 'year', 'month'])

    #             .agg(average_rating=('rating', 'mean'))

    #             .reset_index())

 

    # # Create a combined year-month string for the x-axis (sorted chronologically)

    # monthly_avg['year_month'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str)

 

    # # Create a line plot with Plotly Express

    # fig = px.line(

    #     monthly_avg,

    #     x='year_month',

    #     y='average_rating',

    #     color='Country',

    #     line_dash='AppName',

    #     title='Month-on-Month Average App Ratings by Country and App',

    #     labels={'year_month': 'Year-Month', 'average_rating': 'Average Rating'}

    # )

 

    # fig.update_layout(xaxis=dict(tickangle=45))

    # st.plotly_chart(fig, use_container_width=True)

       

   

    # filtered_df["TimeStamp"] = pd.to_datetime(filtered_df["TimeStamp"])

    # # Sort the DF from oldest to most recent recordings

    # filtered_df.sort_values(by="TimeStamp", inplace=True)

    # # Use the column of dates as the DF's index

    # filtered_df.set_index(["TimeStamp"], inplace=True)

 

    # months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # # Create a column that has the year of each date recording

    # filtered_df["year"] = filtered_df.index.year

    # # Create a column that has the month (1-12) of each date recording

    # filtered_df["month"] = filtered_df.index.month

    # # Map the month integers to their proper names

    # filtered_df["month"] = filtered_df["month"].apply(

    #     lambda data: months[data-1]

    # )

 

    # filtered_df["month"] = pd.Categorical(filtered_df["month"], categories=months)

 

    # df_pivot = pd.pivot_table(

    #     filtered_df,

    #     values="rating",

    #     index="year",

    #     columns="month",

    #     aggfunc=np.mean

    # )

 

    # # Plot a bar chart using the DF

    # axo = df_pivot.plot(kind="bar")

    # # Get a Matplotlib figure from the axes object for formatting purposes

    # figo = axo.get_figure()

    # # Change the plot dimensions (width, height)

    # figo.set_size_inches(15, 10)

    # # Change the axes labels

    # axo.set_xlabel("Years")

    # axo.set_ylabel("Average App Ratings")

   

    # # axo=sns.barplot(x='Country',y='rating',hue='Country',data=filtered_df,palette='Pastel1')

    # axo.set(xlabel='Year', ylabel='Ratings', title='Year on Year Ratings')

    # st.pyplot(figo)




   

 

filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

if not st.sidebar.checkbox("Sunburst Chart", True): #by defualt hide the checkbar

    # Sunburst chart creation

    st.write("### Sunburst Chart")

    fig = px.sunburst(

        filtered_df,

        path=[ 'Country','AppName','rating','review','UserName'],  # Define the hierarchy of data

        values='rating',  # Use ratings as values

        color='rating',    # Color the segments by ratings

        color_continuous_scale='RdBu',

        color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']),

        #color_discrete_sequence=px.colors.qualitative.Pastel ,  # Color scale for ratings          

        title=""

    )

 

    fig.update_traces(

    hovertemplate=""

    )

 

    fig.update_layout(width=800,height=800)

    # Display the chart in Streamlit

    st.plotly_chart(fig, use_container_width=True)




# st.sidebar.markdown("### Hierarchical view - TreeMap")

if not st.sidebar.checkbox("TreeMap", True , key='100'): #by defualt hide the checkbar

 

    filtered_df=filtered_df.fillna('end_of_hierarchy')

    fig3 = px.treemap(filtered_df, path = ["Country","AppName","rating","review"],hover_data = ["rating"],

                     color = "review")

   

    fig3.update_traces(

    hovertemplate='<b>Review:</b> %{label}<br><extra></extra>'

    )

 

    st.plotly_chart(fig3, use_container_width=True)

 

def remove_emojis(text):

    # This function removes emojis from the input text

    return text.encode('ascii', 'ignore').decode('ascii')

 

words=filtered_df['review'].dropna().apply(nltk.word_tokenize)

 

# st.sidebar.header("Customer Reviews Word Cloud")

#word_sentiment = st.sidebar.radio("Display Word Cloud for which sentiment?", ('positive', 'neutral', 'negative'))

 

if not st.sidebar.checkbox("Word Cloud", True, key='3'):

   

   # WordCloud

    # stopwords = set(STOPWORDS)

    # st.subheader("Word Cloud")

    # filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

    # sentiment_option = st.selectbox("Select Sentiment for Word Cloud", filtered_df['sentiment_label'].unique())

    # text = " ".join(filtered_df[filtered_df['sentiment_label'] == sentiment_option]['review'].astype(str))

    # wordcloud = WordCloud(stopwords=stopwords,max_words=15, width=600, height=450, background_color='white').generate(text)

    # fig, ax = plt.subplots(figsize=(10, 5))

    # ax.imshow(wordcloud, interpolation='bilinear')

    # ax.axis('off')

    # st.pyplot(fig)

 

    # Use Western Union colors: yellow and black

    western_union_colors = ["#ffe600", "#000000"]  # Yellow and Black

 

    # Optional: create a simple mask for the shape, for example, a rectangle of specific size

    # Or load a Western Union logo silhouette (if you have an image file)

    mask = np.array(Image.open("wuupdated.png"))

 

    st.subheader("Word Cloud")

 

    filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

 

    sentiment_option = st.selectbox("Select Sentiment for Word Cloud", filtered_df['sentiment_label'].unique())

    text = " ".join(filtered_df[filtered_df['sentiment_label'] == sentiment_option]['review'].astype(str))

 

    stopwords = set(STOPWORDS)

 

    def western_union_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

        # Alternate between yellow and black

        return np.random.choice(western_union_colors)

 

    wordcloud = WordCloud(

        stopwords=stopwords,

        max_words=15,

        width=600,

        height=450,

        background_color='white',

        color_func=western_union_color_func,

        # mask=mask,  # Uncomment if you add a mask image

        contour_color='black',

        contour_width=2,

        collocations=False,

        font_path=None  # You can specify a bold font path if you want

    ).generate(text)

 

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wordcloud, interpolation='bilinear')

    ax.axis('off')

    st.pyplot(fig)














    # st.subheader(f"Word cloud for {word_sentiment} sentiment")

    # df = filtered_df[filtered_df['Sentiment'] == word_sentiment]

    # words = ' '.join(df['review'].dropna())

    # processed_words = ' '.join(word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT')

    # processed_words = remove_emojis(processed_words)

   

    # stopwords = set(STOPWORDS)

    # wordcloud = WordCloud(

    #     stopwords=stopwords,

    #     background_color='white',

    #     max_words=100,

    #     width=600,

    #     height=450

    # ).generate(processed_words)

 

    # fig, ax = plt.subplots()

    # ax.imshow(wordcloud, interpolation="bilinear")

    # ax.axis("off")

   

    # st.pyplot(fig)




    # st.title("App Review World Map by Platform and Date")

 

    # # --- Platform Filter

    # platform = st.selectbox("Select Platform", options=filtered_df["AppName"].unique(), index=0)

    # # st.write(platform)

    # # --- Date Range Filter

    # # min_date, max_date = filtered_df["TimeStamp"].min(), filtered_df["TimeStamp"].max()

    # # start_date, end_date = st.date_input("Select date range", [min_date, max_date])

 

    # # --- Filtered Data

    # filtered_df = filtered_df[

    #     (filtered_df["AppName"] == platform)

    #     # (filtered_df["TimeStamp"] >= pd.to_datetime(date1)) &

    #     # (filtered_df["TimeStamp"] <= pd.to_datetime(date2))

    # ]

 

    # filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')




# --- Caching country code conversions

@st.cache_data

def iso2_to_name(code):

    try:

        return pycountry.countries.get(alpha_2=code).name

    except:

        return None

 

@st.cache_data

def name_to_iso3(name):

    try:

        return pycountry.countries.get(name=name).alpha_3

    except:

        return None

 

# --- Caching grouped data computation

 

@st.cache_data(show_spinner=False)

def compute_grouped_data(df):

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df["CountryName"] = df["Country"].apply(iso2_to_name)

    df["ISO3"] = df["CountryName"].apply(name_to_iso3)

    grouped = (

        df.groupby(["CountryName", "ISO3"])

        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))

        .reset_index()

    )

    return grouped

 

# --- Caching figure generation

 

@st.cache_data(show_spinner=False)

 

def generate_figures(grouped):

    figures = []

    font_size = 12

 

    def rating_to_color(rating):

        if rating < 2.5:

            return "#B22222"  # Dark Red

        elif rating < 4.0:

            return "#FF8C00"  # Dark Orange

        else:

            return "#228B22"  # Forest Green

 

    for _, row in grouped.iterrows():

        fill_color = rating_to_color(row["avg_rating"])

 

        fig = go.Figure()

 

        # Country fill

        fig.add_trace(go.Choropleth(

            locations=[row["ISO3"]],

            z=[row["avg_rating"]],

            locationmode="ISO-3",

            colorscale=[[0, fill_color], [1, fill_color]],

            showscale=False,

            marker_line_color="gray",

            marker_line_width=0.5,

            hoverinfo="skip"

        ))

 

        # Annotation box in bottom center

        annotation_text = (

            f"<b>{row['CountryName']}</b><br>"

            f"⭐ Rating: {row['avg_rating']:.2f}<br>"

            f"📝 Reviews: {row['review_count']}"

        )

       

 

        fig.update_layout(

            annotations=[

                dict(

                    x=0.5,

                    y=0.01,

                    xref='paper',

                    yref='paper',

                    showarrow=False,

                    align='center',

                    text=annotation_text,

                    font=dict(size=font_size, color="black"),

                    bgcolor="white",

                    bordercolor="gray",

                    borderwidth=3,

                    opacity=0.98  

                )

            ],

            title={

                "text": f"🌐 Ratings for {row['CountryName']}",

                "x": 0.5,

                "xanchor": "center",

                "font": dict(size=18, family="Arial Black", color="black")

            },

            margin=dict(l=0, r=0, t=50, b=0),

            paper_bgcolor='white',

            plot_bgcolor='white',

            geo=dict(

                showcoastlines=True,

                coastlinecolor="LightGray",

                showland=True,

                landcolor="whitesmoke",

                showocean=True,

                oceancolor="aliceblue",

                showlakes=True,

                lakecolor="lightblue",

                showrivers=True,

                rivercolor="lightblue",

                showcountries=True,

                countrycolor="gray",

                projection_type="equirectangular",

                bgcolor='white',

                resolution=50,                

                showsubunits=True,

                subunitcolor="lightgray",

                showframe=True,

                framecolor="black",

               

                center=dict(lat=20, lon=0),

                projection_scale=1  # Zoom level

 

 

            )

        )

 

        figures.append(fig)

 

    return figures




# --- Main logic

if not st.sidebar.checkbox("World Map", True, key='23'):

       # Display the selected date range with smaller font size above the map

 

    st.markdown(

        """

        <style>

        .date-range-text {

            font-size: 14px;

            font-weight: bold;

            text-align: center;

            margin-bottom: 10px;

            z-index: 9999;

        }

        </style>

        """,

        unsafe_allow_html=True,

    )

 

    # Create and display the date range HTML with the styled class

    date_html = f"<p class='date-range-text'>Ratings from <strong>{date1.strftime('%Y-%m-%d')}</strong> to <strong>{date2.strftime('%Y-%m-%d')}</strong></p>"

    st.markdown(date_html, unsafe_allow_html=True)

 

    with st.spinner("⏳ Loading Country-wise ratings on world map..."):

        grouped = compute_grouped_data(filtered_df)

 

    if grouped.empty:

        st.warning("No records found within the specified date range")

    else:

        map_placeholder = st.empty()

        figures = generate_figures(grouped)

        while True:

            for fig in figures:

                map_placeholder.plotly_chart(fig, use_container_width=True)

                time.sleep(3)




qr_img = Image.open('app_qr_code.png')

# Add vertical space or put this block at the very end of your app

# Convert QR image to base64

buffered = io.BytesIO()

qr_img.save(buffered, format="PNG")

img_str = base64.b64encode(buffered.getvalue()).decode()

 






