import streamlit as st

import plotly.graph_objects as go

import asyncio

import pandas as pd

import base64

import io

import asyncio

import pandas as pd

import qrcode

from concurrent.futures import ThreadPoolExecutor

# from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from streamlit.column_config import TextColumn

# st.set_page_config(page_title="WU Customer Sentiment Analyzer!!!", page_icon=":sparkles:",layout="wide")

 

st.set_page_config(

    page_title="Customer Sentiment Analyzer",

    page_icon="Images/WUNEW.png",  # File must be in the root directory

    layout="wide"

)

# st.title(" :sparkles: Sentiment Anaylzer")

st.markdown('<style>div.block-container{padding-top:0rem;text-align: center}</style>',unsafe_allow_html=True)

import plotly.express as px

import pandas as pd

import numpy as np

from datetime import date, timedelta

from pprint import pprint

import time

from fpdf import FPDF

from sklearn.cluster import KMeans

import warnings

import requests

import datetime

from languages import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
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

nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from streamlit_plotly_events import plotly_events

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import snscrape.modules.twitter as sntwitter

from langdetect import detect

import pandas as pd

import warnings

import streamlit.components.v1 as components

from textblob import TextBlob

import matplotlib.pyplot as plt

# from transformers import pipeline

from nltk.corpus import stopwords

from streamlit_autorefresh import st_autorefresh

st.write(st.__version__)

 

st.markdown("""

    <style>

        /* Hide Streamlit header */

        header {visibility: hidden;}

 

        /* Hide Streamlit footer */

        footer {visibility: hidden;}

 

        /* Hide Main Menu */

        #MainMenu {visibility: hidden;}

    </style>

""", unsafe_allow_html=True)




st.markdown(

    """

    <style>

    /* App background */

    .stApp {

        background-color: #fd0;

        color: black;

    }

 

    html, body, [class*="css"] {

        font-family:  sans-serif;

    }

   

    /* Headers */

    h1, h2, h3, h4, h5, h6 {

        color: black;

        font-weight: bold;

    }

 

    /* Buttons */

    div.stButton > button {

        background-color: black;

        color: #ffe600;

        border: none;

        padding: 0.5em 1em;

        # font-weight: bold;

        border-radius: 1px;

    }

 

    /* Sidebar background and text */

    section[data-testid="stSidebar"] {

        background-color: black;

        # padding: 2px;

    }

 

    section[data-testid="stSidebar"] * {

        color:  white !important;

    }

 

    /* DataFrame styling */

    .stDataFrame {

        background-color: white;

        border: 1px solid black;

        border-radius: 5px;

        padding: 8px;

    }

 

    .stDataFrame table {

        color: black;

        font-weight: 500;

    }

 

    .stDataFrame thead {

        background-color: black;

        color: #FFFF00;

    }

 

    .stVerticalBlock{

      display: block;

    }

    </style>

    """,

    unsafe_allow_html=True

)




st.markdown("""

    <style>

    .date-range-text {

        font-size: 18px;

        color: #4B4B4B;

        text-align: center;

        margin-top: 10px;

        margin-bottom: 20px;

    }

    </style>

""", unsafe_allow_html=True)





def show_timed_warning(message="⚠️ No records found within the specified date range", duration=4):

    # Create three columns to center the message

   

    col1, col2, col3 = st.columns([1, 2, 1])

       

    with col2:

            warning_placeholder = st.empty()

            warning_placeholder.markdown(

                f"""

                <div style="margin-top: 5px;text-align: center; padding: 10px; background-color: #f8d7da;

                            border: 1px solid #f5c6cb; border-radius: 4px;">

                    <strong style="color: black; font-size: 15px;">{message}</strong>

                </div>

                """,

                unsafe_allow_html=True

            )

            time.sleep(duration)

            warning_placeholder.empty()




def show_timed_warning_Sunburst(message="⚠️ Sunburst chart is disabled for date ranges longer than two months", duration=4):

    # Create three columns to center the message

   

    col1, col2, col3 = st.columns([1, 2, 1])

       

    with col2:

            warning_placeholder = st.empty()

            warning_placeholder.markdown(

                f"""

                <div style="margin-top: 5px;text-align: center; padding: 10px; background-color: #f8d7da;

                            border: 1px solid #f5c6cb; border-radius: 4px;">

                    <strong style="color: black; font-size: 15px;">{message}</strong>

                </div>

                """,

                unsafe_allow_html=True

            )

            time.sleep(duration)

            warning_placeholder.empty()

 

def show_timed_warning_TreeMap(message="⚠️ TreeMap chart is disabled for date ranges longer than two months", duration=4):

    # Create three columns to center the message

   

    col1, col2, col3 = st.columns([1, 2, 1])

       

    with col2:

            warning_placeholder = st.empty()

            warning_placeholder.markdown(

                f"""

                <div style="margin-top: 5px;text-align: center; padding: 10px; background-color: #f8d7da;

                            border: 1px solid #f5c6cb; border-radius: 4px;">

                    <strong style="color: black; font-size: 15px;">{message}</strong>

                </div>

                """,

                unsafe_allow_html=True

            )

            time.sleep(duration)

            warning_placeholder.empty()            





# st.sidebar.image("images/wufull.png", use_column_width=True)

 

# Load and encode image

dir = os.path.dirname(__file__)

filename = os.path.join(dir, 'WUNEWEST.png')

with open(filename, "rb") as image_file:

    encoded_image = base64.b64encode(image_file.read()).decode()

 

# Inject custom HTML into sidebar

st.sidebar.markdown(f"""

    <style>

        .no-fullscreen-sidebar img {{

            pointer-events: none;

            user-select: none;

        }}

        [title="View fullscreen"] {{

            display: none !important;

        }}

    </style>

    <div class="no-fullscreen-sidebar" style="text-align: center;">

        <img src="data:image/png;base64,{encoded_image}" style="width: 100%;margin-top: 20px;margin-bottom: 20px;"/>

    </div>

""", unsafe_allow_html=True)

 

st.markdown(

    """

    <style>

    /* Style for download button */

    .stDownloadButton button {

        background-color: black;

        color: #ffe600;

        border: none;

        padding: 0.5em 1em;

        font-weight: bold;

        border-radius: 5px;

        cursor: pointer;

    }

 

    .stDownloadButton button:hover {

        background-color: #333333;

        color: #FFFF00;

    }

    </style>

    """,

    unsafe_allow_html=True

)

 

st.markdown(

    """

    <style>

    /* Disable hover color change for expanders */

    .streamlit-expanderHeader:hover {

        color: inherit !important;

    }

 

    /* Optional: Disable hover color change for all markdown text */

    .markdown-text-container:hover {

        color: inherit !important;

    }

 

    /* Optional: Disable hover effect for all text */

    *:hover {

        color: inherit !important;

    }

    </style>

    """,

    unsafe_allow_html=True

)

 

st.markdown("""

    <style>

        # .reportview-container {

        #     margin-top: -2em;

        # }

        #MainMenu {visibility: hidden;}

        .stDeployButton {display:none;}

        footer {visibility: hidden;}

        #stDecoration {display:none;}

    </style>

""", unsafe_allow_html=True)




st.markdown("""

    <style>

        [data-testid="stVerticalBlock"] {

            # flex: none !important;

            # # width: 596px;

            # position: relative;

            display: inline;

            # flex-direction: column;

            gap: 1rem;

        }

    </style>

""", unsafe_allow_html=True)

 

st.markdown("""

    <style>

        [data-testid="stSidebarUserContent"] {

            # flex: none !important;

            # # width: 596px;

            position: relative;

            # display: inline;

            # flex-direction: column;

            padding : 2rem;

            gap: 1rem;

        }

    </style>

""", unsafe_allow_html=True)

 

st.markdown("""

    <div style='text-align: center; margin-top: 25px;'>

        # <h1 style='color: black; font-weight: bold;  font-size: 45px;'>

            Customer Sentiment Analyzer

        </h1>

    </div>

""", unsafe_allow_html=True)

 

app_url = ""

 

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

 

# count = st_autorefresh(interval=3600000, key="fizzbuzzcounter")

sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

# st.cache_data.clear()

# print(st.__version__)

translator = Translator()

 

# # wu_mask = np.array(Image.open('wul.png'))

# dir = os.path.dirname(__file__)

# filename = os.path.join(dir, 'Images/wufull.png')

# image = Image.open(filename)

# left_co, cent_co = st.columns(2)

# with cent_co:

#     # st.image(image, caption='',width=150)

#     st.markdown("<h6 style='color: black; font-weight: bold; font-family: PP Right Grotesk; font-size: 35px;'>WU Customer Voice Analyzer<h6/>", unsafe_allow_html=True)

 

# dir = os.path.dirname(__file__)

# filename = os.path.join(dir, 'Images/wufull.png')

# image = Image.open(filename)

 

# Optional logo/image

# st.image(image, width=150)






BEARER_TOKEN = ""

# Setup VADER Sentiment Analyzer

sid = SentimentIntensityAnalyzer()

 

# --- Android Review Fetch (No caching here!) ---

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

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    df['AppName'] = app_name

    df['Country'] = country.lower()

    # try:

    #     df['translated_text'] = df['review'].apply(

    #         lambda x: translator.translate(x, dest='en').text if isinstance(x, str) else x

    #     )

    # except Exception as e:

    #     print(f"Translation failed: {e}")

    #     df['translated_text'] = df['review']

    df.rename(columns={

        'content': 'review',

        'userName': 'UserName',

        'score': 'rating',

        'at': 'TimeStamp',

        'replyContent': 'WU_Response'

    }, inplace=True)

    return df

 

def fetch_all_android(app_details):

    frames = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [executor.submit(load_android_data, app_id, country, app_name)

                   for app_id, country, app_name in app_details]

        for future in futures:

            try:

                result = future.result()

                frames.append(result)

            except Exception as e:

                print(f"Android fetch failed: {e}")

                frames.append(pd.DataFrame())

    if frames:

        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()

 

# --- iOS Review Fetch (No caching here!) ---

def fetch_ios_reviews(app_id, country_code, pages=5):

    reviews = []

    for p in range(1, pages + 1):

        url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/page={p}/id={app_id}/sortBy=mostRecent/json"

        try:

            resp = requests.get(url)

            if resp.status_code != 200:

                continue

            entries = resp.json().get("feed", {}).get("entry", [])[1:]  # skip app metadata

            for entry in entries:

                reviews.append({

                    "rating": entry.get("im:rating", {}).get("label"),

                    "date": entry.get("updated", {}).get("label"),

                    "review": entry.get("content", {}).get("label"),

                    "WU_Response": entry.get("im:developerResponse", {}).get("label"),

                    "UserName": entry.get("author", {}).get("name", {}).get("label"),

                    "Platform": "iOS",

                    "AppName": "iOS",

                    "Country": country_code,

                    "AppID": app_id

                })

        except Exception as e:

            print(f"Error for {app_id}-{country_code}: {e}")

            continue

    return pd.DataFrame(reviews)

 

def fetch_all_ios(app_country_list, pages=5):

    frames = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [executor.submit(fetch_ios_reviews, app_id, cc, pages)

                   for app_id, cc in app_country_list]

        for future in futures:

            try:

                df = future.result()

                if not df.empty:

                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                    df["TimeStamp"] = df["date"].dt.strftime('%Y-%m-%d')

                    frames.append(df)

            except Exception as e:

                print(f"iOS fetch failed: {e}")

    if frames:

        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()

 

# --- All App Details, as before ---

app_details = [

    ('com.westernunion.android.mtapp', 'us', 'Android'),

    ('com.westernunion.moneytransferr3app.eu','fr','Android'),  

    ('com.westernunion.moneytransferr3app.au', 'au', 'Android'),

    # ('com.westernunion.moneytransferr3app.eu','de','Android'),

    ('com.westernunion.moneytransferr3app.ca', 'ca', 'Android'),  

    # ('com.westernunion.moneytransferr3app.eu','it','Android'),

    # ('com.westernunion.moneytransferr3app.eu3','se','Android'),

    ('com.westernunion.moneytransferr3app.nz', 'nz', 'Android'),

    # ('com.westernunion.android.mtapp', 'co', 'Android'),

    ('com.westernunion.moneytransferr3app.nl','nl','Android'),

    ('com.westernunion.moneytransferr3app.acs3','br','Android'),

    ('com.westernunion.moneytransferr3app.eu2','be','Android'),

    ('com.westernunion.moneytransferr3app.eu3','no','Android'),

    # ('com.westernunion.moneytransferr3app.eu','at','Android'),    

    ('com.westernunion.moneytransferr3app.eu2','ch','Android'),

    ('com.westernunion.moneytransferr3app.sg','sg','Android'),

    # ('com.westernunion.moneytransferr3app.eu3','dk','Android'),

    # ('com.westernunion.moneytransferr3app.eu','ie','Android'),

    ('com.westernunion.moneytransferr3app.pt','pt','Android'),

    ('com.westernunion.moneytransferr3app.eu4','po','Android'),

    ('com.westernunion.moneytransferr3app.eu3','po','Android'),

    ('com.westernunion.moneytransferr3app.apac','my','Android'),

    ('com.westernunion.moneytransferr3app.hk','hk','Android'),

    # ('com.westernunion.moneytransferr3app.ae', 'ae', 'Android'),

    ('com.westernunion.moneytransferr3app.bh', 'bh', 'Android'),    

    ('com.westernunion.moneytransferr3app.kw', 'kw', 'Android'),

    ('com.westernunion.moneytransferr3app.qa', 'qa', 'Android'),

    ('com.westernunion.moneytransferr3app.sa', 'sa', 'Android'),

    ('com.westernunion.moneytransferr3app.in', 'in', 'Android'),

    ('com.westernunion.moneytransferr3app.th', 'th', 'Android')  

]

 

app_country_list = [

    ("424716908", "us"),

    ("1045347175","fr"),

    ("1122288720", "au"),

    # ("1045347175", "de"),

    ("1110191056","ca"),

    # ("1045347175","it"),

    # ("1152860407","se"),

    ("1268771757","es"),

    ("1226778839","nz"),

    ("1199782520","nl"),

    ("1148514737","br"),

    ("1110240507","be"),

    ("1152860407","no"),

    ("1045347175","at"),

    ("1110240507","ch"),

    ("1451754888","ch"),

    # ("1152860407","dk"),

    # ("1045347175","ie"),

    ("1229307854","pt"),

    ("1168530510","pl"),

    ("1152860407","fi"),

    ("1165109779","hk"),

    # ("1171330611","ae"),

    # ("1329774999","co"),

    ("1314010624","bh"),

    ("1304223498","cl"),

    ("1459023219","jo"),

    ("1173794098","kw"),

    ("1483742169","mv"),

    ("1459024696","sa"),

    ("1459226729","th"),

    ("1173792939","qa"),

    ("1150872438","in")

]

 

# --- MAIN STREAMLIT BLOCK ---

# st.title("🌍 Western Union Reviews Dashboard")

@st.cache_data(ttl=86400, show_spinner=False)

def get_all_reviews(app_details, app_country_list):

    finaldfandroid = fetch_all_android(app_details)

    finaldfios = fetch_all_ios(app_country_list, pages=5)

    if not finaldfandroid.empty and not finaldfios.empty:

        finaldf = pd.concat([finaldfandroid, finaldfios], ignore_index=True)

    elif not finaldfandroid.empty:

        finaldf = finaldfandroid

    elif not finaldfios.empty:

        finaldf = finaldfios

    else:

        finaldf = pd.DataFrame()

    return finaldf

 

with st.spinner("Fetching Android & iOS reviews..."):

    finaldf = get_all_reviews(app_details, app_country_list)

 

finaldf.columns = finaldf.columns.str.strip("'")

finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]

 

# Convert TimeStamp to datetime for filtering

finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"])

# finaldf["TimeStampFormatted"] = finaldf["TimeStamp"].dt.strftime("%d/%m/%Y")

 

today = datetime.date.today()

default_start = today.replace(day=1)

default_end = today

 

col1, col2 = st.columns((2))

with col1:

    date1 = st.date_input("**Start Date**", value=default_start)

with col2:

    date2 = st.date_input("**End Date**", value=default_end)

 

# Convert selected dates to datetime

date1 = pd.to_datetime(date1)

date2 = pd.to_datetime(date2)

 

# Filter the dataframe based on selected date range

filtered_df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)]

 

try:

 df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)].copy()

except KeyError:

 df = pd.DataFrame()

 

country_map = {

    "in": "India",

    "au": "Australia",

    "us": "United States",

    "ca": "Canada",

    "nz": "New Zealand",

    "uk": "United Kingdom",

    "sg": "Singapore",

    "de": "Germany",

    "fr": "France",

    "th": "Thailand",

    "hk": "HongKong",

    "be": "Belgium",

    "bh": "Bahrain",

    "br": "Brazil",

    "ch": "Switzerland",

    "cl": "Chile",

    "es": "Spain",

    "kw": "Kuwait",

    "my": "Malaysia",

    "nl": "Netherlands",

    "no": "Norway",

    "qa": "Qatar",

    "sa": "Saudi Arabia",

    "fi": "Finland"

 

}

 

# Create list of full country names for dropdown

country_names = [country_map.get(code, code) for code in df["Country"].unique()]

 

# Sidebar country selection

selected_country_names = st.sidebar.multiselect(

    "**Country Selection**",

    options=sorted(country_names),

    placeholder="Select one or more countries"

)

 

# Convert selected country names back to codes

selected_country_codes = [code for code, name in country_map.items() if name in selected_country_names]

 

# Filter by country

if not selected_country_codes:

    df1 = df.copy()

else:

    df1 = df[df["Country"].isin(selected_country_codes)]

 

# Sidebar app type selection

region = st.sidebar.multiselect(

    "**Select the App Type**",

    options=sorted(df["AppName"].unique()),

    placeholder="Select one or more app types"

)

 

# Filter by app type

if not region:

    df2 = df1.copy()

else:

    df2 = df1[df1["AppName"].isin(region)]

 

# Final filtered DataFrame

if not selected_country_codes and not region:

    filtered_df = df.copy()

else:

    filtered_df = df2





if 'rating' in filtered_df.columns:

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce') # convert rating to numeric (int/float)

    rating = st.sidebar.slider("**Filter by Ratings Range**", 1, 5, (1, 5))

    if rating:

        filtered_df = filtered_df[(filtered_df['rating'] >= rating[0]) & (filtered_df['rating'] <= rating[1])]

 

# # Function to apply row-wise styling

# def highlight_rating(row):

#     try:

#         rating = int(row['rating']) # Convert to int

#     except:

#         return [''] * len(row) # No styling if conversion fails

 

#     if rating >= 4:

#         return ['background-color: lightgreen'] * len(row)

#     elif rating == 3:

#         return ['background-color: yellow'] * len(row)

#     else:

#         return ['background-color: salmon'] * len(row)

 

# df=df.reset_index(drop=True)

# styled_df = df.style.apply(highlight_rating, axis=1)

# st.write(styled_df)

 

def show_centered_warning(message="⚠️ No records found within the specified date range"):

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.warning(message)





if filtered_df.empty:

  #st.warning("No records found within the specified date range")

   show_centered_warning()

else:

 filtered_df = filtered_df.reset_index(drop=True)

 filtered_df.index += 1  # Start index from 1

 filtered_df.index.name = "S.No."  # Give index column a name

 

 # filtered_df_display=filtered_df_display.head()

 filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

 filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

 filtered_df['sentiment_score'] = filtered_df['sentiment_score'].apply(lambda x: int(x * 100) / 100.0)

 

 #filtered_df['TimeStamp']=pd.to_datetime(filtered_df["TimeStamp"]).dt.date

 filtered_df = filtered_df.reindex(['TimeStamp', 'review','rating','sentiment_score','sentiment_label','Country','AppName','appVersion','UserName'], axis=1)

search_query = st.text_input("**Search Reviews :**")

st.markdown("<br>", unsafe_allow_html=True)

 

# st.write("")

 

# Filter the DataFrame in place based on the search query on the 'review' column

if search_query:

    with st.spinner("🔍 Fetching reviews..."):

        # filtered_df = filtered_df[filtered_df['review'].str.contains(search_query, case=False, na=False)]

       

        placeholder = st.empty()

        progress_bar = st.progress(0)

 

        for i in range(100):

            time.sleep(0.01)  # Simulate loading

            progress_bar.progress(i + 1)

            placeholder.text(f"Fetching reviews... {i+1}%")

 

        # Actual filtering

        filtered_df = filtered_df[filtered_df['review'].str.contains(search_query, case=False, na=False)]

 

        placeholder.empty()

        progress_bar.empty()

 

def format_column_label(s):

    # Split by underscores and capitalize each part

    return '_'.join(word.capitalize() for word in s.split('_'))

 

def format_column_label(col):

    custom_labels = {

        "TimeStamp": "Date",    

        "AppName": "App Type",

        "UserName": "User Name",

        "appVersion":"Version",

        "sentiment_score":"Sentiment Score",

        "sentiment_label":"Sentiment Label"

    }

    if col in custom_labels:

        return custom_labels[col]

    return '_'.join(word.capitalize() for word in col.split('_'))

 

def to_title_case_with_underscores(s):

    return '_'.join(word.capitalize() for word in s.split('_'))

 

column_config = {

    col: st.column_config.TextColumn(label=to_title_case_with_underscores(col))

    for col in filtered_df.columns

}

 

column_config = {

    col: st.column_config.TextColumn(label=format_column_label(col))

    for col in filtered_df.columns

}

 

# column_config = {

#     "Review": TextColumn("Review", width="large"),

#     # Add other columns as needed

# }

 

filtered_df["Country"] = filtered_df["Country"].map(country_map).fillna(filtered_df["Country"])




# column_config = {col: st.column_config.TextColumn(label=col.capitalize()) for col in filtered_df.columns}

# filtered_df.columns = [col.capitalize() for col in filtered_df.columns]

if not filtered_df.empty:

#  st.dataframe(filtered_df, height=300, use_container_width=True)

 

    progress = st.progress(0)

    status = st.empty()

 

    for i in range(100):

        time.sleep(0.005)  # Simulate loading

        progress.progress(i + 1)

        status.text(f"Loading Data... {i + 1}%")

 

    progress.empty()

    status.empty()

 

    # Display the filtered DataFrame

    filtered_df["TimeStamp"] = pd.to_datetime(filtered_df["TimeStamp"], unit='ms')

    # Create a formatted column for display

    filtered_df["TimeStamp"] = filtered_df["TimeStamp"].dt.strftime('%Y/%m/%d')

 

    st.dataframe(filtered_df,column_config=column_config, height=275, use_container_width=True)

    st.success(f"✅ Displaying {len(filtered_df)} reviews.")

    st.markdown("<br>", unsafe_allow_html=True)

 

   

 

    # st.markdown(f"""

    # <div style='color: green; font-size: 16px; font-weight: 500;'>

    #     ✅ Displaying {len(filtered_df)} reviews.

    # </div>

    # """, unsafe_allow_html=True)




 

 

# Download button just below the data grid

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")






   

st.sidebar.markdown("**Select from the below options**")

 

if  st.sidebar.checkbox("Language Translation", False):

    source_text = st.text_area("**Enter Text to translate:**", height=100)

    default_language = 'English'  

    default_index = languages.index(default_language) if default_language in languages else 0

 

    # Set default selected language using index

    target_language = st.selectbox("**Select target language:**", languages, index=default_index)

    st.markdown("<br>", unsafe_allow_html=True)

 

    translate = st.button('Translate')

    if translate:

        translator = Translator()

        out = translator.translate(source_text, dest=target_language)

        st.write(out.text)





if  st.sidebar.checkbox("Keyword Analysis", False):

    if not filtered_df.empty:    

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

    else:

        # st.warning("⚠️ No records found within the specified date range")

        show_timed_warning()

 

st.markdown("<br><br>", unsafe_allow_html=True)

 

if  st.sidebar.checkbox("Topic Modeling", False):

    # Topic Modeling (LDA)

 

   

 

    positive_reviews = filtered_df[filtered_df['rating'] >= 4]['review']

    negative_reviews = filtered_df[filtered_df['rating'] <= 2]['review']

    neutral_reviews = filtered_df[filtered_df['rating'] == 3]['review']

 

   

# Function to extract topics and keywords

    def extract_topics(reviews, n_topics=5, n_keywords=5):

        vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)

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

 

    def display_topic_summary(topics, sentences, section_title):

        st.markdown(f"### {section_title}")

        for i, (topic_name, keywords) in enumerate(topics):

            # Join keywords into a string to use as the expander header

            keywords_str = ', '.join(keywords)

            with st.expander(keywords_str):

                #st.markdown(f"**Keywords:** {keywords_str}")

                st.markdown("**Representative Sentences:**")

                for sent in sentences[i]:

                    st.markdown(f"- {sent}")

                   

   

    # Initialize topic variables

    positive_topics, positive_sentences = [], []

    negative_topics, negative_sentences = [], []

    neutral_topics, neutral_sentences = [], []

 

    # Run topic modeling and display summaries

    if len(positive_reviews) > 0:

        positive_topics, positive_topic_distributions = extract_topics(positive_reviews)

        positive_sentences = get_representative_sentences(positive_reviews, positive_topic_distributions)

        display_topic_summary(positive_topics, positive_sentences, "Top 5 Best Aspects (Rating >= 4)")

        st.divider()

 

    if len(negative_reviews) > 0:

        negative_topics, negative_topic_distributions = extract_topics(negative_reviews)

        negative_sentences = get_representative_sentences(negative_reviews, negative_topic_distributions)

        display_topic_summary(negative_topics, negative_sentences, "Top 5 Issues (Rating <= 2)")

        st.divider()

   

 

    if len(neutral_reviews) > 0:

        neutral_topics, neutral_topic_distributions = extract_topics(neutral_reviews)

        neutral_sentences = get_representative_sentences(neutral_reviews, neutral_topic_distributions)

        display_topic_summary(neutral_topics, neutral_sentences, "Top 5 Neutral Topics (Rating = 3)")

        st.divider()

   

   

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

    add_topic_section_to_pdf(pdf, "Top 5 Neutral Aspects", neutral_topics, neutral_sentences)

 

    # Save the PDF

    pdf.output("topic_modeling_summary.pdf")

 

    #Download button

    if positive_topics or negative_topics or neutral_topics:

        st.subheader("Topic Modeling")

    # Save the PDF

        pdf.output("topic_modeling_summary.pdf")

 

        # Download button

        with open("topic_modeling_summary.pdf", "rb") as f:

            st.download_button(

                label="📄 Download Topic Modeling Summary as PDF",

                data=f,

                file_name="topic_modeling_summary.pdf",

                mime="application/pdf"

            )

    else:

        # st.info("📄 No topic modeling data available for download.")

          show_timed_warning()

 

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

if st.sidebar.checkbox("Visual Charts", False):

  if not filtered_df.empty:  

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(

        "<div style='text-align: center; font-size: 18px;'><b>Consolidated Sentiment across Countries</b></div>",

        unsafe_allow_html=True

    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

 

 

   




    fig = px.pie(

    filtered_df,

    names='sentiment_label',

    # title='Sentiment Distribution',

    color='sentiment_label',

    color_discrete_map={

        'Positive': 'green',

        'Negative': 'red',

        'Neutral': 'yellow'

    },

    hole=0.2

    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show percentage and label inside the chart

    fig.update_traces(textposition='inside', textinfo='percent+label')  

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br><b>Consolidated Ratings across Countries<b><br>", unsafe_allow_html=True)

 

    fig = plt.figure(figsize=(15, 5))

    ax = sns.countplot(x='rating',hue='rating', orient='h',dodge=False,data=filtered_df,palette='turbo')

    #ax.set(xlabel='Rating', ylabel='Ratings Count', title='Ratings vs Count')

 

 

    for label in ax.containers:

        ax.bar_label(label)

    st.pyplot(fig)

 

    st.markdown("<br><br>", unsafe_allow_html=True)

    # st.markdown("<br><br>", unsafe_allow_html=True)

 

 

    mean_ratings = filtered_df.groupby('Country')['rating'].mean().reset_index()

    figNewer = plt.figure(figsize=(15, 5))

    axar = sns.barplot(x='Country', y='rating', data=mean_ratings, palette='Pastel1')

    st.markdown("<br><b>Average Rating By Country<b><br>", unsafe_allow_html=True)

    axar.set(xlabel='Country', ylabel='Average Rating', title='')

 

    # Add value labels on top of each bar with 2 decimal places

    for container in axar.containers:

        axar.bar_label(container, fmt='%.2f')

 

    st.pyplot(figNewer)

    # st.divider()

    st.markdown("<br><br>", unsafe_allow_html=True)

   

 

   # ---- issue keywords and funnel labels ----

    issue_keywords = {

        'Crashes': 'Freezes',

        'time': 'Interval',

        'Hang': 'hangs',

        'Bugs': 'bug',

        'Performance': 'performance',

        'customer': 'helpdesk',

        'update': 'update',

        'notification': 'alert',

        'otp': 'message',

        'ui': 'interface',

        'app': 'application',

       

       

    }

    funnel_labels = ['All Reviews', 'Filtered Negatives'] + list(issue_keywords.keys()) + ['Other Issues']

 

    # ---- Filter for negative sentiment_label and rating 1,2,3 ----

    filtered_negatives = filtered_df[

        (filtered_df['sentiment_label'].str.lower() == 'negative') &

        (filtered_df['rating'].isin([1,2,3]))

    ]

 

    # ---- Compute counts and index mapping ----

    all_reviews = len(filtered_df)

    counts = [all_reviews, len(filtered_negatives)]

 

    stage_indices = {

        'All Reviews': df.index.tolist(),

        'Filtered Negatives': filtered_negatives.index.tolist(),

    }

 

    covered_indices = set()

    for issue, keyword in issue_keywords.items():

        mask = filtered_negatives['review'].fillna("").str.lower().str.contains(keyword.lower(), na=False)

        indices = filtered_negatives[mask].index.tolist()

        counts.append(len(indices))

        stage_indices[issue] = indices

        covered_indices.update(indices)

 

    other_issues_indices = filtered_negatives.drop(index=list(covered_indices)).index.tolist()

    counts.append(len(other_issues_indices))

    stage_indices['Other Issues'] = other_issues_indices

 

    # ---- Color theme ----

    custom_colors = [

        "#1f77b4", # blue

        "#d62728", # red

        "#ff7f0e", # orange

        "#2ca02c", # green

        "#9467bd", # purple

        "#8c564b", # brown

        "#bcbd22", # olive

        "#7f7f7f" # gray

    ]

 

    # ---- Plotly funnel chart ----

    fig = go.Figure(go.Funnel(

        y=funnel_labels,

        x=counts,

        textinfo="value+percent initial",

        marker=dict(color=custom_colors)

 

    ))

    # fig.update_traces(hovertemplate='')

    fig.update_layout(

        title=dict(

            text="",

            x=0.5,

            xanchor="center"

        ),

        margin=dict(l=160, r=40, t=60, b=20)

    )

 

    # st.subheader("Click Keywords below to see detailed Customer Reviews")

    st.markdown("<br><b>Click Keywords below to see detailed Customer Reviews<b><br>", unsafe_allow_html=True)

    selected = plotly_events(fig, click_event=True, hover_event=False)

 

    if selected:

        idx = selected[0]['pointIndex']

        label = funnel_labels[idx]

        st.markdown(f"**Reviews for Stage: {label}**")

        indices = stage_indices.get(label, [])

        if indices:

          with st.spinner("⏳ Loading Customer Reviews..."):

               st.dataframe(filtered_df.loc[indices].reset_index(drop=True))

        else:

            st.info('No reviews for this stage.')

    else:

        st.markdown(

            """

            <div style="justify-content: center;">

                 <div style="background-color: #eaf4fb; color: #262730; border-left: .5rem solid #1c83e1;

                #             padding: 1rem 1.5rem; border-radius: .25rem; font-size: 1.1rem; width: fit-content;">

             

            </div>

            """,

            unsafe_allow_html=True,

        )

  else:

    #  st.warning("⚠️ No records found within the specified date range")

    show_timed_warning()




# if st.sidebar.checkbox("Interactive Sunburst", False):

#     if not filtered_df.empty:    

#         filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

#         date_diff = (date2 - date1).days  

#         if date_diff <= 61:

#                 st.write("### Interactive Sunburst")

#                 fig = px.sunburst(

#                     filtered_df,

#                     path=['Country', 'AppName', 'rating', 'review', 'UserName'],

#                     values='rating',

#                     color='rating',

#                     color_continuous_scale='RdBu',

#                     color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']),

#                     title=""

#                 )

#                 fig.update_traces(hovertemplate="")

#                 fig.update_layout(width=800, height=800)

#                 fig.update_layout(coloraxis_showscale=False)

#                 st.plotly_chart(fig, use_container_width=True)

#         else:

#                 # st.warning("⚠️ **Sunburst chart is disabled for date ranges longer than two months.**")

#                 show_timed_warning_Sunburst()

#     else:

#         #  st.warning("⚠️ No records found within the specified date range")

#         show_timed_warning()

 

if st.sidebar.checkbox("Interactive Sunburst", False):

    if not filtered_df.empty:

        filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

        date_diff = (date2 - date1).days

 

        if date_diff <= 61:

            st.write("### Interactive Sunburst")

 

            fig = px.sunburst(

                filtered_df,

                path=['Country', 'AppName', 'rating', 'review', 'UserName'],

                values='rating',

                color='rating',

                color_continuous_scale='RdBu',

                color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']),

                # title="App Ratings and Reviews by Country"

            )

 

            fig.update_traces(

                hovertemplate=""

            )

 

            fig.update_layout(

                width=900,

                height=900,

                margin=dict(t=50, l=0, r=0, b=0),

                coloraxis_showscale=False,

                font=dict(family="Arial", size=12),

                paper_bgcolor="white",

                plot_bgcolor="white"

            )

 

            st.plotly_chart(fig, use_container_width=True)

        else:

            show_timed_warning_Sunburst()

    else:

        show_timed_warning()






# st.sidebar.markdown("### Hierarchical view - TreeMap")

if st.sidebar.checkbox("Interactive TreeMap", False , key='100'):

 if not filtered_df.empty:  

    date_diff = (date2 - date1).days  

    if date_diff <= 61:

        st.write("### Interactive TreeMap")

        filtered_df=filtered_df.fillna('end_of_hierarchy')

        fig3 = px.treemap(filtered_df, path = ["Country","AppName","rating","review"],hover_data = ["rating"],

                        color = "review")

   

        fig3.update_traces(

        hovertemplate=''

        # <b>Review:</b> %{label}<br><extra></extra>

        )

 

        st.plotly_chart(fig3, use_container_width=True)

    else:

            # st.warning("⚠️ **TreeMap chart is disabled for date ranges longer than two months.**")

            show_timed_warning_TreeMap()

 else:

        #  st.warning("⚠️ No records found within the specified date range")

        show_timed_warning()

     

 

def remove_emojis(text):

    # This function removes emojis from the input text

    return text.encode('ascii', 'ignore').decode('ascii')




# st.sidebar.header("Customer Reviews Word Cloud")

#word_sentiment = st.sidebar.radio("Display Word Cloud for which sentiment?", ('positive', 'neutral', 'negative'))

 

if st.sidebar.checkbox("Word Cloud", False, key='3'):

 if not filtered_df.empty:      

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

 

    words=filtered_df['review'].dropna().apply(nltk.word_tokenize)

    # Use Western Union colors: yellow and black

    western_union_colors = ["#ffe600", "#000000"]  # Yellow and Black

 

    # Optional: create a simple mask for the shape, for example, a rectangle of specific size

    # Or load a Western Union logo silhouette (if you have an image file)

    mask = np.array(Image.open("Images/wuupdated.png"))

 

    st.subheader("Word Cloud")

   

    filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))

 

    # sentiment_option = st.selectbox("Select Sentiment for Word Cloud", filtered_df['sentiment_label'].unique())

    st.markdown("<h4 style='text-align: center; font-weight: bold;'>Select Sentiment for Word Cloud</h4>", unsafe_allow_html=True)

 

 

    sentiment_option = st.selectbox("", filtered_df['sentiment_label'].unique())

    st.markdown("<br>", unsafe_allow_html=True)

 

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

 else:

    #   st.warning("⚠️ No records found within the specified date range")

    show_timed_warning()

 

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

# def compute_grouped_data(df):

#     df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

#     df["CountryName"] = df["Country"].apply(iso2_to_name)

#     df["ISO3"] = df["CountryName"].apply(name_to_iso3)

#     grouped = (

#         df.groupby(["CountryName", "ISO3"])

#         .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))

#         .reset_index()

#     )

#     return grouped

 

def compute_grouped_data(df):

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

   

    grouped = (

        df.groupby("Country")

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

 

        fig.add_trace(go.Choropleth(

            locations=[row["Country"]],

            z=[row["avg_rating"]],

            locationmode="country names",

            colorscale=[[0, fill_color], [1, fill_color]],

            showscale=False,

            marker_line_color="gray",

            marker_line_width=0.5,

            hoverinfo="skip"

        ))

 

        annotation_text = (

            f"<b>{row['Country']}</b><br>"

            f"⭐ Rating: {row['avg_rating']:.2f}<br>"

            f"📝 Reviews: {row['review_count']}"

        )

 

        fig.update_layout(

            annotations=[dict(

                x=0.5, y=0.01, xref='paper', yref='paper',

                showarrow=False, align='center',

                text=annotation_text,

                font=dict(size=font_size, color="black"),

                bgcolor="white", bordercolor="gray",

                borderwidth=3, opacity=0.98

            )],

            title={

                "text": f"🌐 Ratings for {row['Country']}",

                "x": 0.5, "xanchor": "center",

                "font": dict(size=18, family="Arial Black", color="black")

            },

            margin=dict(l=0, r=0, t=50, b=0),

            paper_bgcolor='white',

            plot_bgcolor='white',

            geo=dict(

                showcoastlines=True, coastlinecolor="LightGray",

                showland=True, landcolor="whitesmoke",

                showocean=True, oceancolor="aliceblue",

                showlakes=True, lakecolor="lightblue",

                showrivers=True, rivercolor="lightblue",

                showcountries=True, countrycolor="gray",

                projection_type="equirectangular",

                bgcolor='white', resolution=50,

                showsubunits=True, subunitcolor="lightgray",

                showframe=True, framecolor="black",

                center=dict(lat=20, lon=0),

                projection_scale=1

            )

        )

 

        figures.append(fig)

 

    return figures





# --- Main logic

if st.sidebar.checkbox("World Map", False, key='23'):

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

    if not filtered_df.empty:

        # st.write(filtered_df)

        date_html = f"<p class='date-range-text'>Ratings from <strong>{date1.strftime('%Y-%m-%d')}</strong> to <strong>{date2.strftime('%Y-%m-%d')}</strong></p>"

        st.divider()

        st.markdown("<br>", unsafe_allow_html=True)

 

        st.markdown(date_html, unsafe_allow_html=True)        

 

        with st.spinner("⏳ Loading Country-wise ratings on world map..."):

            grouped = compute_grouped_data(filtered_df)

 

        if grouped.empty:

            # st.warning("No records found within the specified date range")

            show_timed_warning()

        else:

            st.markdown("<br><br>", unsafe_allow_html=True)

 

       

 

            map_placeholder = st.empty()

            figures = generate_figures(grouped)

            while True:

                for fig in figures:

                    map_placeholder.plotly_chart(fig, use_container_width=True)

                    time.sleep(2)

    else:

         show_timed_warning()

 

qr_img = Image.open('app_qr_code.png')

# Add vertical space or put this block at the very end of your app

# Convert QR image to base64

buffered = io.BytesIO()

qr_img.save(buffered, format="PNG")

img_str = base64.b64encode(buffered.getvalue()).decode()

 

st.markdown(f"""

    <style>

 

    .fixed-bottom-right {{

        position: fixed;

        right: 0;

        bottom: 0;

        margin: 20px;

        z-index: 1000;

        text-align: right;

    }}

    .bottom-link {{

        font-size: 14px;

        color: #003366;

        font-weight: bold;

        text-decoration: none;

        background: #fff;

        padding: 6px 12px;

        border-radius: 5px;

        box-shadow: 0 2px 8px rgba(0,0,0,0.08);

    }}

 

    </style>

 

   

    # <div class="fixed-bottom-right">

    #     <a href="mailto:test@gmail.com?subject=Feedback | Issue | Suggestion" class="bottom-link">

    #       Feedback

    #     </a>

 

    # </div>

""", unsafe_allow_html=True)












