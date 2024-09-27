import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from io import BytesIO
import json
import warnings
from languages import *
import nltk
import re
import plotly.express as px
import os
from google_play_scraper import Sort, reviews_all
from app_store_scraper import AppStore
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from PIL import Image
from googletrans import Translator
import plotly.express as px
# from languages import *
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
nltk.download('punkt')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# st.cache_data.clear()

st.markdown('<style>div.block-container{padding-top:1rem;text-align: center}</style>',unsafe_allow_html=True)
translator = Translator()




@st.cache_data(persist=True)
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
    try:
     df['translated_text'] = df['review'].apply(lambda x: translator.translate(x, dest='English').text) 
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
    ('com.westernunion.moneytransferr3app.au', 'au', 'Android'),
    ('com.westernunion.moneytransferr3app.ae', 'ae', 'Android'),
    ('com.westernunion.moneytransferr3app.eu','at','Android'),      
    ('com.westernunion.moneytransferr3app.eu2','be','Android'),
    ('com.westernunion.moneytransferr3app.bh', 'bh', 'Android'),
    ('com.westernunion.moneytransferr3app.ca', 'ca', 'Android'),    
    ('com.westernunion.moneytransferr3app.eu2','ch','Android'),
    ('com.westernunion.moneytransferr3app.eu','de','Android'),
    ('com.westernunion.moneytransferr3app.eu3','dk','Android'),
    ('com.westernunion.moneytransferr3app.eu','fr','Android'),    
    ('com.westernunion.moneytransferr3app.eu','gb','Android'),
     ('com.westernunion.moneytransferr3app.eu','ie','Android'),
    ('com.westernunion.moneytransferr3app.eu','it','Android'),
    ('com.westernunion.moneytransferr3app.kw', 'kw', 'Android'),
    ('com.westernunion.moneytransferr3app.eu3','no','Android'),
    ('com.westernunion.moneytransferr3app.nl','nl','Android'),
    ('com.westernunion.moneytransferr3app.nz', 'nz', 'Android'),
    ('com.westernunion.moneytransferr3app.pt','pt','Android'),
    ('com.westernunion.moneytransferr3app.qa', 'qa', 'Android'),
    ('com.westernunion.moneytransferr3app.sa', 'sa', 'Android'),
    ('com.westernunion.moneytransferr3app.eu3','se','Android'),
    ('com.westernunion.moneytransferr3app.th', 'th', 'Android'),   
    ('com.westernunion.android.mtapp', 'us', 'Android') 
]

frames = []
for app_id, country, app_name in app_details:
    try:
        frames.append(load_android_data(app_id, country, app_name))
    except KeyError:
        frames.append(pd.DataFrame())

finaldfandroid = pd.concat(frames)



#ios reviews section
@st.cache_data(persist=True)
def fetch_and_process_ios_reviews(country, app_name, app_id, how_many=200):
    """Fetch and process iOS reviews for a given country, app name, and app ID."""
    try:
        app_store = AppStore(country=country, app_name=app_name, app_id=app_id)
        app_store.review(how_many=how_many)
        df_ios = pd.DataFrame(np.array(app_store.reviews), columns=['review'])
        df = df_ios.join(pd.DataFrame(df_ios.pop('review').tolist()))
        
        # Drop unnecessary columns
        columns_to_drop = ['isEdited', 'title']
        df = df.drop(columns_to_drop, axis=1)
        
        # Add columns
        df['AppName'] = 'iOS'
        df['Country'] = country.lower()
        # df['appVersion'] = ''
        df['translated_text'] = df['review'].apply(lambda x: translator.translate(x, dest='English').text) 
        # try:
        #     df['WU_Response']=df['WU_Response'].apply(lambda x: x['body'])
        # except KeyError:
        #     st.warning("Exception occured while transalation") 
        # Rename columns
        df.rename(columns={'date': 'TimeStamp', 'userName': 'UserName', 'content': 'Review', 'score': 'Rating', 'developerResponse': 'WU_Response'}, inplace=True)
      
        return df
    except KeyError:
        return pd.DataFrame()


@st.cache_data(persist=True)
def load_reviews_for_countries_ios(app_data):
    """Load iOS reviews data for multiple countries."""
    all_dfs = []
    
    for country, app_name, app_id in app_data:
        df = fetch_and_process_ios_reviews(country, app_name, app_id)
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

# Define app data for different countries
app_data = [

    ('ae', 'western-union-send-money', '1171330611'),      # UAE
    ('au', 'western-union-money-transfers', '1122288720'), #AU
    ('at', 'western-union-money-transfer', '1045347175'),   #Austria 
    ('bh', 'western-union-send-money', '1314010624'),  # Bahrain
    ('be', 'western-union-money-transfer', '1110240507'),   #Belgium
    ('ca', 'western-union-send-money', '1110191056'),  #canada
    ('ch', 'western-union-send-cash-abroad', '1110240507'),   #Switzerland
    ('cl', 'western-union-envío-de-dinero', '1304223498'), #chile
    ('de', 'western-union-money-transfer', '1045347175'),   #Germany  
    ('dk', 'western-union-send-money-se', '1152860407'),   #Denmark  
    # ('be','western-union-send-cash-abroad',  '1110240507'), #France
    ('fr', 'western-union-money-transfer', '1045347175'), #France 
    ('gb', 'western-union-money-transfer', '1045347175'),   #UK
    ('ie', 'western-union-money-transfer', '1045347175'),   #Ireland
    ('it', 'western-union-invio-denaro', '1045347175'),   #Italy    
    ('jo', 'western-union-send-money', '1459023219'),   #Jordan   
    ('kw', 'western-union-send-money', '1173794098'), #kuwait
    ('mv', 'western-union-send-money', '1483742169'), #Maldives
    ('no', 'western-union-send-money-se', '1152860407'),   #Norway   
    ('nl', 'western-union-send-money-nl', '1199782520'),   #Netherland  
    ('pt', 'western-union-enviar-fundos', '1229307854'),   #Portugal    
    ('sa', 'western-union-send-money', '1459024696'), #saudi arabia     
    ('se', 'western-union-send-money-se', '1152860407'),   #Sweden  
    ('th', 'western-union-send-money', '1459226729'), #thailand
    ('qa', 'western-union-send-money', '1173792939'), #qatar       
    ('us', 'western-union-send-money-now', '424716908')  # USA
    
]

# Load iOS reviews data for multiple countries
finaldfios = load_reviews_for_countries_ios(app_data)

frames = [finaldfandroid,finaldfios]

finaldf = pd.concat(frames)


# def clean_json(x):
#     "Create apply function for decoding JSON"
#     return json.loads(x)

# json_cols = ['WU_Response']

# # Apply the function column wise to each column of interest
# for x in json_cols:
#     try:
#      finaldf[x] = finaldf[x].apply(clean_json)
#     except:
#      print("An exception occurred")

try: 
 finaldf['WU_Response'] = finaldf.apply(lambda x: json.loads(x['WU_Response'])['body'], axis = 1)
 print(finaldf['WU_Response'])
except:
 print('Some exception occured')  

#st.write(finaldf.head())
#st.write("Columns in finaldf:", finaldf.columns)
finaldf.columns = finaldf.columns.str.strip("'")
finaldf.sort_values(['TimeStamp', 'review','rating','translated_text'])
finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]
col1, col2 = st.columns((2))
finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"])
#finaldf['Date'] = finaldf['TimeStamp'].dt.date

# Getting the min and max date 
startDate = pd.to_datetime(finaldf["TimeStamp"]).min()
endDate = pd.to_datetime(finaldf["TimeStamp"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))


try: 
 df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)].copy()
except KeyError:
 df = pd.DataFrame()

st.sidebar.header("Choose your filter: ")


# df1=df.copy()

country = st.sidebar.multiselect("Select the Country", df["Country"].unique())
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


def getsentiment(Rating):
    if Rating < 3:
        return 'negative'
    elif Rating == 4 or Rating == 5:
        return 'positive'
    else:
        return 'neutral'
try: 
 filtered_df['Sentiment'] = filtered_df['rating'].apply(getsentiment)
except:
 st.warning("Please select the correct Date range ")   

if filtered_df.empty:
 st.warning("No records found within the specified date range")
else:
 filtered_df = filtered_df.reindex(['TimeStamp', 'review', 'WU_Response','rating','Sentiment','translated_text','UserName','AppName','Country','appVersion'], axis=1)
 
#  print(df)
 st.write(filtered_df)
 csv = filtered_df.to_csv(index = False).encode('utf-8')
 st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")  


st.sidebar.markdown("### Data Visualization")
select = st.sidebar.selectbox('Type of Visualization', ['Bar plot', 'Pie chart'], key='1')
try:
    sentiment_count = filtered_df['Sentiment'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Remarks':sentiment_count.values})
except:
 st.warning("Please select the correct Date range ")  


source_text = st.text_area("Enter text to translate:")
target_language = st.selectbox("Select target language:", languages)
translate = st.button('Translate')
if translate:
    translator = Translator()
    out = translator.translate(source_text,dest=target_language)
    st.write(out.text)



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
if not st.sidebar.checkbox("Uncheck to see Visualization", True): #by defualt hide the checkbar
    st.markdown("### Customer Sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Remarks', color='Remarks', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Remarks', names='Sentiment')
        st.plotly_chart(fig)
    
    figNew = plt.figure(figsize=(20, 5))    
    axNew = sns.barplot(x='AppName', y='rating', hue='AppName', data=filtered_df, errwidth=0)
    axNew.set(xlabel='App Type', ylabel='Ratings', title='App Type vs Ratings')
    for i in axNew.containers:
        axNew.bar_label(i,)
    st.pyplot(figNew)   
    
    fig = plt.figure(figsize=(15, 5)) 
    ax = sns.countplot(x='rating',hue='rating', orient='h',dodge=False,data=filtered_df,palette='turbo')
    ax.set(xlabel='Rating', ylabel='Ratings Count', title='Ratings vs Count')
    for label in ax.containers:
       ax.bar_label(label)
    st.pyplot(fig)

    # Save the plot to a PDF buffer
    # pdf_buffer = BytesIO()
    # plt.savefig(pdf_buffer, format='pdf')
    # pdf_buffer.seek(0)

# ReportLab
   

    # figN = plt.figure(figsize=(20, 5))    
    
    # axN = sns.barplot(x='TimeStamp', y=filtered_df['rating']==1, data=filtered_df, errwidth=0)
    # for i in axN.containers:
    #     # axN.bar_label(i,)
    #     st.pyplot(figN)   
    
  
    
    # rating_more_than_mean=(filtered_df[filtered_df['rating'] > filtered_df['rating'].mean()])
    # sort_more_than_mean=rating_more_than_mean.sort_values('rating',ascending=False)
    # figNewest,axNewest = plt.subplots(figsize=(15,15))
    # figNewest.tight_layout(pad=5)
    # plot_bar(2,sort_more_than_mean)
    # plt.show()
    # st.pyplot(figNewest)
    # plot_bar(1,rating_more_than_mean)
    figNewer = plt.figure(figsize=(15, 5)) 
    axar=sns.barplot(x='Country',y='rating',hue='Country',data=filtered_df,palette='Pastel1')
    axar.set(xlabel='Country', ylabel='Ratings', title='Country vs Ratings')
    for labelN in ax.containers:
        axar.bar_label(label)
    st.pyplot(figNewer)  

    # chart_data = pd.DataFrame(
    # {
    #     "col1": filtered_df["TimeStamp"],Data Visualization
    #     "col2": filtered_df["rating"]==5,
       
    # }
    # )

    # st.line_chart(chart_data, x="col1", y="col2")
    
    # figo = plt.figure(figsize=(15, 5)) 
    # axo=sns.barplot(x='Country',y='rating',hue='Country',data=filtered_df,palette='Pastel1')
    # axo.set(xlabel='Country', ylabel='Ratings', title='Country vs Ratings')
    # for labelo in axo.containers:
    #     # axo.bar_label(labelo)
    #     df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)
    #     df.groupby([df['TimeStamp'].dt.month_name()], sort=False).plot(kind='bar')
    # st.pyplot(figo)  

    filtered_df["TimeStamp"] = pd.to_datetime(filtered_df["TimeStamp"])
    # Sort the DF from oldest to most recent recordings
    filtered_df.sort_values(by="TimeStamp", inplace=True)
    # Use the column of dates as the DF's index
    filtered_df.set_index(["TimeStamp"], inplace=True)

    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    # Create a column that has the year of each date recording
    filtered_df["year"] = filtered_df.index.year
    # Create a column that has the month (1-12) of each date recording
    filtered_df["month"] = filtered_df.index.month
    # Map the month integers to their proper names
    filtered_df["month"] = filtered_df["month"].apply(
        lambda data: months[data-1]
    )
    # Make this a categorical column so it can be sorted by the order of values\
    # in the `months` list, i.e., the proper month order
    filtered_df["month"] = pd.Categorical(filtered_df["month"], categories=months)

    # Pivot the DF so that there's a column for each month, each row\
    # represents a year, and the cells have the mean page views for the\
    # respective year and month
    df_pivot = pd.pivot_table(
        filtered_df,
        values="rating",
        index="year",
        columns="month",
        aggfunc=np.mean
    )

    # Plot a bar chart using the DF
    axo = df_pivot.plot(kind="bar")
    # Get a Matplotlib figure from the axes object for formatting purposes
    figo = axo.get_figure()
    # Change the plot dimensions (width, height)
    figo.set_size_inches(25, 10)
    # Change the axes labels
    axo.set_xlabel("Years")
    axo.set_ylabel("Average App Ratings")

    # Use this to show the plot in a new window
    # plt.show()
    # Export the plot as a PNG file
    # figo.savefig("page_views_barplot.png")
    
    # axo=sns.barplot(x='Country',y='rating',hue='Country',data=filtered_df,palette='Pastel1')
    axo.set(xlabel='Year', ylabel='Ratings', title='Year on Year Ratings')
    st.pyplot(figo)


    # def create_download_link(val, filename):
    #  b64 = base64.b64encode(val)  # val looks like b'...'
    #  return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
    
    # figs = []

    # for col in df.columns:
    #     # fig, ax = plt.subplots()
    #     # ax.plot(filtered_df[col])
    #     # st.pyplot(fig)
    #     figs.append(figNew)
    #     figs.append(figNewer)
    #     figs.append(figo)

    # export_as_pdf = st.button("Export Report")

    # if export_as_pdf:
    #     pdf = FPDF()
    #     for fig in figs:
    #         pdf.add_page()
    #         with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
    #                 fig.savefig(tmpfile.name)
    #                 pdf.image(tmpfile.name, 10, 10, 200, 100)
    #     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
    #     st.markdown(html, unsafe_allow_html=True)



st.sidebar.markdown("### Hierarchical view - TreeMap")
if not st.sidebar.checkbox("Uncheck to see TreeMap", True , key='100'): #by defualt hide the checkbar
# st.subheader("Hierarchical view using TreeMap")
    # fig3 = px.treemap(df, path = ["Network","Connected_Profile","Received_From_(Network_Name)"], values = "Review_Rating",hover_data = ["Review_Rating"],
    #                 color = "Received_From_(Network_Name)")
    fig3 = px.treemap(filtered_df, path = ["Country","AppName","review"], values = "rating",hover_data = ["rating"],
               color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']),      color = "review")
    # ,"Connected_Profile"
    # fig3.update_layout(width = 800, height = 650)
    st.plotly_chart(fig3, use_container_width=True)


st.sidebar.markdown("### Scatter Plot")
if not st.sidebar.checkbox("Uncheck to see Scatter Plot", True , key='10'): #by defualt hide the checkbar
    st.markdown("### ScatterPlot")
    fig = plt.figure(figsize=(15, 4))
    if filtered_df.empty:
      st.warning("No records found within the specified date range", icon="⚠️")
    else:
      try:
        x=pd.to_datetime(filtered_df['TimeStamp'])
        sns.scatterplot(x="TimeStamp", y="rating", data=filtered_df,hue="Country",style="Country")
        plt.xticks(rotation=90)
        # plt.yticks(rotation=90)
        st.pyplot(fig)
      except:
         st.info('')

def remove_emojis(text):
    # This function removes emojis from the input text
    return text.encode('ascii', 'ignore').decode('ascii')





words=filtered_df['review'].dropna().apply(nltk.word_tokenize)

st.sidebar.header("Customer Reviews Word Cloud")
word_sentiment = st.sidebar.radio('Display Word Cloud for which sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Uncheck to see Word Cloud", True, key='3'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = filtered_df[filtered_df['Sentiment']==word_sentiment]
    words = ' '.join(df['review'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    processed_words=remove_emojis(processed_words)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',max_words=40, width=600, height=450).generate(processed_words)
    plt.imshow(wordcloud,interpolation = "bilinear")
    plt.xticks([])
    plt.yticks([])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    # st.write(filtered_df[filtered_df['Sentiment']==word_sentiment])


    
