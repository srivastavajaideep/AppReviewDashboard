import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from io import BytesIO
import warnings
import base64
import nltk
import random
import plotly.express as px
import re
import os
from languages import *
from googletrans import Translator
from google_play_scraper import Sort, reviews_all
from app_store_scraper import AppStore
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
nltk.download('punkt')


# st.cache_data.clear()
st.set_page_config(page_title="WU App Review DashBoard!!!", page_icon=":sparkles:",layout="wide")
st.title(" :sparkles: WU App Review DashBoard",)
st.markdown('<style>div.block-container{padding-top:1rem;text-align: center}</style>',unsafe_allow_html=True)

translator = Translator()
# wu_mask = np.array(Image.open('wul.png'))
# dir = os.path.dirname(__file__)
# filename = os.path.join(dir, 'Images/wu.png')
image = Image.open('wu.png')
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image, caption='',use_column_width=True)


@st.cache_data(persist=True)
def fetch_and_process_reviews(app_id, country, app_name, sleep_milliseconds=0, lang='en', sort=Sort.NEWEST):
    """Fetch and process reviews for a given app and country."""
    reviews = reviews_all(
        app_id,
        sleep_milliseconds=sleep_milliseconds,
        lang=lang,
        country=country,
        sort=sort,
    )
    
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    
    # Drop unnecessary columns
    columns_to_drop = ['reviewId', 'thumbsUpCount', 'reviewCreatedVersion', 'repliedAt', 'userImage']
    df = df.drop(columns_to_drop, axis=1)
    
    # Add app name and country
    df['AppName'] = app_name
    df['Country'] = country
    df['WU_Response']=df['WU_Response'].apply(lambda x: x['body'])
    # df['translated_text'] = df['review'].apply(lambda x: translator.translate(x, dest='English').text)   

    # Rename columns
    df = df.rename(columns={'content': 'review', 'userName': 'UserName', 'score': 'rating', 'at': 'TimeStamp', 'replyContent': 'WU_Response'})
    
    return df

@st.cache_data(persist=True)
def load_reviews_data(app_ids_countries, sleep_milliseconds=0, lang='en', sort=Sort.NEWEST):
    """Load reviews data for multiple apps and countries."""
    all_dfs = []
    
    for app_id, country, app_name in app_ids_countries:
        try:
            df = fetch_and_process_reviews(app_id, country, app_name, sleep_milliseconds, lang, sort)
            all_dfs.append(df)
        except KeyError:
            continue
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# Define app IDs, countries, and app names
app_ids_countries = [
    ('com.westernunion.moneytransferr3app.au', 'au', 'Android'),
    ('com.westernunion.moneytransferr3app.bh', 'bh', 'Android'),
    ('com.westernunion.moneytransferr3app.ca', 'ca', 'Android'),
    ('com.westernunion.moneytransferr3app.kw','kw','Android'),
    ('com.westernunion.moneytransferr3app.mcc2','mx','Android'),
    ('com.westernunion.moneytransferr3app.nz','nz','Android'),
    ('com.westernunion.moneytransferr3app.qa','qa','Android'),
    ('com.westernunion.moneytransferr3app.sa','sa','Android'),
    ('com.westernunion.moneytransferr3app.th','th','Android'),
    ('com.westernunion.moneytransferr3app.ae','ae','Android'),
    ('com.westernunion.android.mtapp','us','Android'),
    ('com.westernunion.moneytransferr3app.eu2','be','Android'),
    ('com.westernunion.moneytransferr3app.eu','de','Android'),
    ('com.westernunion.moneytransferr3app.eu','gb','Android'),

]

# Parameters
sleep_milliseconds = 0
lang = 'en'
sort = Sort.NEWEST

# Load reviews data for multiple apps and countries
finaldfandroid = load_reviews_data(app_ids_countries, sleep_milliseconds, lang, sort)


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
        df['Country'] = country
        df['appVersion'] = ''
        df['WU_Response']=df['WU_Response'].apply(lambda x: x['body'])
        
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
    ('us', 'western-union-send-money-now', '424716908'),  # USA
    ('ae', 'western-union-send-money', '1171330611'),      # UAE
    ('au', 'western-union-money-transfers', '1122288720'), #AU
    ('bh', 'western-union-send-money', '1314010624'),  # Bahrain
    ('ca', 'western-union-send-money', '1110191056'),  #canada
    ('cl', 'western-union-envío-de-dinero', '1304223498'), #chile
    ('be','western-union-send-cash-abroad',  '1110240507'), #France
    ('kw', 'western-union-send-money', '1173794098'), #kuwait
    ('qa', 'western-union-send-money', '1173792939'), #qatar
    ('sa', 'western-union-send-money', '1459024696'), #saudi arabia
    ('th', 'western-union-send-money', '1459226729'), #thailand
    ('mv', 'western-union-send-money', '1483742169'), #Maldives
    ('jo', 'western-union-send-money', '1459023219'),   #Jordan
     ('de', 'western-union-money-transfer', '1045347175'),   #Germany
    ('gb', 'western-union-money-transfer', '1045347175'),   #UK
       
]

# Load iOS reviews data for multiple countries
finaldfios = load_reviews_for_countries_ios(app_data)

frames = [finaldfandroid,finaldfios]

finaldf = pd.concat(frames)


#     #dfNewCL['translated_text'] = dfNewCL['review'].apply(lambda x: translator.translate(x, dest='English').text)   


# # mv_reviews = reviews_all(
# #     'com.westernunion.moneytransferr3app.mv',
# #     sleep_milliseconds=0, # defaults to 0
# #     lang='en', # defaults to 'en'
# #     country='mv', # defaults to 'us'
# #     sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
# # )
# # @st.cache_data(persist=True)
# # def loadAndroiddata_MV(): 
# #     dfAndroidMV = pd.DataFrame(np.array(mv_reviews),columns=['review'])
# #     dfAndroidMV = dfAndroidMV.join(pd.DataFrame(dfAndroidMV.pop('review').tolist()))
# #     dfAndroidMV=dfAndroidMV.drop(['reviewId'], axis=1)
# #     dfAndroidMV=dfAndroidMV.drop(['thumbsUpCount'], axis=1)
# #     dfAndroidMV=dfAndroidMV.drop(['reviewCreatedVersion'], axis=1)
# #     # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
# #     dfAndroidMV=dfAndroidMV.drop(['repliedAt'], axis=1)
# #     dfAndroidMV=dfAndroidMV.drop(['appVersion'], axis=1)
# #     dfAndroidMV['AppName']='Android'
# #     dfAndroidMV['Country']='Maldives'
# #     dfAndroidMV.rename(columns = {'content':'review'}, inplace = True)
# #     dfAndroidMV.rename(columns = {'userName':'UserName'}, inplace = True)
# #     dfAndroidMV.rename(columns = {'score':'rating'}, inplace = True)
# #     dfAndroidMV.rename(columns = {'at':'TimeStamp'}, inplace = True)
# #     dfAndroidMV.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
# #     dfAndroidMV=dfAndroidMV.drop(['userImage'], axis=1)
# #     return dfAndroidMV

# # AndroidMV=loadAndroiddata_MV() 


# # jo_reviews = reviews_all(
# #     'com.westernunion.moneytransferr3app.jo',
# #     sleep_milliseconds=0, # defaults to 0
# #     lang='en', # defaults to 'en'
# #     country='jo', # defaults to 'us'
# #     sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
# # )
# # @st.cache_data(persist=True)
# # def loadAndroiddata_JO(): 
# #     dfAndroidJO = pd.DataFrame(np.array(jo_reviews),columns=['review'])
# #     dfAndroidJO = dfAndroidJO.join(pd.DataFrame(dfAndroidJO.pop('review').tolist()))
# #     dfAndroidJO=dfAndroidJO.drop(['reviewId'], axis=1)
# #     dfAndroidJO=dfAndroidJO.drop(['thumbsUpCount'], axis=1)
# #     dfAndroidJO=dfAndroidJO.drop(['reviewCreatedVersion'], axis=1)
# #     # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
# #     dfAndroidJO=dfAndroidJO.drop(['repliedAt'], axis=1)
# #     # dfAndroidJO=dfAndroidJO.drop(['appVersion'], axis=1)
# #     dfAndroidJO['AppName']='Android'
# #     dfAndroidJO['Country']='Jordan'
# #     dfAndroidJO.rename(columns = {'content':'review'}, inplace = True)
# #     dfAndroidJO.rename(columns = {'userName':'UserName'}, inplace = True)
# #     dfAndroidJO.rename(columns = {'score':'rating'}, inplace = True)
# #     dfAndroidJO.rename(columns = {'at':'TimeStamp'}, inplace = True)
# #     dfAndroidJO.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
# #     dfAndroidJO=dfAndroidJO.drop(['userImage'], axis=1)
# #     return dfAndroidJO

# # AndroidJO=loadAndroiddata_JO() 


# # frames = [AndroidAU,iOSAU,AndroidMX]

# frames = [AndroidUS,iOSUS,AndroidAU,iOSAU,AndroidBH,iOSBH,AndroidNZ,iOSNZ,AndroidCA,iOSMX,iOSCA,AndroidTH,iOSTH,AndroidSA,iOSSA,AndroidKW,iOSKW,AndroidQA,iOSQA,AndroidAE,iOSAE,iOSMV,iOSJO,iOSCL,AndroidFR,iOSFR]

# finaldf = pd.concat(frames)
# st.write(finaldf)

# st.write(finaldf.loc[finaldf['WU_Response'].notnull()] )
st.write(finaldf.head())
st.write("Columns in finaldf:", finaldf.columns)
finaldf.columns = finaldf.columns.str.strip("'")
finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]
col1, col2 = st.columns((2))
finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"])

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

filtered_df['Sentiment'] = filtered_df['rating'].apply(getsentiment)

if filtered_df.empty:
 st.warning("No records found within the specified date range")
else:
 st.write(filtered_df)
 csv = filtered_df.to_csv(index = False).encode('utf-8')
 st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")  


st.sidebar.markdown("### Data Visualization")
select = st.sidebar.selectbox('Type of Visualization', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = filtered_df['Sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Remarks':sentiment_count.values})

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
    x=pd.to_datetime(filtered_df['TimeStamp'])
    sns.scatterplot(x="TimeStamp", y="rating", data=filtered_df,hue="Country",style="Country")
    plt.xticks(rotation=90)
    # plt.yticks(rotation=90)
    st.pyplot(fig)
    

def remove_emojis(text):
    # This function removes emojis from the input text
    return text.encode('ascii', 'ignore').decode('ascii')




# def remove_emojis(data):
#     emoj = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002500-\U00002BEF"  # chinese char
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u"\U00010000-\U0010ffff"
#         u"\u2640-\u2642" 
#         u"\u2600-\u2B55"
#         u"\u200d"
#         u"\u23cf"
#         u"\u23e9"
#         u"\u231a"
#         u"\ufe0f"  # dingbats
#         u"\u3030"
#                       "]+", re.UNICODE)
#     return re.sub(emoj, '', data)



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


    

