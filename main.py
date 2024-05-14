import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from io import BytesIO
import warnings
import base64
# from fpdf import FPDF
# import tempfile
# from tempfile import NamedTemporaryFile
import nltk
import random
import plotly.express as px
import re
import os
from googletrans import Translator
from languages import *
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



# wu_mask = np.array(Image.open('wul.png'))
# dir = os.path.dirname(__file__)
# filename = os.path.join(dir, 'Images/wu.png')
image = Image.open('wu.png')
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image, caption='',use_column_width=True)


au_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.au',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='au', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_AU(): 
    dfAndroidAU = pd.DataFrame(np.array(au_reviews),columns=['review'])
    dfAndroidAU = dfAndroidAU.join(pd.DataFrame(dfAndroidAU.pop('review').tolist()))
    dfAndroidAU=dfAndroidAU.drop(['reviewId'], axis=1)
    dfAndroidAU=dfAndroidAU.drop(['thumbsUpCount'], axis=1)
    dfAndroidAU=dfAndroidAU.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidAU=dfAndroidAU.drop(['repliedAt'], axis=1)
    # dfAndroidAU=dfAndroidAU.drop(['appVersion'], axis=1)
    dfAndroidAU['AppName']='Android'
    dfAndroidAU['Country']='Australia'
    dfAndroidAU.rename(columns = {'content':'review'}, inplace = True)
    dfAndroidAU.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidAU.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidAU.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidAU.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidAU=dfAndroidAU.drop(['userImage'], axis=1)
    return dfAndroidAU

try:
 AndroidAU=loadAndroiddata_AU() 
except KeyError:
 AndroidAU = pd.DataFrame()

@st.cache_data(persist=True)
def loadiOSdata_AU():
    wu_au = AppStore(country='au', app_name='western-union-money-transfers', app_id = '1122288720')
    wu_au.review(how_many=200)
    dfiOS = pd.DataFrame(np.array(wu_au.reviews),columns=['review'])
    dfNew = dfiOS.join(pd.DataFrame(dfiOS.pop('review').tolist()))
    dfNew=dfNew.drop(['developerResponse'], axis=1)
    dfNew=dfNew.drop(['isEdited'], axis=1)
    dfNew=dfNew.drop(['title'], axis=1)
    dfNew['AppName']='iOS'
    dfNew['Country']='Australia'
    dfNew['appVersion'] = dfNew.apply(lambda _: '', axis=1)
    dfNew.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfNew.rename(columns = {'userName':'UserName'}, inplace = True)
    dfNew.rename(columns = {'content':'Review'}, inplace = True)
    dfNew.rename(columns = {'score':'Rating'}, inplace = True) 
    dfNew.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfNew

try:
 iOSAU=loadiOSdata_AU()
except KeyError:
 iOSAU = pd.DataFrame()


bh_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.bh',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='bh', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_BH(): 
    dfAndroidBH = pd.DataFrame(np.array(bh_reviews),columns=['review'])
    dfAndroidBH = dfAndroidBH.join(pd.DataFrame(dfAndroidBH.pop('review').tolist()))
    dfAndroidBH=dfAndroidBH.drop(['reviewId'], axis=1)
    dfAndroidBH=dfAndroidBH.drop(['thumbsUpCount'], axis=1)
    dfAndroidBH=dfAndroidBH.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidBH=dfAndroidBH.drop(['repliedAt'], axis=1)
    # dfAndroidBH=dfAndroidBH.drop(['appVersion'], axis=1)
    dfAndroidBH['AppName']='Android'
    dfAndroidBH['Country']='Bahrain'
    dfAndroidBH.rename(columns = {'content':'review'}, inplace = True)
    dfAndroidBH['translated_text'] = dfAndroidBH['review'].apply(lambda x: translator.translate(x, dest='English').text)
    dfAndroidBH.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidBH.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidBH.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidBH.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidBH=dfAndroidBH.drop(['userImage'], axis=1)
    return dfAndroidBH

try:
 AndroidBH=loadAndroiddata_BH() 
except KeyError:
 AndroidBH = pd.DataFrame()

@st.cache_data(persist=True)
def loadiOSdata_BH():
    wu_bh = AppStore(country='bh', app_name='western-union-send-money', app_id = '1314010624')
    wu_bh.review(how_many=200)
    dfiOSBH = pd.DataFrame(np.array(wu_bh.reviews),columns=['review'])
    dfBH = dfiOSBH.join(pd.DataFrame(dfiOSBH.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfBH=dfBH.drop(['isEdited'], axis=1)
    dfBH=dfBH.drop(['title'], axis=1)
    dfBH['AppName']='iOS'
    dfBH['Country']='Bahrain'
    dfBH['appVersion'] = dfBH.apply(lambda _: '', axis=1)
    dfBH['appVersion'] = dfBH.apply(lambda _: '', axis=1) 
    dfBH.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfBH.rename(columns = {'userName':'UserName'}, inplace = True)
    dfBH.rename(columns = {'content':'Review'}, inplace = True)
    dfBH['translated_text'] = dfBH['review'].apply(lambda x: translator.translate(x, dest='English').text) 
    dfBH.rename(columns = {'score':'Rating'}, inplace = True)  
    dfBH.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfBH


try:
 iOSBH=loadiOSdata_BH()
except KeyError:
 iOSBH = pd.DataFrame()


ca_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.ca',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='ca', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_CA(): 
    dfAndroidCA = pd.DataFrame(np.array(ca_reviews),columns=['review'])
    dfAndroidCA = dfAndroidCA.join(pd.DataFrame(dfAndroidCA.pop('review').tolist()))
    dfAndroidCA=dfAndroidCA.drop(['reviewId'], axis=1)
    dfAndroidCA=dfAndroidCA.drop(['thumbsUpCount'], axis=1)
    dfAndroidCA=dfAndroidCA.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroidCA=dfAndroidCA.drop(['replyContent'], axis=1)
    dfAndroidCA=dfAndroidCA.drop(['repliedAt'], axis=1)
    # dfAndroidCA=dfAndroidCA.drop(['appVersion'], axis=1)
    dfAndroidCA['AppName']='Android'
    dfAndroidCA['Country']='Canada'
    dfAndroidCA.rename(columns = {'content':'review'}, inplace = True)
    # dfAndroidCA['translated_text'] = dfAndroidCA['review'].apply(lambda x: translator.translate(x, dest='English').text) 
    dfAndroidCA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidCA.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidCA.rename(columns = {'at':'TimeStamp'}, inplace = True) 
    dfAndroidCA=dfAndroidCA.drop(['userImage'], axis=1)
    dfAndroidCA.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    return dfAndroidCA

try: 
 AndroidCA=loadAndroiddata_CA() 
except KeyError:
 AndroidCA = pd.DataFrame()


@st.cache_data(persist=True)
def loadiOSdata_CA():
    wu_ca = AppStore(country='ca', app_name='western-union-send-money', app_id = '1110191056')
    wu_ca.review(how_many=200)
    dfiOSCA = pd.DataFrame(np.array(wu_ca.reviews),columns=['review'])
    dfCA = dfiOSCA.join(pd.DataFrame(dfiOSCA.pop('review').tolist()))
    dfCA=dfCA.drop(['developerResponse'], axis=1)
    dfCA=dfCA.drop(['isEdited'], axis=1)
    dfCA=dfCA.drop(['title'], axis=1)
    dfCA['AppName']='iOS'
    dfCA['Country']='Canada'
    dfCA['appVersion'] = dfCA.apply(lambda _: '', axis=1) 
    dfCA.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfCA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfCA.rename(columns = {'content':'Review'}, inplace = True)
    #dfCA['translated_text'] = dfCA['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfCA.rename(columns = {'score':'Rating'}, inplace = True)  
    dfCA.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfCA

try: 
 iOSCA=loadiOSdata_CA()
except KeyError:
 iOSCA = pd.DataFrame()


@st.cache_data(persist=True)
def loadiOSdata_CL():
    wu_cl = AppStore(country='cl', app_name='western-union-envío-de-dinero', app_id = '1304223498')
    wu_cl.review(how_many=200)
    dfiOSCL = pd.DataFrame(np.array(wu_cl.reviews),columns=['review'])
    dfNewCL = dfiOSCL.join(pd.DataFrame(dfiOSCL.pop('review').tolist()))
    # dfNewCL=dfNewCL.drop(['developerResponse'], axis=1)
    dfNewCL=dfNewCL.drop(['isEdited'], axis=1)
    dfNewCL=dfNewCL.drop(['title'], axis=1)
    dfNewCL['AppName']='iOS'
    dfNewCL['Country']='Chile'
    dfNewCL['appVersion'] = dfNewCL.apply(lambda _: '', axis=1)
    dfNewCL.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfNewCL.rename(columns = {'userName':'UserName'}, inplace = True)
    dfNewCL.rename(columns = {'content':'Review'}, inplace = True)
    dfNewCL['translated_text'] = dfNewCL['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfNewCL.rename(columns = {'score':'Rating'}, inplace = True)  
    dfNewCL.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfNewCL

    
    try:
        dfNewCL['developerResponse'].apply(lambda x: x['body'])
    except KeyError:
        dfNewCL = pd.DataFrame()
    dfNewCL.rename(columns = {'developerResponse':'WU_Response'}, inplace = True) 
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfNewCL

try:
 iOSCL=loadiOSdata_CL()
except KeyError:
 iOSCL = pd.DataFrame()



kw_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.kw',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='kw', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_KW(): 
    dfAndroidKW = pd.DataFrame(np.array(kw_reviews),columns=['review'])
    dfAndroidKW = dfAndroidKW.join(pd.DataFrame(dfAndroidKW.pop('review').tolist()))
    dfAndroidKW=dfAndroidKW.drop(['reviewId'], axis=1)
    dfAndroidKW=dfAndroidKW.drop(['thumbsUpCount'], axis=1)
    dfAndroidKW=dfAndroidKW.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidKW=dfAndroidKW.drop(['repliedAt'], axis=1)
    # dfAndroidKW=dfAndroidKW.drop(['appVersion'], axis=1)
    dfAndroidKW['AppName']='Android'
    dfAndroidKW['Country']='Kuwait'
    dfAndroidKW.rename(columns = {'content':'review'}, inplace = True)
    dfAndroidKW['translated_text'] = dfAndroidKW['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfAndroidKW.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidKW.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidKW.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidKW.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidKW=dfAndroidKW.drop(['userImage'], axis=1)
    return dfAndroidKW

try:
 AndroidKW=loadAndroiddata_KW() 
except KeyError:
 AndroidKW = pd.DataFrame()


@st.cache_data(persist=True)
def loadiOSdata_KW():
    wu_kw = AppStore(country='kw', app_name='western-union-send-money', app_id = '1173794098')
    wu_kw.review(how_many=200)
    dfiOSKW = pd.DataFrame(np.array(wu_kw.reviews),columns=['review'])
    dfKW = dfiOSKW.join(pd.DataFrame(dfiOSKW.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfKW=dfKW.drop(['isEdited'], axis=1)
    dfKW=dfKW.drop(['title'], axis=1)
    dfKW['AppName']='iOS'
    dfKW['Country']='Kuwait'
    dfKW['appVersion'] = dfKW.apply(lambda _: '', axis=1) 
    dfKW.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfKW.rename(columns = {'userName':'UserName'}, inplace = True)
    dfKW.rename(columns = {'content':'Review'}, inplace = True)
    dfKW['translated_text'] = dfKW['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfKW.rename(columns = {'score':'Rating'}, inplace = True)  
    dfKW.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfKW

try: 
 iOSKW=loadiOSdata_KW()
except KeyError:
 iOSKW = pd.DataFrame()

# mx_reviews = reviews_all(
#     'com.westernunion.moneytransferr3app.mcc2',
#     sleep_milliseconds=0, # defaults to 0
#     lang='mx', # defaults to 'en'
#     country='mx', # defaults to 'us'
#     sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
# )
# @st.cache_data(persist=True)
# def loadAndroiddata_MX(): 
#     dfAndroidMX = pd.DataFrame(np.array(mx_reviews),columns=['review'])
#     dfAndroidMX = dfAndroidMX.join(pd.DataFrame(dfAndroidMX.pop('review').tolist()))
#     dfAndroidMX=dfAndroidMX.drop(['reviewId'], axis=1)
#     dfAndroidMX=dfAndroidMX.drop(['thumbsUpCount'], axis=1)
#     dfAndroidMX=dfAndroidMX.drop(['reviewCreatedVersion'], axis=1)
#     # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
#     dfAndroidMX=dfAndroidMX.drop(['repliedAt'], axis=1)
#     # dfAndroidKW=dfAndroidKW.drop(['appVersion'], axis=1)
#     dfAndroidMX['AppName']='Android'
#     dfAndroidMX['Country']='Mexico'
#     dfAndroidMX.rename(columns = {'content':'review'}, inplace = True)
#     dfAndroidMX.rename(columns = {'userName':'UserName'}, inplace = True)
#     dfAndroidMX.rename(columns = {'score':'rating'}, inplace = True)
#     dfAndroidMX.rename(columns = {'at':'TimeStamp'}, inplace = True)
#     dfAndroidMX.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
#     dfAndroidMX=dfAndroidMX.drop(['userImage'], axis=1)
#     return dfAndroidMX

# AndroidMX=loadAndroiddata_MX() 




@st.cache_data(persist=True)
def loadiOSdata_MX():
    wu_mx = AppStore(country='mx', app_name='western-union-send-money', app_id = '1146349983')
    wu_mx.review(how_many=200)
    dfiOSMX = pd.DataFrame(np.array(wu_mx.reviews),columns=['review'])
    dfNewMX = dfiOSMX.join(pd.DataFrame(dfiOSMX.pop('review').tolist()))
    # dfNewMX=dfNewMX.drop(['developerResponse'], axis=1)
    dfNewMX=dfNewMX.drop(['isEdited'], axis=1)
    dfNewMX=dfNewMX.drop(['title'], axis=1)
    dfNewMX['AppName']='iOS'
    dfNewMX['Country']='Mexico'
    dfNewMX['appVersion'] = dfNewMX.apply(lambda _: '', axis=1)
    dfNewMX.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfNewMX.rename(columns = {'userName':'UserName'}, inplace = True)
    dfNewMX.rename(columns = {'content':'Review'}, inplace = True)
    dfNewMX['translated_text'] = dfNewMX['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfNewMX.rename(columns = {'score':'Rating'}, inplace = True) 
    dfNewMX.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)   
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfNewMX

try: 
 iOSMX=loadiOSdata_MX()
except KeyError:
 iOSMX = pd.DataFrame()



nz_reviews = reviews_all(
   'com.westernunion.moneytransferr3app.nz',
    #com.westernunion.moneytransferr3app.nz&hl=en_ZA
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='nz', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
) 

@st.cache_data(persist=True)
def loadAndroiddata_NZ():
    dfAndroidNZ = pd.DataFrame(np.array(nz_reviews),columns=['review'])
    dfAndroidNZ = dfAndroidNZ.join(pd.DataFrame(dfAndroidNZ.pop('review').tolist()))
    dfAndroidNZ=dfAndroidNZ.drop(['reviewId'], axis=1)
    dfAndroidNZ=dfAndroidNZ.drop(['thumbsUpCount'], axis=1)
    dfAndroidNZ=dfAndroidNZ.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroidNZ=dfAndroidNZ.drop(['replyContent'], axis=1)
    dfAndroidNZ=dfAndroidNZ.drop(['repliedAt'], axis=1)
    # dfAndroidNZ=dfAndroidNZ.drop(['appVersion'], axis=1)
    dfAndroidNZ['AppName']='Android'
    dfAndroidNZ['Country']='New Zealand'
    dfAndroidNZ.rename(columns = {'content':'review'}, inplace = True)
    #dfAndroidNZ['translated_text'] = dfAndroidNZ['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfAndroidNZ.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidNZ.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidNZ.rename(columns = {'at':'TimeStamp'}, inplace = True) 
    dfAndroidNZ.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidNZ=dfAndroidNZ.drop(['userImage'], axis=1)
    return dfAndroidNZ

try:
 AndroidNZ=loadAndroiddata_NZ()
except KeyError:
 AndroidNZ = pd.DataFrame()


@st.cache_data(persist=True)
def loadiOSdata_NZ():
    wu_nz = AppStore(country='nz', app_name='western-union-remit-money', app_id = '1226778839')
    wu_nz.review(how_many=200)
    dfiOSNZ = pd.DataFrame(np.array(wu_nz.reviews),columns=['review'])
    dfNZ = dfiOSNZ.join(pd.DataFrame(dfiOSNZ.pop('review').tolist()))
    # dfNZ=dfNZ.drop(['developerResponse'], axis=1)
    dfNZ=dfNZ.drop(['isEdited'], axis=1)
    dfNZ=dfNZ.drop(['title'], axis=1)
    dfNZ['AppName']='iOS'
    dfNZ['Country']='New Zealand'
    dfNZ['appVersion'] = dfNZ.apply(lambda _: '', axis=1)
    dfNZ.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfNZ.rename(columns = {'userName':'UserName'}, inplace = True)
    #dfNZ['translated_text'] = dfNZ['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfNZ.rename(columns = {'content':'Review'}, inplace = True)
    dfNZ.rename(columns = {'score':'Rating'}, inplace = True)  
    dfNZ.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfNZ

try:
 iOSNZ=loadiOSdata_NZ()
except KeyError:
 iOSNZ = pd.DataFrame()





qa_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.qa',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='qa', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_QA(): 
    dfAndroidQA = pd.DataFrame(np.array(qa_reviews),columns=['review'])
    dfAndroidQA = dfAndroidQA.join(pd.DataFrame(dfAndroidQA.pop('review').tolist()))
    dfAndroidQA=dfAndroidQA.drop(['reviewId'], axis=1)
    dfAndroidQA=dfAndroidQA.drop(['thumbsUpCount'], axis=1)
    dfAndroidQA=dfAndroidQA.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidQA=dfAndroidQA.drop(['repliedAt'], axis=1)
    # dfAndroidQA=dfAndroidQA.drop(['appVersion'], axis=1)
    dfAndroidQA['AppName']='Android'
    dfAndroidQA['Country']='Qatar'
    dfAndroidQA.rename(columns = {'content':'review'}, inplace = True)
    dfAndroidQA['translated_text'] = dfAndroidQA['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfAndroidQA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidQA.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidQA.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidQA.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidQA=dfAndroidQA.drop(['userImage'], axis=1)
    return dfAndroidQA

try:
 AndroidQA=loadAndroiddata_QA() 
except KeyError:
 AndroidQA = pd.DataFrame()



@st.cache_data(persist=True)
def loadiOSdata_QA():
    wu_qa = AppStore(country='qa', app_name='western-union-send-money', app_id = '1173792939')
    wu_qa.review(how_many=100)
    dfiOSQA = pd.DataFrame(np.array(wu_qa.reviews),columns=['review'])
    dfQA = dfiOSQA.join(pd.DataFrame(dfiOSQA.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfQA=dfQA.drop(['isEdited'], axis=1)
    dfQA=dfQA.drop(['title'], axis=1)
    dfQA['AppName']='iOS'
    dfQA['Country']='Qatar'
    dfQA['appVersion'] = dfQA.apply(lambda _: '', axis=1) 
    dfQA.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfQA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfQA.rename(columns = {'content':'Review'}, inplace = True)
    dfQA['translated_text'] = dfQA['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfQA.rename(columns = {'score':'Rating'}, inplace = True)  
    dfQA.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfQA

try: 
 iOSQA=loadiOSdata_QA()
except KeyError:
 iOSQA = pd.DataFrame()



sa_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.sa',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='sa', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_SA(): 
    dfAndroidSA = pd.DataFrame(np.array(sa_reviews),columns=['review'])
    dfAndroidSA = dfAndroidSA.join(pd.DataFrame(dfAndroidSA.pop('review').tolist()))
    dfAndroidSA=dfAndroidSA.drop(['reviewId'], axis=1)
    dfAndroidSA=dfAndroidSA.drop(['thumbsUpCount'], axis=1)
    dfAndroidSA=dfAndroidSA.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidSA=dfAndroidSA.drop(['repliedAt'], axis=1)
    # dfAndroidSA=dfAndroidSA.drop(['appVersion'], axis=1)
    dfAndroidSA['AppName']='Android'
    dfAndroidSA['Country']='Saudi Arabia'
    dfAndroidSA.rename(columns = {'content':'review'}, inplace = True)
    dfAndroidSA['translated_text'] = dfAndroidSA['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfAndroidSA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidSA.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidSA.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidSA.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidSA=dfAndroidSA.drop(['userImage'], axis=1)
    return dfAndroidSA

try:
 AndroidSA=loadAndroiddata_SA() 
except KeyError:
 AndroidSA = pd.DataFrame()


@st.cache_data(persist=True)
def loadiOSdata_SA():
    wu_sa = AppStore(country='sa', app_name='western-union-send-money', app_id = '1459024696')
    wu_sa.review(how_many=200)
    dfiOSSA = pd.DataFrame(np.array(wu_sa.reviews),columns=['review'])
    dfSA = dfiOSSA.join(pd.DataFrame(dfiOSSA.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfSA=dfSA.drop(['isEdited'], axis=1)
    dfSA=dfSA.drop(['title'], axis=1)
    dfSA['AppName']='iOS'
    dfSA['Country']='Saudi Arabia'
    dfSA['appVersion'] = dfSA.apply(lambda _: '', axis=1)
    dfSA.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfSA.rename(columns = {'userName':'UserName'}, inplace = True)
    dfSA.rename(columns = {'content':'Review'}, inplace = True)
    dfSA['translated_text'] = dfSA['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfSA.rename(columns = {'score':'Rating'}, inplace = True)  
    dfSA.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfSA

try:
 iOSSA=loadiOSdata_SA()
except KeyError:
 iOSSA = pd.DataFrame()


th_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.th',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='th', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
th_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.th',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='th', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_TH(): 
    dfAndroidTH = pd.DataFrame(np.array(th_reviews),columns=['review'])
    dfAndroidTH = dfAndroidTH.join(pd.DataFrame(dfAndroidTH.pop('review').tolist()))
    dfAndroidTH=dfAndroidTH.drop(['reviewId'], axis=1)
    dfAndroidTH=dfAndroidTH.drop(['thumbsUpCount'], axis=1)
    dfAndroidTH=dfAndroidTH.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidTH=dfAndroidTH.drop(['repliedAt'], axis=1)
    # dfAndroidTH=dfAndroidTH.drop(['appVersion'], axis=1)
    dfAndroidTH['AppName']='Android'
    dfAndroidTH['Country']='Thailand'
    dfAndroidTH.rename(columns = {'content':'review'}, inplace = True)
    #dfAndroidTH['translated_text'] = dfAndroidTH['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfAndroidTH.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidTH.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidTH.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidTH.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidTH=dfAndroidTH.drop(['userImage'], axis=1)
    return dfAndroidTH

try:
 AndroidTH=loadAndroiddata_TH() 
except KeyError:
 AndroidTH = pd.DataFrame()



@st.cache_data(persist=True)
def loadiOSdata_TH():
    wu_th = AppStore(country='th', app_name='western-union-send-money', app_id = '1459226729')
    wu_th.review(how_many=200)
    dfiOSTH = pd.DataFrame(np.array(wu_th.reviews),columns=['review'])
    dfTH = dfiOSTH.join(pd.DataFrame(dfiOSTH.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfTH=dfTH.drop(['isEdited'], axis=1)
    dfTH=dfTH.drop(['title'], axis=1)
    dfTH['AppName']='iOS'
    dfTH['Country']='Thailand'
    dfTH['appVersion'] = dfTH.apply(lambda _: '', axis=1)
    dfTH.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfTH.rename(columns = {'userName':'UserName'}, inplace = True)
    dfTH.rename(columns = {'content':'Review'}, inplace = True)
    #dfTH['translated_text'] = dfTH['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfTH.rename(columns = {'score':'Rating'}, inplace = True)  
    dfTH.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfTH

try:
 iOSTH=loadiOSdata_TH()
except KeyError:
 iOSTH = pd.DataFrame()


ae_reviews = reviews_all(
    'com.westernunion.moneytransferr3app.ae',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='ae', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_AE(): 
    dfAndroidAE = pd.DataFrame(np.array(ae_reviews),columns=['review'])
    dfAndroidAE = dfAndroidAE.join(pd.DataFrame(dfAndroidAE.pop('review').tolist()))
    dfAndroidAE=dfAndroidAE.drop(['reviewId'], axis=1)
    dfAndroidAE=dfAndroidAE.drop(['thumbsUpCount'], axis=1)
    dfAndroidAE=dfAndroidAE.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidAE=dfAndroidAE.drop(['repliedAt'], axis=1)
    # dfAndroidAE=dfAndroidAE.drop(['appVersion'], axis=1)
    dfAndroidAE['AppName']='Android'
    dfAndroidAE['Country']='UAE'
    dfAndroidAE.rename(columns = {'content':'review'}, inplace = True)
    #dfAndroidAE['translated_text'] = dfAndroidAE['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfAndroidAE.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidAE.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidAE.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidAE.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidAE=dfAndroidAE.drop(['userImage'], axis=1)
    return dfAndroidAE

try:
 AndroidAE=loadAndroiddata_AE() 
except KeyError:
 AndroidAE = pd.DataFrame()

@st.cache_data(persist=True)
def loadiOSdata_AE():
    wu_ae = AppStore(country='ae', app_name='western-union-send-money', app_id = '1171330611')
    wu_ae.review(how_many=200)
    dfiOSAE = pd.DataFrame(np.array(wu_ae.reviews),columns=['review'])
    dfAE = dfiOSAE.join(pd.DataFrame(dfiOSAE.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfAE=dfAE.drop(['isEdited'], axis=1)
    dfAE=dfAE.drop(['title'], axis=1)
    dfAE['AppName']='iOS'
    dfAE['Country']='UAE'
    dfAE['appVersion'] = dfAE.apply(lambda _: '', axis=1)
    dfAE.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfAE.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAE.rename(columns = {'content':'Review'}, inplace = True)
    #dfAE['translated_text'] = dfAE['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfAE.rename(columns = {'score':'Rating'}, inplace = True)  
    dfAE.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfAE

try:
 iOSAE=loadiOSdata_AE()
except KeyError:
 iOSAE = pd.DataFrame()


# mv_reviews = reviews_all(
#     'com.westernunion.moneytransferr3app.mv',
#     sleep_milliseconds=0, # defaults to 0
#     lang='en', # defaults to 'en'
#     country='mv', # defaults to 'us'
#     sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
# )
# @st.cache_data(persist=True)
# def loadAndroiddata_MV(): 
#     dfAndroidMV = pd.DataFrame(np.array(mv_reviews),columns=['review'])
#     dfAndroidMV = dfAndroidMV.join(pd.DataFrame(dfAndroidMV.pop('review').tolist()))
#     dfAndroidMV=dfAndroidMV.drop(['reviewId'], axis=1)
#     dfAndroidMV=dfAndroidMV.drop(['thumbsUpCount'], axis=1)
#     dfAndroidMV=dfAndroidMV.drop(['reviewCreatedVersion'], axis=1)
#     # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
#     dfAndroidMV=dfAndroidMV.drop(['repliedAt'], axis=1)
#     dfAndroidMV=dfAndroidMV.drop(['appVersion'], axis=1)
#     dfAndroidMV['AppName']='Android'
#     dfAndroidMV['Country']='Maldives'
#     dfAndroidMV.rename(columns = {'content':'review'}, inplace = True)
#     dfAndroidMV.rename(columns = {'userName':'UserName'}, inplace = True)
#     dfAndroidMV.rename(columns = {'score':'rating'}, inplace = True)
#     dfAndroidMV.rename(columns = {'at':'TimeStamp'}, inplace = True)
#     dfAndroidMV.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
#     dfAndroidMV=dfAndroidMV.drop(['userImage'], axis=1)
#     return dfAndroidMV

# AndroidMV=loadAndroiddata_MV() 



@st.cache_data(persist=True)
def loadiOSdata_MV():
    wu_mv = AppStore(country='mv', app_name='western-union-send-money', app_id = '1483742169')
    wu_mv.review(how_many=200)
    dfiOSMV = pd.DataFrame(np.array(wu_mv.reviews),columns=['review'])
    dfMV = dfiOSMV.join(pd.DataFrame(dfiOSMV.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfMV=dfMV.drop(['isEdited'], axis=1)
    dfMV=dfMV.drop(['title'], axis=1)
    dfMV['AppName']='iOS'
    dfMV['Country']='Maldives'
    dfMV['appVersion'] = dfMV.apply(lambda _: '', axis=1)
    dfMV.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfMV.rename(columns = {'userName':'UserName'}, inplace = True)
    dfMV.rename(columns = {'content':'Review'}, inplace = True)
    #dfMV['translated_text'] = dfMV['review'].apply(lambda x: translator.translate(x, dest='English').text)   
    dfMV.rename(columns = {'score':'Rating'}, inplace = True)  
    dfMV.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfMV

try: 
 iOSMV=loadiOSdata_MV()
except KeyError:
 iOSMV = pd.DataFrame()

# jo_reviews = reviews_all(
#     'com.westernunion.moneytransferr3app.jo',
#     sleep_milliseconds=0, # defaults to 0
#     lang='en', # defaults to 'en'
#     country='jo', # defaults to 'us'
#     sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
# )
# @st.cache_data(persist=True)
# def loadAndroiddata_JO(): 
#     dfAndroidJO = pd.DataFrame(np.array(jo_reviews),columns=['review'])
#     dfAndroidJO = dfAndroidJO.join(pd.DataFrame(dfAndroidJO.pop('review').tolist()))
#     dfAndroidJO=dfAndroidJO.drop(['reviewId'], axis=1)
#     dfAndroidJO=dfAndroidJO.drop(['thumbsUpCount'], axis=1)
#     dfAndroidJO=dfAndroidJO.drop(['reviewCreatedVersion'], axis=1)
#     # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
#     dfAndroidJO=dfAndroidJO.drop(['repliedAt'], axis=1)
#     # dfAndroidJO=dfAndroidJO.drop(['appVersion'], axis=1)
#     dfAndroidJO['AppName']='Android'
#     dfAndroidJO['Country']='Jordan'
#     dfAndroidJO.rename(columns = {'content':'review'}, inplace = True)
#     dfAndroidJO.rename(columns = {'userName':'UserName'}, inplace = True)
#     dfAndroidJO.rename(columns = {'score':'rating'}, inplace = True)
#     dfAndroidJO.rename(columns = {'at':'TimeStamp'}, inplace = True)
#     dfAndroidJO.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
#     dfAndroidJO=dfAndroidJO.drop(['userImage'], axis=1)
#     return dfAndroidJO

# AndroidJO=loadAndroiddata_JO() 


@st.cache_data(persist=True)
def loadiOSdata_JO():
    wu_jo = AppStore(country='jo', app_name='western-union-send-money', app_id = '1459023219')
    wu_jo.review(how_many=200)
    dfiOSJO = pd.DataFrame(np.array(wu_jo.reviews),columns=['review'])
    dfJO = dfiOSJO.join(pd.DataFrame(dfiOSJO.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfJO=dfJO.drop(['isEdited'], axis=1)
    dfJO=dfJO.drop(['title'], axis=1)
    dfJO['AppName']='iOS'
    dfJO['Country']='Jordan'
    dfJO['appVersion'] = dfJO.apply(lambda _: '', axis=1)
    dfJO.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfJO.rename(columns = {'userName':'UserName'}, inplace = True)
    dfJO.rename(columns = {'content':'Review'}, inplace = True)
    dfJO['translated_text'] = dfJO['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfJO.rename(columns = {'score':'Rating'}, inplace = True)  
    dfJO.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfJO

try:
 iOSJO=loadiOSdata_JO()
except KeyError:
 iOSJO = pd.DataFrame()


us_reviews = reviews_all(
    'com.westernunion.android.mtapp',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)
@st.cache_data(persist=True)
def loadAndroiddata_US(): 
    dfAndroidUS = pd.DataFrame(np.array(us_reviews),columns=['review'])
    dfAndroidUS = dfAndroidUS.join(pd.DataFrame(dfAndroidUS.pop('review').tolist()))
    dfAndroidUS=dfAndroidUS.drop(['reviewId'], axis=1)
    dfAndroidUS=dfAndroidUS.drop(['thumbsUpCount'], axis=1)
    dfAndroidUS=dfAndroidUS.drop(['reviewCreatedVersion'], axis=1)
    # dfAndroid=dfAndroid.drop(['replyContent'], axis=1)
    dfAndroidUS=dfAndroidUS.drop(['repliedAt'], axis=1)
    # dfAndroidJO=dfAndroidJO.drop(['appVersion'], axis=1)
    dfAndroidUS['AppName']='Android'
    dfAndroidUS['Country']='USA'
    dfAndroidUS.rename(columns = {'content':'review'}, inplace = True)
    #dfAndroidUS['translated_text'] = dfAndroidUS['review'].apply(lambda x: translator.translate(x, dest='English').text)
    dfAndroidUS.rename(columns = {'userName':'UserName'}, inplace = True)
    dfAndroidUS.rename(columns = {'score':'rating'}, inplace = True)
    dfAndroidUS.rename(columns = {'at':'TimeStamp'}, inplace = True)
    dfAndroidUS.rename(columns = {'replyContent':'WU_Response'}, inplace = True) 
    dfAndroidUS=dfAndroidUS.drop(['userImage'], axis=1)
    return dfAndroidUS

try:
 AndroidUS=loadAndroiddata_US() 
except KeyError:
 AndroidUS = pd.DataFrame()

@st.cache_data(persist=True)
def loadiOSdata_US():
    wu_us = AppStore(country='us', app_name='western-union-send-money-now', app_id = '424716908')
    wu_us.review(how_many=200)
    dfiOSUS = pd.DataFrame(np.array(wu_us.reviews),columns=['review'])
    dfUS = dfiOSUS.join(pd.DataFrame(dfiOSUS.pop('review').tolist()))
    # dfSA=dfSA.drop(['developerResponse'], axis=1)
    dfUS=dfUS.drop(['isEdited'], axis=1)
    dfUS=dfUS.drop(['title'], axis=1)
    dfUS['AppName']='iOS'
    dfUS['Country']='USA'
    dfUS['appVersion'] = dfUS.apply(lambda _: '', axis=1)
    dfUS.rename(columns = {'date':'TimeStamp'}, inplace = True) 
    dfUS.rename(columns = {'userName':'UserName'}, inplace = True)
    dfUS.rename(columns = {'content':'Review'}, inplace = True)
    #dfUS['translated_text'] = dfUS['review'].apply(lambda x: translator.translate(x, dest='English').text)    
    dfUS.rename(columns = {'score':'Rating'}, inplace = True)  
    dfUS.rename(columns = {'developerResponse':'WU_Response'}, inplace = True)  
    # json_column = dfUS['WU_Response']
    # extracted_data = json_column.apply(lambda x: x['body'])
    # dfUS['New_Response'] = dfUS.apply(lambda x: json.dumps(x['WU_Response'])['body'], axis = 1)
    # dfNew.iloc[:,[3,1,2,0,4]]
    return dfUS

try: 
 iOSUS=loadiOSdata_US()
except KeyError:
 iOSUS = pd.DataFrame()


# frames = [AndroidAU,iOSAU,AndroidMX]

frames = [AndroidUS,iOSUS,AndroidAU,iOSAU,AndroidBH,iOSBH,AndroidNZ,iOSNZ,AndroidCA,iOSMX,iOSCA,AndroidTH,iOSTH,AndroidSA,iOSSA,AndroidKW,iOSKW,AndroidQA,iOSQA,AndroidAE,iOSAE,iOSMV,iOSJO,iOSCL]

finaldf = pd.concat(frames)

finaldf = pd.concat(frames)
# st.write(finaldf)

# st.write(finaldf.loc[finaldf['WU_Response'].notnull()] )

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

df = finaldf[(finaldf["TimeStamp"] >= date1) & (finaldf["TimeStamp"] <= date2)].copy()

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
st.write(filtered_df)
csv = filtered_df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")  


st.sidebar.markdown("### Data Visualization")
select = st.sidebar.selectbox('Type of Visualization', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = filtered_df['Sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Remarks':sentiment_count.values})



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
    



def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)



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


    

