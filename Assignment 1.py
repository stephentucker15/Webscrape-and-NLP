# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:05:11 2021

@author: steph
"""


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#now download the required conventions:
nltk.download('vader_lexicon')
nltk.download('wordnet')


#write a script that combines the 3 files:

#start by importing the files:
comps = pd.read_csv('comps.csv')
company_pull = pd.read_csv('CompanyPull.csv')
fake_pull = pd.read_csv('fake_company_pull.csv')


#company_pull has removed the word "Purpose" from the purpose column - do the same for the other 2 dataframes:
comps['Purpose'] = comps['Purpose'].str.replace('Purpose: ', '')
fake_pull['Purpose'] = fake_pull['Purpose'].str.replace('Purpose: ', '')


#remove the first column for company_pull:
company_pull = company_pull.iloc[:,1:]


#join these dataframes together:
purposes = comps.append(company_pull)
purposes = purposes.append(fake_pull)

purposes.index = np.arange(0, len(purposes))


#run NLP on the companies - sort by sentiment of their purpose:
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiments = []

for purpose in purposes['Purpose']:
    sentiment = sentiment_analyzer.polarity_scores(purpose)
    sentiment = sentiment['compound']
    sentiments.append(sentiment)

purposes['Purpose Sentiment'] = sentiments


#sort this dataframe by sentiment:
purposes = purposes.sort_values(by = 'Purpose Sentiment', ascending = False)

purposes.to_csv('Purposes.csv')

#report the top 5 and bottom 5 companies by sentiment:
purposes['Title'][0:5]
purposes['Title'][-5:]
