#cell 1
#dependencies
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#cell 2
#twitter API credentials
consumerKey = 'gaApf7xuRDogxySZ8UJhLR5ni'
consumerSecret = 'ceFlTYl00Sl0DEpCoRytGvt5B7Ziof7Y55C6LVawjKEJ37bWwG'
accessToken = '1317746981842735104-6rOQjuldnRwpmr048727py4WPIW27I'
accessTokenSecret = 'wJMwT6aQZgPfKAWOwC5Vd7Ux6seLmYulcszC5V5BTgV1S'


#cell 3 
#create authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

#set the access tokens and access token secrets
authenticate.set_access_token(accessToken, accessTokenSecret)

#create the API object while passing the auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)


#cell 4
#lets extract 100 tweets from a user
posts = api.user_timeline(screen_name='BillGates', count = 100, lang = 'en', tweet_mode = 'extended')

#print the last five tweets from the account
print("Print the last 5 tweets: \n")
i = 1
for tweet in posts[0:5]:
    print( str(i) + ')' + tweet.full_text + '\n' )
    i=i+1


#cell 5
#create a dataframe with a column called tweets
df = pd.DataFrame([tweet.full_text for tweet in posts] , columns = ['Tweets'])


#show the first five rows of data
df.head()


#cell 6
#clean the text
#create a function to clear the texts
def cleanTxt(text):
    text = re.sub( r'@[A-Za-z0-9]+' , '' , text ) #removing @mentions
    text = re.sub( r'#', '' , text) #removing hash tags
    text = re.sub( r'RT[\s]+' , '' , text) #removing RTs
    text = re.sub( r'https?:\/\/\s+' , '' , text) #removing the hyper links

    return text

#cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanTxt)

#show the clean text
df


#cell 7
#function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#show new dataframe
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

#show the new dataframe
df


#cell 8
#plot the word cloud
allWords = ' '.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#cell 9
#create a function to complete negative positive and neutral analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

#show the dataframe
df


#cell 10
#print all the positive tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range (0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Positive'):
        print(str(j) + ') ' + sortedDF['Tweets'][i])
        print()
        j = j+1



#cell 11
#lets print negative tweete
j=1
sortedDF = df.sort_values(by=['Polarity'], ascending = 'false')
for i in range (0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Negative'):
        print(str(j) + ') ' + sortedDF['Tweets'][i])
        print()
        j = j+1


#cell 12
#plot the polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color = 'Blue')

plt.title('Twitter Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()



#cell 13
#get the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

round( (ptweets.shape[0] / df.shape[0]) *100, 1)



#cell 14
#get the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']

round( (ntweets.shape[0] / df.shape[0]) *100, 1)



#cell 15
#show the value counts
df['Analysis'].value_counts()

#plot the chart for count
plt.title('Twitter Sentiment Analysis')
plt.xlabel('Sentiments')
plt.ylabel('counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()



print("code completed")
