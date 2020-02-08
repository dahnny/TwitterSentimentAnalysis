import twitter
import csv
import time
from sentiment_analysis import PreProcessTweets

token = 'Bb5icqwCig3hEkNKvOkmIFOea'
token_secret = 'fbjhyc5gUiBeIDI8hSXTgGVjkw7gtgz17OE1liQV6wMZcaMsg9'
access_token = '1413417127-UX39Te8YsO6Ni2GlK6LYHSwc0EH4iXbNeeqHw06'
access_secret = 'sNVRBk7LQvLDzPudXSrYC2yYBYHaZSmigUYq3ZBKtwPF5'

twitter_api = twitter.Api(consumer_key=token, consumer_secret=token_secret,
                          access_token_key=access_token,access_token_secret=access_secret)
print(twitter_api.VerifyCredentials())


def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count= 100)

        print('Fetched '+str(len(tweets_fetched)) + ' tweets for the term '+ search_keyword)

        return[{'text': status.text, 'label':None} for status in tweets_fetched]

    except:
        print('Unfortunately, something went wrong..')
        return None

search_item = input('Enter a search keyword')
testDataSet = buildTestSet(search_item)

print(testDataSet[0:4])


def buildTrainSet(corpusFile, tweetDataFile):


    # corpus = []
    #
    # csvfile = open(corpusFile,'r')
    # lineReader = csv.reader(csvfile,delimiter = ',', quotechar = "\"")
    # for row in lineReader:
    #     corpus.append({'tweet_id':row[2], 'label':row[1], 'topic': row[0]})
    #
    # rate_limit = 180
    # sleep_time = 900/180
    #
    trainingDataSet = []
    #
    # for tweet in corpus:
    #     try:
    #         var = (tweet['tweet_id'])
    #         print(var)
    #         status = twitter_api.GetStatus(tweet['tweet_id'])
    #         print('tweet fetched' + status.text)
    #         tweet['text'] = status.text
    #         trainingDataSet.append(tweet)
    #         time.sleep(sleep_time)
    #     except:
    #         continue

    csvfile = open(tweetDataFile, 'r')
    linewriter = csv.reader(csvfile, delimiter = ',',quotechar = "\"")
    for tweet in linewriter:
        try:
            if tweet != []:
                trainingDataSet.append({'id': tweet[0],'text':tweet[1],'label':tweet[2],'topic':tweet[3]})
        except Exception as e:
            print(e)


    return trainingDataSet


corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"

trainingData = buildTrainSet(corpusFile, tweetDataFile)
print(len(trainingData))
tweetProcessor = PreProcessTweets()
preProcessedTrainingSet = tweetProcessor.processTweets(trainingData)
preProcessedTestSet = tweetProcessor.processTweets(testDataSet)

print(preProcessedTrainingSet)
print(preProcessedTestSet)


import nltk

def build_vocabulary(preprocessedTrainingData):
    all_words = []

    for (word, sentiments) in preprocessedTrainingData:
        all_words.extend(word)

    word_list = nltk.FreqDist(all_words)
    word_features = word_list.keys()

    return word_features

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

word_features = build_vocabulary(preProcessedTrainingSet)
trainingFeatures = nltk.classify.apply_features(extract_features, preProcessedTrainingSet)

NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preProcessedTestSet]

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")

