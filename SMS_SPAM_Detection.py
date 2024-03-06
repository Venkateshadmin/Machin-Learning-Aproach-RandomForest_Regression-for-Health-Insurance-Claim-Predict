################### SMS Spam Detection ##################################

#The file we are going to use contains a collection of more than 5 thousand SMS phone messages. 
#Using labeled ham and spam examples, we'll train a machine learning model to learn to discriminate between ham/spam automatically.
#Then, with a trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam.
#Here I am going to develop an SMS spam detector using SciKit Learn's Naive Bayes classifier algorithm.
#However before feeding data to Machine Learning NB algorithim, we need to process each SMS with the help of Natural Language libraries.

import nltk
from nltk.corpus import stopwords
import string

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

messages = pd.read_csv(r'C:\Users\Administrator\Desktop\NLP Project\spam.csv',encoding = 'latin-1')
messages.head()
messages.tail()

#Remove the unnecessary columns for dataset and rename the column names

messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "message"]

messages.head()

#check out some of the stats with some plots and the built-in methods in pandas

messages.info()

messages.describe()

#There is two unique labels.
#There are some repeated messages as unique is less that the count due to some comman messages

#So will use groupby to use describe by label, this way we can begin to think about the features that separate ham and spam!

messages.groupby('label').describe().T

#4825 ham messages out of which 4516 are unique..
#747 span messages out of which 653 are unique.
#"Sorry, I'll call later" is the most popular ham message with repetition of 30 times.
#"Please call our customer service representativ..." is the most popular spam message with repetition 4 times.

#Let's make a new feature to detect how long the text messages are:

messages['length'] = messages['message'].apply(len)
messages.head()

# Count the frequency of top 5 messages.
messages['message'].value_counts().rename_axis(['message']).reset_index(name='counts').head()

#Data Visualization of ham and spam

messages["label"].value_counts().plot(kind = 'pie',explode=[0, 0.1],figsize=(6, 6),autopct='%1.1f%%',shadow=True)
plt.title("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()

# as per studay from graph: A lot of messages are actually not spam. About 86% of our dataset consists of normal messages.

messages['length'].describe()

#910 characters, let's use masking to find this message

messages[messages['length'] == 910]['message'].iloc[0]

########################## Text Pre-processing  ######################3

#Our main issue with our data is that it is all in text format (strings).
#The classification algorithms will need some sort of numerical feature vector in order to 
#perform the classification task. There are actually many methods to convert a corpus to a 
#vector format. The simplest is the the bag-of-words approach, where each unique word in a 
#text will be represented by one number.

#First removing punctuation. We can just take advantage of Python's built-in string library 
#to get a quick list of all the possible punctuation

def text_preprocess(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower()
    
    # Now just remove any stopwords and non alphabets
    nostop=[word for word in nopunc.split() if word.lower() not in stopwords.words('english') and word.isalpha()]
    
    return nostop

#Now "tokenize" these spam or ham messages. 
#Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).

#study of individual spam and ham messages
spam_messages = messages[messages["label"] == "spam"]["message"]
ham_messages = messages[messages["label"] == "ham"]["message"]
print("No of spam messages : ",len(spam_messages))
print("No of ham messages : ",len(ham_messages))

spam_words = text_preprocess(spam_messages)
# will see some spam words
spam_words[:10]

#Wordcloud for Spam Messages
#from wordcloud import WordCloud
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.stem import WordNetLemmatizer

#spam_wordcloud = WordCloud(width=600, height=400).generate(' '.join(spam_words))
#plt.figure( figsize=(10,8), facecolor='k')
#plt.imshow(spam_wordcloud)
#plt.axis("off")

#Print top 10 spam words

print("Top 10 Spam words are :\n")
print(pd.Series(spam_words).value_counts().head(10))

#Ham words
ham_words = text_preprocess(ham_messages)
ham_words[:10]
print("Top 10 Ham words are :\n")
print(pd.Series(ham_words).value_counts().head(10))

################  clean our data by removing punctuations/ stopwords ############
messages.head()
# Lets remove punctuations/ stopwords from all SMS 
messages["message"] = messages["message"].apply(text_preprocess)

# Conver the SMS into string from list
messages["message"] = messages["message"].agg(lambda x: ' '.join(map(str, x)))

messages.head()
messages["message"][7]

####################### Vectorization #################
#Currently, we have the messages as lists of tokens (also known as lemmas) and 
#now we need to convert each of these messages into a vector the SciKit Learn's algorithm models can work with.
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
# Creating the Bag of Words
vectorizer = CountVectorizer()
bow_transformer = vectorizer.fit(messages['message'])

print("20 Bag of Words (BOW) Features: \n")
#print(vectorizer.get_feature_names()[20:40])

print("\nTotal number of vocab words : ",len(vectorizer.vocabulary_))

message4 = messages['message'][3]
print(message4)

#Now see its vector representation:
# fit_transform : Learn the vocabulary dictionary and return term-document matrix.
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

#This means that there are seven unique words in message number 4 
#(after removing common stop words). Let's go ahead and check and confirm which ones appear twice
print(bow_transformer.get_feature_names()[5945])

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

############TF-IDF( term frequency-inverse document frequency)##############
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

#try classifying our single random message and checking how we do
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
#print(bow_transformer.get_feature_names()[5945])
#print(bow_transformer.get_feature_names()[3141])

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['say']])

#To transform the entire bag-of-words corpus into TF-IDF corpus at once
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

messages["message"][:10]

#Lets convert our clean text into a representation that a machine learning model 
#can understand. I'll use the Tfifd for this

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(messages["message"])
print(features.shape)

print(len(vec.vocabulary_))

####################### Model Evaluation ###############################
#With messages represented as vectors, we can finally train our spam/ham classifier. 
#the Naive Bayes classifier algorithm.
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
msg_train, msg_test, label_train, label_test = \
train_test_split(messages_tfidf, messages['label'], test_size=0.2)

print("train dataset features size : ",msg_train.shape)
print("train dataset label size", label_train.shape)

print("\n")

print("test dataset features size", msg_test.shape)
print("test dataset lable size", label_test.shape)

#The test size is 20% of the entire dataset (1115 messages out of total 5572), 
#and the training is the rest (4457 out of 5572). Note the default split would 
#have been 30/70.

#create a Naive Bayes classifier Model using Scikit-learn


clf = MultinomialNB()
spam_detect_model = clf.fit(msg_train, label_train)

predict_train = spam_detect_model.predict(msg_train)

print("Classification Report \n",metrics.classification_report(label_train, predict_train))
print("\n")
print("Confusion Matrix \n",metrics.confusion_matrix(label_train, predict_train))
print("\n")
print("Accuracy of Train dataset : {0:0.3f}".format(metrics.accuracy_score(label_train, predict_train)))
