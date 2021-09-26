#Model for Kaggle's Natural Language Processing with Disasters Tweet challenge
#https://www.kaggle.com/c/nlp-getting-started

import pandas as pd
from os import getcwd
from string import punctuation
from re import sub
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
from nltk.corpus import stopwords
#from nltk import download

#Importing English stopwords from the NLTK library
#download("stopwords")
stopwords = stopwords.words('english')

#Setting the working directory and training data CSV path
wd = getcwd()
train_file = wd + "\\train.csv"

#Listing numbers and punctuation characters for preprocessing
punctuations = list(punctuation)
numbers = list("0123456789")
punc_numb = punctuations + numbers

#Reading data from CSV and loading it into dataframes
train_df = pd.read_csv(train_file)

#Removing text data for pre-processing
train_list = list(train_df['text'])

#Data preprocessing by filtering out punctuations, numbers, websites, stopwords and converting all letters to lowercase
def preprocess(data_list):
	data_list = list(map(lambda x:''.join(i for i in x if not i in punc_numb),data_list)) #Filtering out all the punctuations and numbers, including hashtags and twitter handles
	data_list = list(map(lambda x:sub(r'http\s?\w+',r'',x),data_list)) #Filtering out all the websites beginning with http or https
	data_list = list(map(str.lower,data_list)) #Converting all characters to lowercase
	data_list = list(map(lambda x:' '.join(list(i for i in str.split(x) if i not in stopwords)),data_list)) #Removing stopwords from the NLTK library stopwords list
	return data_list

train_list = preprocess(train_list)

#Loading the preprocessed list back into a dataframe
train_text_df = pd.DataFrame(train_list)
train_text_df.columns = ["text"]

#Merging the processed dataframe back into the main dataframe. This is to ensure that the dimensions of the dataframe matches and corresponds to the label value
train_df['text'] = train_text_df 

#Splitting the training data at a 80/20 ratio for scoring
text_train, text_test, target_train, target_test = train_test_split(train_df['text'],train_df['target'],test_size=0.2)

#Logistic Regression predictor pipeline using the word count vectorizer
logreg_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('logreg', LogisticRegression())
])

#Multinominal Naive Bayes predictor pipeline using the TF-IDF vectorizer
mnb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB())
])

#Stacking the classifiers using the Sklearn.ensemble function
classifiers = [
    ("unigram", logreg_pipe),
    ("tfidf", mnb_pipe)
]
sc = StackingClassifier(classifiers)

#Fitting and predicting
sc.fit(text_train,target_train)
predictions = sc.predict(text_test)

#Checking accuracy of the prediction
print(classification_report(target_test,predictions))