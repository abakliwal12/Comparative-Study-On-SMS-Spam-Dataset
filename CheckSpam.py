import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
df = pd.read_csv('./spam.csv', encoding='latin-1')
df = df.loc[:,['v1','v2']]
d={'spam':1,'ham':0}
d1={0:'Ham',1:'Spam'}
df.v1 = list(map(lambda x:d[x],df.v1))

class stemmed_tfidf():
    def __init__(self,max_features=5000):
        self.ps = PorterStemmer()
        self.vc = TfidfVectorizer(analyzer='word',#{‘word’, ‘char’}  Whether the feature should be made of word or character n-grams
                             stop_words = 'english',
                             max_features = max_features)
    def tfidf(self,ListStr):
        '''
        return: sklearn.feature_extraction.text.TfidfVectorizer
        '''
        table = self.vc.fit_transform([self.stem_string(s) for s in ListStr])
        return table
    def stem_string(self,s):
        '''
        s:str, e.g. s = "Get strings with string. With. Punctuation?"
        ps: stemmer from nltk module
        return: bag of words.e.g. 'get string with string with punctuat'
        '''    
        s = re.sub(r'[^\w\s]',' ',s)# remove punctuation.
        tokens = word_tokenize(s) # list of words.
        #a = [w for w in tokens if not w in stopwords.words('english')]# remove common no meaning words
        return ' '.join([self.ps.stem(w) for w in tokens])# e.g. 'desks'->'desk'
a=list(df.v2)
stf = stemmed_tfidf()
feature = stf.tfidf(a) # this will be a sparse matrix of size (n,5000)
svc = SVC(kernel='sigmoid', C=1.25, gamma=0.825,class_weight='balanced',probability=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(feature, df.v1, test_size=0.0, random_state=1)
svc.fit(Xtrain,ytrain)
x=input("Enter your message:\n")
a[0]=x
test=stf.tfidf(a)
ans=list(svc.predict_proba(test[0])[0])
print("Prediction :  ",d1[ans.index(max(ans))],"\nConfidence:  ",max(ans))
