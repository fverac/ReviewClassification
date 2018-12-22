import csv
import re

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt


#reads in data into xtrain and ytrain variables
with open("trainreviews.txt", "r") as ins:
    xtrain = []
    ytrain = []
    for line in ins:
        head,sep,tail = line.partition("\t")
        #head = re.sub('[^A-Za-z0-9]+', '', head)
        head = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", head) #replace special characters with space
        head = ''.join([i for i in head if not i.isdigit()]) #remove digits
        head = head.lower() #make all characters lowercase

        tail = float(tail) #make the value a number
        xtrain.append(head) #add sentence to xtrain
        ytrain.append(tail) #add class value to ytrain



#reads in test data into xvalidation matrix
with open("testreviewsunlabeled.txt", "r") as ins:
    xreal = []
    yreal = []
    for line in ins:
        head,sep,tail = line.partition("\t")
        head = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", head) #replace special characters with space
        head = ''.join([i for i in head if not i.isdigit()]) #remove digits
        head = head.lower() #make all characters lowercase

        #tail = float(tail)
        xreal.append(head)
        yreal.append(tail)

#done reading in reviews





#word embeddings
textmatrix = []
with open("glove.6B.50d.txt", "r") as f:
	textmatrix = [line.split() for line in f]


#create dict of word embeddings where keys are words and values are word vectors
dict = {}
for line in textmatrix:
	list2 = [float(i) for i in line[1:]]
	dict[line[0]] = list2



embeddings = []
lineno = 0
#for every document add calculate sum of word embeddings
for line in xtrain:
	sum = np.zeros(50)
	for word in line.split():
		try:
			sum = sum + np.array( dict[word] )
		except:
			pass
	
	embeddings.append(sum)
	lineno = lineno+1


#embeddiings is the final data matrix







#vectorizer for bag of words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(xtrain) #makes a big of words matrix for training data

Xreal = vectorizer.transform(xreal) #bag of words for validation set


clforig = LogisticRegression(C = 10) #final model chosen after experiments
#clforig = KNeighborsClassifier()
#clforig = AdaBoostClassifier()

clforig.fit(X,ytrain)
preds = clforig.predict(Xreal)




"""this is code that writes the predicted-labels file
with open('predicted-labels.txt', 'w') as f:
    for item in preds:
    	item = int(item)
    	f.write("%s\n" % item)
"""




###
#The following code was used to run cross validation for 3 models and plot errors
###

parameters = {'C':[0.01, 0.1, 1, 10, 100, 1000]}
#parameters = {'n_neighbors':[1, 10, 50, 100, 200]}
#parameters = {'n_estimators':[50,100,150,200]}


clf = GridSearchCV(clforig, parameters, cv=5) 
clf.fit(embeddings,ytrain) #run cross validation


#
#debugging print statements
print("score is:")
print( clf.best_score_ )
print("best param?:")
print(clf.best_params_)
print(clf.cv_results_)

print("\ntesterror:")
testscores = clf.cv_results_['mean_test_score'] 
testscores = 1-testscores
print(testscores)

print("trainingerror")
trainscores = clf.cv_results_['mean_train_score']
trainscores = 1- trainscores
print(trainscores)


print("parameter values:")
paramvalues = []
paramdicts = clf.cv_results_['params']
for paramdict in paramdicts:
	paramvalues.append(paramdict['C'])
paramvalues = np.log10(paramvalues) #for logreg only
print(paramvalues)
#
#



#plotting code
plt.plot(paramvalues, testscores, label = "validation set" )
plt.plot(paramvalues, trainscores , label = "training set")
plt.legend()
#plt.title("Adaboost training/test accuracy vs n_estimators (Word Embedding)")
#plt.title("KNN training/valid. error vs n_neighbors (Word Embedding)")
plt.title("Logistic Regression training/valid. error vs C (Bag of Words)")

#plt.xlabel("n_estimators")
#plt.xlabel("n_neighbors")
plt.xlabel("log10(C)")

plt.ylabel("error")
plt.show()


