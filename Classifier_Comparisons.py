import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

f=open("wine.csv")
samples=[]
labels=[]

for line in f:
	#if(line.startswith("\"")):
    	#continue
    l=line.replace("\n","").split(",")
    instance=[]
    for i in range(len(l)-1):
        instance.append(float(l[i]))
    samples.append(instance)
    labels.append(int(l[-1]))

sampleset=[]
labelset=[]
for i in range(10):
    sampleset.append([])
    labelset.append([])
for i in range(len(samples)):
    sampleset[i%10].append(samples[i])
    labelset[i%10].append(labels[i])

trainx=[]
trainy=[]
testx=[]
testy=[]
N=9
for i in range(10):
	if(i!=N):
		for j in sampleset[i]:
			trainx.append(j)
		for j in labelset[i]:
			trainy.append(j)
	else:
		for j in sampleset[i]:
			testx.append(j)
		for j in labelset[i]:
			testy.append(j)

# DECISION TREE
dt = DecisionTreeClassifier()

# train the model
dt.fit(trainx,trainy)
# predict the labels and report accuracy
hard_pred = dt.predict(testx)
acc = np.isclose(hard_pred,testy).sum()/float(len(hard_pred))
print("Decision Tree Accuracy: {}".format(acc))

# use predicted probabilities to construct ROC curve and AUC score
soft_pred = dt.predict_proba(testx)
fpr1,tpr1,thresh = roc_curve(testy,soft_pred[:,1])
auc = roc_auc_score(testy,soft_pred[:,1])
print("Decision Tree AUC: {}".format(auc))


# RANDOM FOREST
rf = RandomForestClassifier()

# train the model
rf.fit(trainx,trainy)

# predict the labels and report accuracy
hard_pred = rf.predict(testx)
acc = np.isclose(hard_pred,testy).sum()/float(len(hard_pred))
print("Random Forest Accuracy: {}".format(acc))

# use predicted probabilities to construct ROC curve and AUC score
soft_pred = rf.predict_proba(testx)
fpr2,tpr2,thresh = roc_curve(testy,soft_pred[:,1])
auc = roc_auc_score(testy,soft_pred[:,1])
print("Random Forest AUC: {}".format(auc))


# In[16]:

# GRADIENT BOOSTED TREES
gb = GradientBoostingClassifier()

# train the model
gb.fit(trainx,trainy)

# predict the labels and report accuracy
hard_pred = gb.predict(testx)
acc = np.isclose(hard_pred,testy).sum()/float(len(hard_pred))
print("Boosting Accuracy: {}".format(acc))

# use predicted probabilities to construct ROC curve and AUC score
soft_pred = gb.predict_proba(testx)
fpr3,tpr3,thresh = roc_curve(testy,soft_pred[:,1])
auc = roc_auc_score(testy,soft_pred[:,1])
print("Boosting AUC: {}".format(auc))


# In[6]:

# LOGISTIC REGRESSION
# initialize a logistric regression object
lr = LogisticRegression()

# train the model
lr.fit(trainx,trainy)

# predict the labels and report accuracy
hard_pred = lr.predict(testx)
acc = np.isclose(hard_pred,testy).sum()/float(len(hard_pred))
print("Logistic Regression Accuracy: {}".format(acc))

# use predicted probabilities to construct ROC curve and AUC score
soft_pred = lr.predict_proba(testx)
fpr4,tpr4,thresh = roc_curve(testy,soft_pred[:,1])
auc = roc_auc_score(testy,soft_pred[:,1])
print("Logistic Regression AUC: {}".format(auc))


# In[9]:

# SUPPORT VECTOR MACHINE
sv = SVC(probability=True)

# train the model
sv.fit(trainx,trainy)

# predict the labels and report accuracy
hard_pred = sv.predict(testx)
acc = np.isclose(hard_pred,testy).sum()/float(len(hard_pred))
print("SVM Accuracy: {}".format(acc))

# use predicted probabilities to construct ROC curve and AUC score
soft_pred = sv.predict_proba(testx)
fpr5,tpr5,thresh = roc_curve(testy,soft_pred[:,1])
auc = roc_auc_score(testy,soft_pred[:,1])
print("SVM AUC: {}".format(auc))


#plot result
print("ROC Curve:")
plt.title(str(N+1))
plt.plot(fpr1,tpr1,color='r')
plt.plot(fpr2,tpr2,color='b')
plt.plot(fpr3,tpr3,color='y')
plt.plot(fpr4,tpr4,color='k')
plt.plot(fpr5,tpr5,color='g')
plt.plot([0,1],[0,1],"r--",alpha=.5)
plt.show()