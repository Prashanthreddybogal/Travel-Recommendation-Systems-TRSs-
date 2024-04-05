









import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.feature_selection import RFE

dataset = pd.read_csv('dataset.txt')
dataset.head()

test = pd.read_csv('test.txt')

feature_cols = ['userid','art_galleries','dance_clubs','juice_bars','restaurants','museums','resorts','parks_picnic_spots','beaches','theaters','religious_institutions']

y = dataset['location']
X = dataset.drop(['location'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


clf = DecisionTreeClassifier()
rfe = RFE(clf, 3)

# Train Decision Tree Classifer
fit = rfe.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = rfe.predict(test)
print("predicted : "+str(y_pred))

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

#r = export_text(clf, feature_names=feature_cols)
#print(r)