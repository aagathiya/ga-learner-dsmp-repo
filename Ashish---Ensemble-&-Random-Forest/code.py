# --------------
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(path)
print(df.head())
print('*'*55)
print(df.shape)
print('*'*55)
#print(df.info())
print('*'*55)
#print(df.describe)
print('*'*55)
X = df.iloc[:,:1090]
print(X.shape)
print('*'*55)
y = df.iloc[:,-1]
print(y.shape)
print('*'*55)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state=4)
#Initialize MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)
#X_train = scaler.transform



# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
roc_score = roc_auc_score(y_pred, y_test)
print(roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 4)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
roc_score = roc_auc_score(y_pred, y_test)
print(roc_score)


# --------------
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 4)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
roc_score = roc_auc_score(y_pred,y_test)
print(roc_score)
# Code strats here



# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(n_estimators = 100, max_samples=100, random_state=0)
bagging_clf.fit(X_train, y_train)
score_bagging = bagging_clf.score(X_test, y_test)
print(score_bagging)
# Code starts here


# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
print(model_list)
voting_clf_hard = VotingClassifier(estimators = model_list, voting = 'hard' )
voting_clf_hard.fit(X_train, y_train)
hard_voting_score = voting_clf_hard.score(X_test, y_test)
print('Hard Voting Score-', hard_voting_score)


# Code ends here


