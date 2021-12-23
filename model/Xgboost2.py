# plot feature importance manually
import numpy as np
from xgboost import XGBClassifier
from matplotlib import pyplot
from sklearn.datasets import load_iris
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# load data
dataset = load_iris()
# split data into X and y
X = dataset.data
y = dataset.target

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# feature importance
print(model.feature_importances_)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:%.2f%%" % (accuracy * 100.0))

# fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

#网格搜索

from sklearn.model_selection import GridSearchCV
tuned_parameters= [{'n_estimators':[100,200,500],
                  'max_depth':[3,5,7], ##range(3,10,2)
                  'learning_rate':[0.5, 1.0],
                  'subsample':[0.75,0.8,0.85,0.9]
                  }]
tuned_parameters= [{'n_estimators':[100,200,500,1000]
                  }]
clf = GridSearchCV(XGBClassifier(silent=0,nthread=4,learning_rate= 0.5,min_child_weight=1, max_depth=3,gamma=0,subsample=1,colsample_bytree=1,reg_lambda=1,seed=1000), param_grid=tuned_parameters,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
clf.fit(X_train, y_train)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)
y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
y_proba=clf.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_true, y_proba) )

from sklearn.model_selection import GridSearchCV
parameters= [{'learning_rate':[0.01,0.1,0.3],'n_estimators':[1000,1200,1500,2000,2500]}]
clf = GridSearchCV(XGBClassifier(
             max_depth=3,
             min_child_weight=1,
             gamma=0.5,
             subsample=0.6,
             colsample_bytree=0.6,
             objective= 'binary:logistic', #逻辑回归损失函数
             scale_pos_weight=1,
             reg_alpha=0,
             reg_lambda=1,
             seed=27
            ),
            param_grid=parameters,scoring='roc_auc')
clf.fit(X_train, y_train)
print(clf.best_params_)
y_pre= clf.predict(X_test)
y_pro= clf.predict_proba(X_test)[:,1]
print ("AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro))
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))

import pandas as pd
import matplotlib.pylab as plt
feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

