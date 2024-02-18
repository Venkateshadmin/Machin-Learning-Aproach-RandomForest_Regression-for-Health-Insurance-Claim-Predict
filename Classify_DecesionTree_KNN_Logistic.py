import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import webbrowser

# Read in track metadata with genre labels
audio1=pd.read_csv(r"C:\Users\Administrator\Desktop\Project\Classify Song Genre\Audio Data.csv")
audio=pd.DataFrame(audio1)

# Read in track metrics with the features
echonest_metrics = pd.read_json(r"C:\Users\Administrator\Desktop\Project\echonest-metrics.json",precise_float=True)

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = pd.merge(echonest_metrics,audio[["track_id", "genre_top"]], on="track_id")

# Inspect the resultant dataframe
print(echo_tracks.info())

# Replace 'Rock' with 1 and 'Hip-Hop' with 0
echo_tracks['genre_top'] = echo_tracks['genre_top'].replace({'Rock': 1, 'Hip-Hop': 0})
print(echo_tracks.info())

# Create a correlation matrix

corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()
print(corr_metrics)

# Define our features 
X = echo_tracks.drop(["genre_top","track_id"], axis=1)
y= echo_tracks["genre_top"]

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(X)

scaled_data=pd.DataFrame(scaled_train_features).head(5)
print(scaled_data)

# Import our plotting module, and PCA class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

print("Explained variance ratio : \n{}".format(pca.explained_variance_ratio_))
print("\n")
print("Number of components = {}".format(pca.n_components_))  

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(8), exp_variance)
ax.set_xlabel('Principal Component #')

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(range(8), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

#Since we didn’t find any particular strong correlations between our features, 
#we can instead use a common approach to reduce the number of features called 
#principal component analysis (PCA).
# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)

# Import train_test_split function and Decision tree classifier
from sklearn.model_selection import train_test_split
# Split our data

###############Decesion Tree###################
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection,y, random_state=10)
from sklearn.tree import DecisionTreeClassifier
# Create the classification report for both models
from sklearn.metrics import classification_report,accuracy_score

# Train our decision tree
model = DecisionTreeClassifier(criterion = 'gini', random_state = 10, max_depth=5)
model.fit(train_features, train_labels)

# Predict the labels for the test data
pred_labels_tree =model.predict(test_features)

print('The Decesion-Tree Accuracy_score is ', accuracy_score(test_labels, pred_labels_tree))
class_rep_tree = classification_report(test_labels, pred_labels_tree)
print("Decision Tree: \n", class_rep_tree)

from sklearn import tree
plt.figure(figsize=(20,15))
tree.plot_tree(model,filled=True)

###################### Hyperparameter Tuning in Decision Trees #################
from sklearn.model_selection import GridSearchCV
dt_hp = DecisionTreeClassifier(random_state=43)

params = {'max_depth':[3,5,7,10,15],
          'min_samples_leaf':[3,5,10,15,20],
          'min_samples_split':[8,10,12,18,20,16],
          'criterion':['gini','entropy']}
GS = GridSearchCV(estimator=dt_hp,param_grid=params,cv=5,n_jobs=-1, verbose=True, scoring='accuracy')
GS.fit(train_features,train_labels)
##################### best Parameter####################
print('Best Parameters:',GS.best_params_,end='\n\n')
print('Decesion Tree Hyper tuning Best Score:',GS.best_score_)

###################KK-Nearest Neighbour(KNN) ################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)
# Predict the labels for the test data
pred_labels_knn=knn.predict(test_features)
##knn.score(train_features, train_labels)

#knn.score(X_test,y_test)

from sklearn.metrics import multilabel_confusion_matrix
y_predict =knn.predict(test_features)
###It summarizes the results of classification by showing the counts of true positive, 
#true negative, false positive, and false negative predictions
confusion_matrix = multilabel_confusion_matrix(test_labels, pred_labels_knn)
print(confusion_matrix)
print('The KNN Accuracy_score is ', accuracy_score(test_labels, pred_labels_knn))
from sklearn.metrics import classification_report
print(classification_report(test_labels, pred_labels_knn))

print(echo_tracks.loc[1])
new=pd.DataFrame({
    'track_id':[3.000000],
    'acousticness':[0.374408],
    'danceability':[0.528643],
    'energy':[0.817461],
    'instrumentalness':[0.001851],
    'tempo':[126.957000]
    })

print("\n Predicted Genre_top for new data:",knn.predict(new))

#################### Hyper Tuning##########################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score
kf=KFold(n_splits=5,shuffle=True,random_state=42)
parameter={'n_neighbors': np.arange(2, 30, 1)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn, param_grid=parameter, cv=kf, verbose=1)
knn_cv.fit(train_features,train_labels)
print(knn_cv.best_params_)

knn=KNeighborsClassifier(n_neighbors=23)
knn.fit(train_features,train_labels)
y_pred=knn.predict(test_features)
accuracy_score=accuracy_score(test_labels,y_pred)*100
print("KNN Hyper Tune Best Score  : {:.2f}%".format(accuracy_score))


############################Logistic Regression########
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# Train our logistic regression and predict labels for the test set
model2 = LogisticRegression(random_state=10)
model2.fit(train_features, train_labels)
# Predict the labels for the test data
pred_labels_tree =model2.predict(test_features)
pred_labels_logit = model2.predict(test_features)
##It summarizes the results of classification by showing the counts of true positive, 
#true negative, false positive, and false negative predictions
confusion_matrix=confusion_matrix(test_labels,pred_labels_tree)
print(confusion_matrix)

print("The Logistic-Regression Accuracy score is", accuracy_score(test_labels, pred_labels_tree))
# Create the classification report for both models

class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Logistic Regression: \n", class_rep_log)

#####################Hyper Tune####################
from sklearn.model_selection import GridSearchCV

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(model2, param_grid = param_grid, cv = 3, n_jobs=-1)


best_clf = clf.fit(X,y)

print(best_clf.best_estimator_)

print (f'Accuracy - : {best_clf.score(X,y):.3f}')

################################## Comparision Of DecesionTree-KNN-LogisticRegression#####

print("Decision Tree: \n", classification_report(test_labels,pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels,pred_labels_logit))
print("KNearest Neighbours: \n", classification_report(test_labels,pred_labels_knn))


print('Decesion Tree Hyper tuning Best Score:',GS.best_score_)
print("KNN Hyper Tune Best Score  : {:.2f}%",accuracy_score(test_labels,y_pred)*100)
print (f'Logistic Regression Hyper Tune Best Accuracy - : {best_clf.score(X,y):.3f}')



