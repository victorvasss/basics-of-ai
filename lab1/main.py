#Import scikit-learn dataset library
from sklearn import datasets
#Load dataset
digits = datasets.load_digits()

# print the names of the features
print("Features: ", digits.feature_names)
# print the label type of digits
print("Labels: ", digits.target_names)

# print data(feature)shape
digits.data.shape

# print the digit data features (top 5 records)
print(digits.data[0:5])

# print the digit labels
print(digits.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Import NB model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# #Import svm model
# from sklearn import svm
# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# clf.fit(X_train, y_train)
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))