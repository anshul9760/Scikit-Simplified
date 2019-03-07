# Classification and Regression
# KNN is a lazy Learner
# What to do ?
'''
Objective : Prediction of SEX {Height and Weight}
Procedure : First of all making predefined Set
            Plotting all intances
            Importing KNeighbors
            Learning Binarization
            Predicting a Value
            Forming Train and Test Data Sets
            Checking Scores
'''
######################################################################
#Read procedure again and again for better understanding, I will import one by one
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#######################################################################
#Edit Accordingly
#Else use panda to import Excel datasets
X_train=np.array([[158, 64], [170, 86], [183, 84], [191, 80], [155, 49], [163, 59], [180, 67], [158, 54], [170, 67]])
y_train= ['male','male','male','male','female','female','female','female','female']
#######################################################################
#Lets Plot
plt.figure()
plt.title('Human Heights and Weights by Sex')
plt.xlabel('Height in cm')
plt.ylabel('Weight in Kg')
for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
    plt.grid(True)
plt.show()
#https://www.programiz.com/python-programming/methods/built-in/enumerate
#Understand the looping here
#######################################################################
#Now KNN not going to understand male or female, it understands 0 or 1;
#Therefore, Binarization
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)  # Male to 1, Female to 0;
y_train_binarized.reshape(-1)
print(y_train_binarized.reshape(-1))
#######################################################################
K = 3                                       # Why 3? Then understand Why import KNeighborsClassifier first ?
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized)
nlist=input("Enter Height {space} Weight :  ").split()
nlist=list(map(int, nlist))
prediction = clf.predict(np.asarray(nlist).reshape(1, -1))[0]
label = lb.inverse_transform(prediction)
print(label)
#######################################################################
#Lets form Train and Test Datasets
X_test=np.array([[168,65],[180,96],[160,52],[169,67]])
y_test = ['male','male','female','female']
y_test_binarized = lb.transform(y_test)
prediction_b=clf.predict(X_test)
labels = lb.inverse_transform(prediction_b)
print(y_test_binarized.T[0])               #INPUT
print(prediction_b)                        #OUTPUT
print(labels)                              #SEE THE VARIATION
#######################################################################
# Lets play with Scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#...................................................
print('Accuracy : %s'%accuracy_score(y_test_binarized, prediction_b))
print('Precision Score :  %s'%precision_score(y_test_binarized, prediction_b))
print('F1 Score :  %s'%f1_score(y_test_binarized, prediction_b))
print('Report:%s'%classification_report(y_test_binarized, prediction_b, target_names=['male'], labels=[1]))
#######################################################################
