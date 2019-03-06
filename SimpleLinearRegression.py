import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
print("This is Simple Linear Regression Code Via Sci-kit in Sci-kit Series")
#Input predefined data
print("Enter your set : 1, else go with Predefined : 0\n")
inp=int(input(""))
if inp == 0:
    X=np.array([[6],[8],[10],[14],[18]]).reshape(-1, 1)
    y=np.array([7, 9, 13, 17.5,18])
else:
    X_list=input("Enter depending instances with spaces").split()
    X_list=list(map(float, X_list))
    X=np.asarray(X_list).reshape(-1, 1)
    y_list=input("Enter to predict instances with spaces").split()
    y_list=list(map(float, y_list))
    y=np.asarray(y_list).reshape(-1, 1)
#Lets plot your inputs
plt.figure()
plt.title("Graph for checking variations")
plt.xlabel('Resulting Instances')
plt.ylabel('Depending Parameter')
plt.plot(X, y, 'k.') #Your Data
plt.plot([0,5,10,25],[0,5,10,25]) #Line to differentiate
plt.axis([0, 25, 0 ,25])
plt.grid(True)
plt.show()
#Lets proceed if its Linear
model= LinearRegression()
model.fit(X,y)             # here we are fitting into our algorithm
test=int(input("Enter one depending instance"))
test=np.asarray(test)
final_result=model.predict(test)[0]   # as the name suggest
print(final_result)
#Lets find errors now
#For that we require {1} TEST DATASET {2} TRAIN DATASET
#Input data as test 
print("Enter your set : 1, else go with Predefined : 0\n")
inq=int(input(""))
if inq==0:
    X_train=np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
    y_train = [7, 9, 13, 17.5, 18]
    X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
    y_test = [11, 8.5, 15, 18, 11]
    model= LinearRegression()
    model.fit(X_train, y_train)
    r_squared = model.score(X_test, y_test)
    print("Value should lie between 1 and 0 :")
    print(r_squared)
else:
    P_list=input("Enter depending instances with spaces").split()
    P_list=list(map(float, X_list))
    P=np.asarray(X_list).reshape(-1, 1)
    q_list=input("Enter to predict instances with spaces").split()
    q_list=list(map(float, y_list))
    q=np.asarray(y_list).reshape(-1, 1)
    model= LinearRegression()
    model.fit(X, y)
    r_squared = model.score(P, q)
    print("Value should lie between 1 and 0 :")
    print(r_squared)
