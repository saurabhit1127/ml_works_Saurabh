
####### Necessary modules imported ########
"""

The module used in this project numpy,matplotlib,sklearn,pandas
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB

######### data generation #########
def generate_data():
    np.random.seed(123)  # Set seed for reproducibility. Please do not change/remove this line.
    x = np.random.uniform(-1, 1, (1024, 2))  # changed the no. of samples from 128 to 1024 .
    y = []
    for i in range(x.shape[0]):
        y.append(np.sign(x[i][0] ** 2 + x[i][1] ** 2 - 0.5))  # Forming labels
    return x, y

def flip_labels(y):

	num = int(0.05 * len(y)) #5% of data to be flipped
	np.random.seed(123)
	changeind = np.random.choice(len(y),num,replace=False) #Sampling without replacement
	#For example, np.random.choice(5,3) = ([0,2,3]); first argument is the limit till which we intend to pick up elements, second is the number of elements to be sampled

	#Creating a copy of the array to modify
	yc=np.copy(y) # yc=y is a bad practice since it points to the same location and changing y or yc would change the other which won't be desired always
	#Flip labels -1 --> 1 and 1 --> -1
	for i in changeind:
		if yc[i]==-1.0:
			yc[i]=1.0
		else:
			yc[i]=-1.0

	return yc

######## Model generation for decision tree ########


def train_test_dt(x,y):
    """

    :param x: input matrix for train dataset
    :param y: target value for the model
    printing training and test accuracy for the model and displaying graph between K and train and test accuracy.

    Notes:

    In this function , k-fold validation is used to split the dataset. Decision tree classifier is being used for
    classification


    ------
    """
    yc=flip_labels(y)
    kf = KFold(n_splits=10)
    kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
    print(kf)
    train_acc_list=[]
    test_acc_list=[]
    i=0
    for train_index, test_index in kf.split(x):
        i=i+1
        #print(f"TRAIN:{train_index}, TEST:{test_index}")
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = yc[train_index], yc[test_index]
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)   #create object for decision tree classifier
        classifier.fit(X_train,y_train)
        y_train_pred=classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_acc=accuracy_score(y_train, y_train_pred)    #accuracy score for the training data
        train_acc_list.append(train_acc)
        test_acc=accuracy_score(y_test, y_test_pred)       #accuracy score for the test data
        test_acc_list.append(test_acc)
        print(f'k:{i}')
        print(f"train_acc_dt:{train_acc},test_acc_dt:{test_acc}")
        #no_of_correctly_classified=accuracy_score(y_test, y_pred, normalize=False)
        #print(f"no_of_correctly_classified:{no_of_correctly_classified}")
    avg_train_acc=np.mean(train_acc_list)                                        #average train accuracy
    avg_test_acc=np.mean(test_acc_list)                                          #average test accuracy
    print(f'avg_train_acc_dt:{avg_train_acc},avg_test_acc_dt:{avg_test_acc}')
    k=[i for i in range(1,11)]
    df=pd.DataFrame(list(zip(k,train_acc_list,test_acc_list)),columns =['k','train_acc_dt', 'test_acc_dt']) #created data frame
    df.plot(x='k',y=['train_acc_dt','test_acc_dt'],figsize=(12,10))               #plot for k vs train and test accuracy
    plt.show()



def train_test_nb(x,y):
    """

    :param x: input matrix for train dataset
    :param y:target value for the model
    printing training and test accuracy for the model and displaying graph between K and train and test accuracy.

    Notes:

    In this function , k-fold validation is used to split the dataset. Naive bayes is being used for
    classification

    :return:
    """
    yc=flip_labels(y)
    kf = KFold(n_splits=10)
    kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator
    print(kf)
    train_acc_list=[]
    test_acc_list=[]
    i=0
    for train_index, test_index in kf.split(x):
        i=i+1
        #print(f"TRAIN:{train_index}, TEST:{test_index}")
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = yc[train_index], yc[test_index]
        classifier = GaussianNB()                          #creating object for naive bayes classifier
        classifier.fit(X_train,y_train)
        y_train_pred=classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_acc=accuracy_score(y_train, y_train_pred)
        train_acc_list.append(train_acc)
        test_acc=accuracy_score(y_test, y_test_pred)
        test_acc_list.append(test_acc)
        print(f'k:{i}')
        print(f"train_acc_nb:{train_acc},test_acc_nb:{test_acc}")
        #no_of_correctly_classified=accuracy_score(y_test, y_pred, normalize=False)
        #print(f"no_of_correctly_classified:{no_of_correctly_classified}")
    avg_train_acc=np.mean(train_acc_list)                  #average train accuracy
    avg_test_acc=np.mean(test_acc_list)                    #average test accuracy
    print(f'avg_train_acc_nb:{avg_train_acc},avg_test_acc_nb:{avg_test_acc}')
    k=[i for i in range(1,11)]
    df=pd.DataFrame(list(zip(k,train_acc_list,test_acc_list)),columns =['k','train_acc_nb', 'test_acc_nb']) #created data frame
    df.plot(x='k',y=['train_acc_nb','test_acc_nb'],figsize=(12,10))               #plot for k vs train and test accuracy
    plt.show()


def main():

	x,y = generate_data() #Generate data
	y = flip_labels(y) #Flip labels
	y=np.asarray(y) #Change list to array
	train_test_dt(x,y)
	train_test_nb(x,y)


if __name__=='__main__':
	main()




