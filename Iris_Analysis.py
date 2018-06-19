
#!/usr/bin/python3

from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# store load_iris() into some variable to access it using iris_name in future
iris = load_iris()

#get description 
desc = iris.DESCR

#get names of flowers
flower_name = iris.target_names

#get data for each type of flower like data of features of each flower 
flower_data = iris.data

#print(flower_data[0:10])

#for feature_names
flower_feature = iris.feature_names

# to gain target i.e 0 for setos,1 for versicolor 2 for verginia
flower_output = iris.target

# take 50 data ie 0 for setos to have predicted o/p for m/c 
predicted_sentosa = flower_output[0:50]
#to train for versicolor
predicted_versicolor = flower_output[50:100]
#to train for virginica
predicted_verginica = flower_output[100:150]



# take 49 data ie 0 for setos to have predicted o/p for m/c 
predicted_out_sentosa = predicted_sentosa[0:49]
#to train for versicolor
predicted_out_versicolor = predicted_versicolor[0:49]
#to train for virginica
predicted_out_verginica = predicted_verginica[0:49]


#print("predicted_sentosa ",predicted_sentosa)

# for output take 1 lat data ie 0 for setos to have predicted o/p for m/c 
predicted_test_out_sentosa = predicted_sentosa[-1]
#to train for versicolor
predicted_test_out_versicolor = predicted_versicolor[-1]
#to train for virginica
predicted_test_out_verginica = predicted_verginica[-1]


# to get 1st 49 features for sentosa to keep like weight of fruit
train_sentosa = flower_data[0:50]
# to get features for versicolor
train_versicolor = flower_data[50:100]
# to get features for verginia
train_verginica = flower_data[100:150]



#for training input features
train_in_sentosa = train_sentosa[0:49]
train_in_versicolor = train_versicolor[0:49]
train_in_verginica = train_verginica[0:49]


# to concatenate all the ndarrays of sentosa,versicolor,verginica for i/p
input_data = np.concatenate((train_in_sentosa,train_in_versicolor,train_in_verginica))

# to concatenate all the ndarrays of sentosa,versicolor,verginica for o/p
output_data = np.concatenate((predicted_out_sentosa,predicted_out_versicolor,predicted_out_verginica))


#for testing input features
test_sentosa = train_sentosa[-1]
test_versicolor = train_versicolor[-1]
test_verginica = train_verginica[-1]



algo = tree.DecisionTreeClassifier()
trained = algo.fit(input_data,output_data)


# the input given is for versicolor ie [[244,289,2892,839]],corresponding to feature of versicolor can change it accordingly
result = trained.predict([test_versicolor,test_sentosa,test_verginica])


print("predicted result for input is:",result)

'''
if result == 0:
	print("The given input is for flower:Sentosa")
elif result == 1:
	print("The given input is for flower:Versicolor")
else:
	print("The given input is for flower:Verginica")
'''



plt.title("flower vs dataset")
plt.xlabel("flower")
plt.ylabel("dataset")
plt.scatter(train_sentosa,train_versicolor,color='r',marker='*',label="sent-vers")
plt.scatter(train_versicolor,train_verginica,color='y',marker='*',label="vers-verg")
plt.scatter(train_sentosa,train_verginica,color='g',marker='*',label="sent-verg")
plt.legend()
plt.show()


# to plot having all rows and 1st and 2nd column for sentosa,versicolor,verginica
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.scatter(train_sentosa[:0],train_sentosa[:1],marker='*',color='g',label='setosa')
plt.scatter(train_versicolor[:0],train_versicolor[:1],marker='*',color='r',label='versicolor')
plt.scatter(train_verginica[:0],train_verginica[:1],marker='*',color='y',label='verginica')
plt.legend()
plt.show()


# to plot having all rows and 3rd and 4th column for sentosa,versicolor,verginica
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.scatter(train_sentosa[:2],train_sentosa[:3],marker='*',color='g',label='setosa')
plt.scatter(train_versicolor[:2],train_versicolor[:3],marker='*',color='r',label='versicolor')
plt.scatter(train_verginica[:2],train_verginica[:3],marker='*',color='y',label='verginica')
plt.legend()
plt.show()


