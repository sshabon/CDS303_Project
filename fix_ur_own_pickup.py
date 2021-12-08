#import packages

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 


from kmodes.kmodes import KModes


#dummy data, can be changed to represent actual data. once i get internet back, i can include actual values from our dataset!
#car = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

#car_problem = np.array(['oil change', 'brake change', 'windshield wipers', 'brakes', 'oil change', 'oil change']) #car problems

#make = np.array(['Volkswagen', 'Chevy', 'Audi', 'Chevy', 'Audi', 'Volkswagen'])

#model = np.array(['GTI', 'Tahoe', 'A4', 'A5', 'Tahoe', 'GTI']) #will we even need this? it might complicate things 



data = pd.read_excel(r'D:\MainDataWarehouse.xlsx', sheet_name = 'FLAT_CMPL_DB1') #<----- we use this for our actual data and the kmodes clustering code should work with it

df = pd.DataFrame(data, columns= ['Make']) 
print(df)#<----if we want to show the data as a table 
#remove the pound when we want to test it with our actual data

#data = pd.DataFrame({'car_problem':car_problem, 'make':make, 'model':model}) #<----- this is using the dummy data


#Curve to find optimal K (hopefully this will work? provides some insight on how we can tune K) we can adjust the number as necessary 
#this is from the article on Kmodes clustering for categorical data
#i have adjusted it to include more clusters so we can have better insight for our actual data

cost = []

K = range(1,100)
for num_clusters in list(K):
	kmode = KModes(n_clusters = num_clusters, init = "random", n_init = 100, verbose = 1)
	kmode.fit_predict(data)
	cost.append(kmode.cost_)

#plotting curve
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()


#actual model code for clustering

kmode = KModes(n_clusters = 100, init = "random", n_init = 100, verbose = 1)
clusters = kmode.fit_predict(data)
print(clusters)







#references: https://datatofish.com/read_excel/
#https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/