import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading Csv file
data_points_train = pd.read_csv('train.csv', header=None, nrows=111)
#converting the dataframe into a matrix
train = np.array(data_points_train.values)[1:,:]

#matrix to store values of month and year
graph = np.zeros((len(train),3))
for i in range(len(train)):
    graph[i][2] = float(train[i][1])
    #spliting the string
    l = str(train[i][0]).split("/")
    graph[i][0] = int(l[0])
    graph[i][1] = int(l[2])

#print(nda)
# holds  100 data points
text_x1 = graph[:100,0:1]
text_x2 = graph[:100,1:2]
text_t  = graph[:100,2:3]
M = np.ravel(text_x1+12*text_x2)

#holds remaining data-points
tp_x1 = graph[100:,0:1]
tp_x2 = graph[100:,1:2]
tp_t  = graph[100:,2:3]
U = np.ravel(tp_x1+12*tp_x2)
#ploting the curve on values of t vs month + 12*year
#which appears to be sinusoidal
#so we get initution to take basis function as sinusoidal
'''sorted_x, sorted_y = zip(*sorted(zip(np.ravel(text_x1+12*text_x2),np.ravel(text_t))))
plt.plot(sorted_x,sorted_y)
plt.ylabel('value of actual measurement')
plt.xlabel('month+12*year')
plt.title('month+12*year Vs actual values')
plt.legend()
plt.show()'''

#Reading Csv file
data_points_test = pd.read_csv('test.csv', header=None, nrows=11)
#converting the dataframe into a matrix
test = np.array(data_points_test.values)[1:,:]

#matrix to store values of month and year
matrix = np.zeros((len(test),2))
for i in range(len(test)):
    #spliting the string
    k = str(test[i][0]).split("/")
    matrix[i][0] = int(k[0])
    matrix[i][1] = int(k[2])
    
    
maty_x1 = matrix[:,0:1]
maty_x2 = matrix[:,1:2]
final = maty_x1+12*maty_x2


#constructing design matrix Pi
def design_matrix(x,m):
    n = len(x)
    Pi = np.ones((n,m+1))
    if(m%2==0):
      for j in range(n):
        for i in range(1,m,2):   
            Pi[j,i]   = np.sin((i+1)*0.5*57.1*(x[j])/110)
            Pi[j,i+1] = np.cos((i+1)*0.5*57.1*(x[j])/110)
    else:
      for j in range(n):
        for i in range(1,m,2):   
            Pi[j,i]   = np.sin((i+1)*0.5*57.1*(x[j])/110)
            Pi[j,i+1] = np.cos((i+1)*0.5*57.1*(x[j])/110)
        Pi[j,m] =  np.sin((m+1)*0.5*57.1*(x[j])/110)
    return Pi


#results holds the values of t obtained after multiplying the given value of x with the weights obtained by moore-penrose 
def result(m,cofficient,col_x):
    return np.matmul(design_matrix(col_x,m),(cofficient))
    
#moore-penrose minimization it outputs the weights after optimization
def moore_penrose(P,t,m,lamb):

    #compute moore_penrose psuedoinverse(pinv) and stored in res
    
    res = np.matmul(np.linalg.pinv(lamb*np.identity(m+1)+np.matmul(np.transpose(P),P)),np.transpose(P))
    
    #cofficient matrix store all the weights i.e. wi's
    
    cofficient = np.matmul(res,t)
    
    return cofficient

print(np.ravel(result(24,moore_penrose(design_matrix(M,24),text_t,24,10**-18),final)))

# error is the Erms error 
def error(m,cofficent,x,t):
    tp = result(m,cofficent,x)
    test = np.square((np.subtract(t,tp)))
    Erms = ((np.sum(test))/len(t))**0.57 
    return Erms

'''matrix = np.zeros((100,3))
for m in range(100):
    matrix[m][0] = m
    matrix[m][2] = error(m,moore_penrose(design_matrix(M,m),text_t,m,0),U,tp_t)
    matrix[m][1]=  error(m,moore_penrose(design_matrix(M,m),text_t,m,0),M,text_t)

print(matrix)
fig = plt.figure(1)
#plt.plot(matrix[2:,0:1],matrix[2:,1:2],label = 'Training')
plt.plot(matrix[2:,0:1],matrix[2:,2:3],label = 'Testing')
plt.xlabel('Degree(m)')
plt.ylabel('Erms')
plt.title('Erms vs Degree(m)')
plt.legend()
plt.show()'''
     

