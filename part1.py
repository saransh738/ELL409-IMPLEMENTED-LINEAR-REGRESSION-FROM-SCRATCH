import argparse  
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()

if __name__ == '__main__':
    args = setup()
    
filepath = args.X
method = args.method
lamb = args.lamb
polynomial = args.polynomial
batch_size = args.batch_size   
m = int(polynomial)

# Reading data_points(x,t) from the csv files
data_points = np.genfromtxt(filepath,delimiter=',')
#print(data_points)
n = (len(data_points))
col_t = np.array(data_points[0:n,1]).reshape((n,1))
col_x = np.array(data_points[0:n,0]).reshape((n,1))

# holds  20 data points
p = int(len(data_points)*9/10)
text_x =col_x[:p]
text_t =col_t[:p]
# holds 20-100 data points
tp_x = col_x[p:]
tp_t = col_t[p:]

				
		

#constructing design matrix Pi
def design_matrix(x,m):
    n = len(x)
    
    #initializing matrix with all zeros
    Pi = np.matrix(np.zeros((n,m+1),dtype=float))
    
    for j in range(n): 
      for i in range(m+1):
        Pi[j,i] = (x[j])**i
    return Pi



#results holds the values of t obtained after multiplying the given value of x with the weights obtained by moore-penrose  
def result(m,cofficient,col_x):

    return (np.matmul(design_matrix(col_x,m),cofficient))

#moore-penrose minimization it outputs the weights after optimization
def moore_penrose(Pi,t,m,lamb):

    #compute moore_penrose psuedoinverse(pinv) and stored in res
    
    res = np.matmul(np.linalg.inv(lamb*np.identity(m+1)+np.matmul(np.transpose(Pi),Pi)),np.transpose(Pi))
    
    #cofficient matrix store all the weights i.e. wi's
    
    cofficient = np.matmul(res,t)
    
    return cofficient

# error is the Erms error 
def error(m,cofficent,x,t):
    tp   = result(m,cofficent,x)
    test = np.square((np.subtract(t,tp)))
    Erms = ((np.sum(test))/len(t))**0.5
    return Erms

    
#print(np.ravel(np.transpose(moore_penrose(design_matrix(text_x,m),text_t,m,0))))
#print(result(2,moore_penrose(design_matrix(text_x,2),text_t,2,0),text_x))
#print(np.var(np.subtract(result(15,moore_penrose(design_matrix(text_x,15),text_t,15,0),text_x),text_t)))
#print(np.mean(np.subtract(result(15,moore_penrose(design_matrix(text_x,15),text_t,15,0),text_x),text_t)))



'''matrix = np.zeros((20,3))
for m in range(20):
    matrix[m][0] = m
    matrix[m][2] = error(m,moore_penrose(design_matrix(text_x,m),text_t,0),tp_x,tp_t)
    matrix[m][1]=  error(m,moore_penrose(design_matrix(text_x,m),text_t,0),text_x,text_t)'''

'''fig = plt.figure(1)
plt.plot(matrix[2:,0:1],matrix[2:,1:2],label = 'Training')
plt.plot(matrix[2:,0:1],matrix[2:,2:3],label = 'Testing')
plt.xlabel('Degree(m)')
plt.ylabel('Erms')
plt.title('Erms vs Degree(m)')
plt.legend()
plt.show()'''

'''matrix = np.zeros((10,2))
graph = (np.subtract(text_t,result(2,moore_penrose(design_matrix(text_x,2),text_t,2,0),text_x)))
for m in range(10):
    matrix[m][0] = text_x[m]
    matrix[m][1]=  np.abs(graph[m])
fig = plt.figure(2)
#plt.plot(matrix[2:,0:1],matrix[2:,1:2],label = 'Noise')
sorted_x, sorted_y = zip(*sorted(zip(matrix[2:,0:1],matrix[2:,1:2])))
plt.plot(sorted_x,sorted_y)
plt.xlabel('x-----')
plt.ylabel('t-tp')
plt.title('Noise')
plt.legend()
plt.show()'''

'''maty = np.zeros((21,3))
for i in range(21):
    c = 10**(-i)
    maty[i][0] = c
    maty[i][1] = error(10,moore_penrose(design_matrix(text_x,10),text_t,10,i),text_x,text_t)
    maty[i][2] = error(10,moore_penrose(design_matrix(text_x,10),text_t,10,i),tp_x,tp_t)
	
fig = plt.figure(3)
plt.plot(maty[:,0:1],maty[:,2:3],label = 'Testing')
plt.plot(maty[:,0:1],maty[:,1:2],label = 'Training')
plt.xlabel('lamd')
plt.ylabel('Erms')
plt.title('Erms vs lamb')
plt.legend()
plt.show()'''

'''fig = plt.figure(4)
sorted_x, sorted_y = zip(*sorted(zip(np.ravel(text_x),np.ravel(result(9,moore_penrose(design_matrix(text_x,9),text_t,9,0),text_x)))))
plt.plot(sorted_x,sorted_y)
plt.scatter(text_x,text_t)
plt.xlabel('x_______')
plt.ylabel('t_______')
plt.title('Value of Data_points for 9th order Polynomial ')
plt.legend()
plt.show()'''

#data holds the values from 0-99 such that data[i]=i
data = np.arange(p)
#shuffles the data 
np.random.shuffle(data)

#sgd method of optimization
#eta = learning rate
#lamb = lamda
#m = order of poly
#batch_size = batch size to be taken

def stochastic_gradient_descent(eta,m,lamb,max_iter,batch_size,col_x,col_t):
    #constructing design_matrix
    P  = design_matrix(col_x,m)
    #initializing matrix with all zeros
    weight = np.zeros((m+1,1))
    
    for i in range(max_iter):
        iter = 0
        for j in range(batch_size):
          y = col_t[data[j]]
          x = P[data[j]] 
          r = x*weight
          iter = iter  + (x.T)*(y-r)
          weight = weight + iter*eta  
        weight = weight + lamb*weight*eta  
    return weight

# for autograder if else conditions based on the method
if(method == "pinv"):
     weights = np.ravel(np.transpose(moore_penrose(design_matrix(text_x,m),text_t,m,lamb)))
     print(f"weights={weights}")
    
elif(method == "gd"):    
     weights = np.ravel(stochastic_gradient_descent(0.00001,m,lamb,50000,batch_size,text_x,text_t))
     print(f"weights={weights}")
    

'''maty = np.zeros((100,2))
for i in range(100):
    maty[i][0] = i/1000
    maty[i][1]=  error(3,stochastic_gradient_descent(i/1000,3,10**-20,20000,1,text_x,text_t),text_x,text_t)
    
fig = plt.figure(5)
plt.plot(maty[:,0:1],maty[:,1:2],label = 'Training')
plt.xlabel('eta')
plt.ylabel('Erms')
plt.title('Erms vs eta')
plt.legend()
plt.show()'''


'''maty = np.zeros((10,2))
for i in range(10):
    maty[i][0] = i
    maty[i][1]=  error(3,stochastic_gradient_descent(0.0001,3,10**-20,5000,i,text_x,text_t),text_x,text_t)
    
fig = plt.figure(6)
plt.plot(maty[:,0:1],maty[:,1:2],label = 'Training')
plt.xlabel('batch_size')
plt.ylabel('Erms')
plt.title('Erms vs batch_size')
plt.legend()
plt.show()'''
       

'''maty = np.zeros((10000,3))
for i in range(10000):
    c = i/(10**20)
    maty[i][0] = c
    maty[i][2] = error(3,stochastic_gradient_descent(0.00001,3,c,5000,10,text_x,text_t),text_x,text_t)
    maty[i][1] = error(3,stochastic_gradient_descent(0.00001,3,c,5000,10,text_x,text_t),tp_x,tp_t)
	
fig = plt.figure(6)
plt.plot(maty[:,0:1],maty[:,1:2],label = 'Testing')
plt.plot(maty[:,0:1],maty[:,1:2],label = 'Training')
plt.xlabel('lamd')
plt.ylabel('Erms')
plt.title('Erms vs lamb')
plt.legend()
plt.show()'''


'''maty = np.zeros((6,3))
for i in range(7):
    maty[i][0] = i
    maty[i][2] = error(i,stochastic_gradient_descent(0.00001,i,10**-20,10000,10,text_x,text_t),text_x,text_t)
    maty[i][1] = error(i,stochastic_gradient_descent(0.00001,i,10**-20,10000,10,text_x,text_t),tp_x,tp_t)
	
fig = plt.figure(6)
plt.plot(maty[2:,0:1],maty[2:,1:2],label = 'Testing')
plt.plot(maty[2:,0:1],maty[2:,2:3],label = 'Training')
plt.xlabel('Degree(m)')
plt.ylabel('Erms')
plt.title('Erms vs Degree(m)')
plt.legend()
plt.show()'''         
     

'''laty = np.zeros((5000,2))
for i in range(5000):
    laty[i][0] = i
    laty[i][1]=  error(3,stochastic_gradient_descent(0.00001,3,10**-20,i,1,text_x,text_t),text_x,text_t)
    
fig = plt.figure(8)
plt.plot(laty[:,0:1],laty[:,1:2],label = 'Training')
plt.xlabel('no_iterations')
plt.ylabel('Erms')
plt.title('Erms vs no_iterations')
plt.legend()
plt.show()'''         
       
     

'''fig = plt.figure(9)
sorted_x, sorted_y = zip(*sorted(zip(np.ravel(col_x),np.ravel(result(3,stochastic_gradient_descent(0.0001,3,10**-20,50000,10, col_x,col_t),col_x)))))
plt.plot(sorted_x,sorted_y)
plt.scatter(col_x,col_t)
plt.xlabel('x_______')
plt.ylabel('t_______')
plt.title('Value of Data_points for 3th order Polynomial ')
plt.legend()
plt.show()'''
     
     
     