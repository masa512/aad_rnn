import numpy as np

############ Data generation #############

def data_gen(N,snr=10):
    
    # Randomly generate coefficients
    a = np.random.rand()
    
    # Sample N points in [0,1] uniformly
    X = np.linspace(0,1,N)
    
    # Apply transform for each point
    Y = [a*x+1/snr*np.random.normal() for x in X]
    
    return a, (X,Y)


########### LMS FIT ################

def LMS_fit(X,Y):
    
    N = len(X)
    
    # Evaluate the variance of X using monte carlo
    #mx = 1/N * sum(X)
    vx = 1/N * sum([(x - 0)**2 for x in X])
    
    # Evaluate the covariance of X and Y using monte carlo
    #my = 1/N * sum(Y)
    vxy = 1/N * sum([(x-0)*(y-0) for x,y in zip(X,Y)])
    
    # Evaluate a
    a_pred = vxy/(vx+1e-8)
    
    return a_pred
    

########### WLS ############


def WLS(X,Y,lambd=1):
    
    a_hist = []
    N = len(X)
    for i in range(N):
        # Sample variance and covariance
        vx = 1/(i+1) * sum([lambd**(-(j+1))*(X[j] - 0)*(X[j] - 0) for j in range(i)])
        vxy = 1/(i+1) * sum([lambd**(-(j+1))*(X[j] - 0)*(Y[j] - 0) for j in range(i)])
        a = vxy/(vx+1e-8)
        a_hist.append(a)
    
    return a_hist

######### RLS ###############

def RLS(X,Y,lambd=1,delta=1):
    
    a_hist = []
    P = 1/delta
    Q = 1/delta
    N = len(X)
    for i in range(N):
        # Update P and Q
        Q = Q + 1/lambd * Y[i]*X[i]
        P = P - (1/lambd * P**2 * X[i]**2)/(1+1/lambd*P*X[i]**2)
        
        # Update a
        a = Q * P
        a_hist.append(a)
    
    return a_hist
    