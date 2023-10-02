import numpy as np

############ Data generation #############

def data_gen(N,Nd=2,snr=10):
    
    # Randomly generate coefficients
    A = np.random.rand(Nd,1)
    
    # Sample N points in [0,1] uniformly for each axis
    Nper = int(N**(1/Nd))
    X = np.meshgrid(*tuple([np.linspace(0,1,Nper) for _ in range(Nd)]))
    # Change x as list of points
    X = [x.flatten() for x in X]
    X = np.stack(X).T
    # Apply transform for each point
    Y = [X[i,:]@A+1/snr*np.random.normal() for i in range(X.shape[0])]
    
    return A, (X,Y)

############ LMS #################

def LMS_fit(X,Y):
    N = len(X)
    Cxx = 1/N * sum([x[np.newaxis,:].T @ x[np.newaxis,:] for x in X])
    Cxy = 1/N * sum([x[np.newaxis,:].T @ y[np.newaxis,:] for x,y in zip(X,Y)])
    Apred = np.linalg.inv(Cxx) @ Cxy
    return Apred


########### WLS ##################

def WLS_fit(X,Y,lambd = 0.5):
    N = len(X)
    Nd = X[0].shape[0]
    Ahist = []
    lmat = np.identity(Nd)
    for n in range(N):
        lmat = 1/lambd * lmat
        num = 1/(n+1) * sum([lmat @ (Y[i] * X[i][:,np.newaxis]) for i in range(n+1)])
        den = 1/(n+1) * sum([lmat @ X[i][:,np.newaxis] @ X[i][:,np.newaxis].T for i in range(n+1)])
        A = np.linalg.pinv(den) @ num
        Ahist.append(A)
        
    return Ahist

########### RLS #################

def RLS_fit(X,Y,lambd=0.5,delta=1/2):
    N = len(X)
    Nd = X[0].shape[0]
    Ahist = []
    lmat = 1/lambd * np.identity(Nd)
    
    # Initialize P and Q
    P = 1/delta * np.identity(Nd)
    A = np.zeros((Nd,1))
    
    # Forloop
    for i in range(N):
        # Update Q
        #Q = Q + Y[i] * lmat @ X[i][:,np.newaxis]
        # Update K
        K = 1/(lambd + X[i][:,np.newaxis].T @ P @ X[i][:,np.newaxis]) * P @ X[i][:,np.newaxis]
        # Update A
        A = A + K @ (Y[i] - X[i][:,np.newaxis].T @ A)
        Ahist.append(A)
    return Ahist