####### Import #################
import numpy as np
import scipy.io as spio

####### Loading MATLAB Data ###########
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


####### Data Loading for AAD (Bertrand) #########

def dir2label(direction):
    """
    Converts direction (either "L" or "R" into label 1 or 2
    """
    if direction == 'L':
        return 1

    else:
        return 2

def eval_weighted_env(X,W):
    """
    Linear combination of multi_channel output of the envelopes using weight

    X: T x Nc x 2
    W : 1 x 15

    Returns
    Y : list of [yL,yR] each with T x 1
    """

    # List of tracks : List of T x Nc matrices
    X = [X[:,:,0],X[:,:,1]]
    Y = []
    for i in range(len(X)):
        # Apply the transform
        Yi = X[i]@W.T
        Y.append(Yi.reshape((-1,1)))

    return Y
   
def AAD_data(path,k):
    """
    Loads up Bertrand .mat file for particular application

    path : Path for the AAD .mat file
    k : index of interest

    Output:
    Xk : EEG at trial k
    yk : List of Envelope channel at trial k [yL,yR]
    label : 1 or 2 for left or right
    """
    trial = loadmat(path)['preproc_trials'][k]
    Xk = trial.RawData.EegData
    wk = trial.Envelope.subband_weights
    ck = dir2label(trial.attended_ear)
    yk = trial.Envelope.AudioData # T x channel x 2 (binaual)

    # Evaluate weighted envelope
    yk = eval_weighted_env(yk,wk)

    return {'eeg':Xk, 'env':yk, 'lab':ck}


################## AAD Bertrand Helper Function ######################
def window_XY(X,Y,fs=32, Td=0.500,Tmax=30):
    """
    Window the EEG and the envelopes according to decoding length

    X : EEG (T x Nc_eeg)
    Y : Envelopes (list of Tx1)
    fs : Sample rate (hz)
    Td : Window size (Observation Window) in s
    Tmax : Maximum sample collected before shutdown (Decision Window) in s

    Output:
    Xw : EEG (T,Nc_eeg*Td)
    """

    Xw = []
    Yw = []
    delta = fs
    for t in range(Tmax*fs):

        # We will sample exactly the correct regions of interest
        # We sample from the middle
        Xw.append(X[delta+t:delta+t+int(Td*fs),:].flatten('F'))

    Yw.append(Y[0][delta:delta+int(Tmax*fs)])
    Yw.append(Y[1][delta:delta+int(Tmax*fs)])

    return np.array(Xw),Yw

############ REAL TIME ###########################

def loadup_list(x,x0,Lcap):
    """
    Sequentially (X[t-L],X[t-L+1],....X[t]) generate vector of outputs
    if already full, pop first value and append
    """
    y = x
    if len(y) == Lcap:
        _ = y.pop(0)
    
    y.append(x0)
    
    return y
        


def call_newdata(X,Y,idx,fs=32,Td=0.500):
    """
    Call for new data 
    
    X : EEG (T x Nc_eeg)
    Y : Envelopes (list of Tx1)
    fs : Sampling Rate
    idx : index (scalar)
    Td : Window size (Observation Window) in s

    Output:
    Xw : EEG (Nc_eeg*Td,1)
    """
    
    yout = [Y[0][idx],Y[0][idx]]
    xout = X[idx:idx+int(Td*fs),:].flatten('F')
    
    return xout,yout

def eval_error(A,X,Y,N):
    e = 0 
    for i in range(N):
        x,y = call_newdata(X,Y,idx=i,fs=32,Td=0.500)
        e += abs(y[0] - x[:,np.newaxis].T @ A)
    return e/N

def maxcor(yhat,y0,y1):
    c0 = abs(np.dot(yhat/np.linalg.norm(yhat),y0/np.linalg.norm(y0)))
    c1 = abs(np.dot(yhat/np.linalg.norm(yhat),y1/np.linalg.norm(y1)))
    
    if c0 >= c1:
        return 0,c0
    return 1,c1

def RLS_AAD(X,Y,lambd=0.5,delta=1/2,up_rate = 10):
    # Assume we already know label so take it easy
    # Hyperparam
    N = 10000
    Td = 0.5
    fs = 32
    Nd = int(64*fs*Td)
    Lcap = 960
    # RLS
    Ahist = []
    lmat = 1/lambd * np.identity(Nd)
    
    # Initialize P and Q
    P = 1/delta * np.identity(Nd)
    A = np.random.rand(Nd,1)
    
    # Initialize the AAD list vector
    yhat_list = []
    y0_list = []
    y1_list = []
    
    # Forloop
    for i in range(N):
        # Sample and evaluate pred
        x,y = call_newdata(X,Y,idx=i,fs=32,Td=0.500)
        yhat = (x[:,np.newaxis].T @ A).item()
        
        #Load up the list
        yhat_list = loadup_list(yhat_list,yhat,Lcap)
        y0_list = loadup_list(y0_list,y[0].item(),Lcap)
        y1_list = loadup_list(y1_list,y[1].item(),Lcap)
        
        #Evaluate the max_corr from current state
        idx,c = maxcor(np.array(yhat_list),np.array(y0_list),np.array(y1_list))
        
        # Update K
        K = 1/(lambd + x[:,np.newaxis].T @ P @ x[:,np.newaxis]) * P @ x[:,np.newaxis]
        # Update A using current label
        A = A + K @ (y[idx] - x[:,np.newaxis].T @ A)
        
        if i%10 == 0:
            print(idx,c)