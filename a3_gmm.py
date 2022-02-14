from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/Users/Zoe/Documents/University/2018WINTER/CSC401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d)) 
        self.Sigma = np.ones((M,d)) 


#def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    #mu_m = myTheta.mu[m]
    #sigma_m = myTheta.Sigma[m]
    #result = np.sum(-0.5*np.multiply(np.power(x,2),np.power(sigma_m,-1)))+ np.sum(np.multiply(mu_m,np.multiply(x,np.power(sigma_m,-1))))
    #result = result - preComputedForM[m]
    #return result 

def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    mu_m = myTheta.mu[m]
    sigma_m = myTheta.Sigma[m]
    result = -0.5*np.dot(np.power(x,2),np.power(sigma_m,-1))+ np.dot(x,np.multiply(mu_m,np.power(sigma_m,-1)))
    if len(preComputedForM) != 0:
        result = result - preComputedForM[m]
    return result 
    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    mu = myTheta.mu
    sigma = myTheta.Sigma
    precomputed = 0.5*np.sum(np.log(sigma),axis = 1) +np.sum(0.5*np.multiply(np.power(mu,2),np.power(sigma,-1)),axis = 1) + np.true_divide(mu.shape[1],2)*np.log(2*np.pi) 
    shape = myTheta.omega.shape[0]
    omegas = myTheta.omega.reshape(shape)
    bs = []
    for i in range(shape):
        bi = log_b_m_x(i,x,myTheta,precomputed)
        bs.append(bi)
    bs = np.array(bs,dtype = np.float64)
    logwmbm = np.log(omegas)+bs
    logsum = logsumexp(logwmbm)
    result = np.log(omegas[m]) + bs[m] - logsum

    return result,myTheta

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    shape = myTheta.omega.shape[0]
    omegas = myTheta.omega.reshape(shape)
    logwmbm = log_Bs.T+np.array([np.log(omegas)])
    logsum = logsumexp(logwmbm,axis = 1)
    likelihood = np.sum(logsum)
    return likelihood

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    myTheta = theta( speaker, M, X.shape[1] )
    
    #Initialization
    r = 1/M
    for i in range(M):
        myTheta.omega[i][0] = r
    T = X.shape[0]
    a = np.arange(T,dtype = np.int64)
    ind =np.random.choice(a,size = M,replace = False)
    for i in range(M):
        myTheta.mu[i] = X[ind[i]]
    pre_L = -np.inf
    improvement = np.inf
    iteration = 0
    while iteration <= maxIter and  improvement >= epsilon:  
        mu = myTheta.mu
        sigma = myTheta.Sigma
        precomputed = 0.5*np.sum(np.log(sigma),axis = 1) +np.sum(0.5*np.multiply(np.power(mu,2),np.power(sigma,-1)),axis = 1) + np.true_divide(mu.shape[1],2)*np.log(2*np.pi) 
        log_Bs = np.zeros((M,T))
        for i in range(M):
            log_Bs[i] = log_b_m_x(i,X,myTheta,precomputed)
        log_Ps = np.zeros((M,T))
        for m in range(M):
            for t in range(T):
                logp,my_theta = log_p_m_x( m, X[t], myTheta)
                log_Ps[m][t] = logp
        loglike = logLik(log_Bs,myTheta)
        sump = np.exp(logsumexp((log_Ps),axis = 1))
        omegas = np.exp(logsumexp((log_Ps),axis = 1) - np.log(T))
        myTheta.omega = omegas.reshape((M,1))
        logpexp = np.exp(log_Ps)
        mus = np.true_divide(np.dot(logpexp,X),sump.reshape(M,1))
        myTheta.mu = mus
        sigmas = np.true_divide(np.dot(logpexp,np.power(X,2)),sump.reshape(M,1)) - np.power(mus,2)
        # set floor for sigmas, to prevent singularity
        for i in range(M):
            for j in range(X.shape[1]):
                if sigmas[i][j] <= 0.00001 :
                    sigmas[i][j] += 0.00001
        myTheta.Sigma = sigmas
        improvement = loglike - pre_L
        pre_L = loglike
        iteration += 1
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    loglike = - np.inf
    M = models[0].mu.shape[0]
    logliks = np.zeros(len(models))
    for i in range(len(models)):
        log_Bs = np.zeros(M)
        precomputed = []
        mu = models[i].mu
        sigma = models[i].Sigma
        precomputed = 0.5*np.sum(np.log(sigma),axis = 1) +np.sum(0.5*np.multiply(np.power(mu,2),np.power(sigma,-1)),axis = 1) + np.true_divide(mu.shape[1],2)*np.log(2*np.pi)
        log_Bs = np.zeros((M,mfcc.shape[0]))
        for m in range(M):
            logbm = log_b_m_x( m,mfcc,models[i],precomputed)
            log_Bs[m] = logbm
        newloglike = logLik(log_Bs,models[i])
        logliks[i] = newloglike
        if newloglike > loglike:
            loglike = newloglike
            bestModel = i


    actual_ID = models[correctID].name
    bestkindex = np.argsort(logliks)[::-1][:k]
    print(actual_ID)
    for ind in range(k):
        print(models[bestkindex[ind]].name,logliks[bestkindex[ind]])



    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 3
    epsilon = 0.0
    maxIter = 4
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)

