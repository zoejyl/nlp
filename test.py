import numpy as np
from scipy.special import logsumexp
import random
from a3_gmm import *
import string
from a3_levenshtein import *
#from a3_gmm_2 import *

if __name__ == '__main__':
	myTheta = theta('a',2,3)
	myTheta.omega = np.array([[0.5],[0.5]])
	myTheta.Sigma = np.array([[3.0,2.0,3.0],[1.0,2.0,1.0]])
	myTheta.mu = np.array([[2.0,2.0,2.0],[2.0,2.0,2.0]])
	x = np.array([1.0,2.0,3.0])
	#for i in range(myTheta.omega.shape[0]):
		#sigma_i = myTheta.Sigma[i]
		#mu_i = myTheta.mu[i]
		#pre =0.5*np.sum(np.log(sigma_i))+ np.sum(0.5*np.multiply(np.power(mu_i,2),np.power(sigma_i,-1))) + np.true_divide(mu_i.shape[0],2)*np.log(2*np.pi) 
		#precomputed.append(pre)
	#mu = myTheta.mu
	#sigma = myTheta.Sigma
	#precomputed2 =  0.5*np.sum(np.log(sigma),axis = 1) +np.sum(0.5*np.multiply(np.power(mu,2),np.power(sigma,-1)),axis = 1) + np.true_divide(mu.shape[1],2)*np.log(2*np.pi) 
	#print(precomputed[0] == precomputed2[0])

	#precomputed = np.sum(0.5*np.multiply(np.power(mu_m,2),np.power(sigma_m,-1))) + np.true_divide(mu_m.shape[0],2)*np.log(2*np.pi) + 0.5*np.sum(np.log(sigma_m))
	#r = log_b_m_x(1,x,myTheta,[1,precomputed])
	#print(r)
	#a = np.power(x - mu_m,2)
	#b = np.sum(-0.5*np.true_divide(a,sigma_m))
	#c = np.power(np.prod(sigma_m),0.5)
	#d = np.true_divide(np.exp(b),np.power((2*np.pi),1.5)*c)
	#print(np.log(d))
	#logb0 = log_b_m_x(0,x,myTheta,precomputed)
	#logb1 = log_b_m_x(0,x2,myTheta,precomputed)
	#print(logb0)
	#print(logb1)
	a = Levenshtein("who is there".split(), "".split())
	print(a)
	
	


