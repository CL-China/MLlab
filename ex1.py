import math
import numpy
import numpy.matlib
from matplotlib import pyplot as plt

def gen_sinusoidal(N):
	x = []
	t = []
	for i in range(N):
		x.append(2*math.pi*(i+1)/N)
		t.append(numpy.random.normal(math.sin(x[i]),0.04))
	return (x,t)


def fit_polynomial(x, t, M):
	length = len(x)
	phi = []
	for i in range(length):
		phi.append([])
		for j in range(M):	
			phi[i].append(x[i]**j)
	phi = numpy.matrix(phi);
	w = numpy.linalg.inv(numpy.transpose(phi) *phi) *numpy.transpose(phi)  		
	return w*numpy.transpose(numpy.matrix(t))

def fit_predict(x, w, M):
	length = len(x)
	phi = []
	for i in range(length):
		phi.append([])
		for j in range(M):	
			phi[i].append(x[i]**j)
	phi = numpy.matrix(phi);
	return numpy.array(phi*w)

def run(N):
	(x,t) = gen_sinusoidal(N)
	j = 0
	for i in [0,1,3,9]:
		w = fit_polynomial(x,t,i)
		y = fit_predict(x,w,i)
		plt.subplot(1,4,j+1)
		plt.plot(x,y)
		plt.plot(x,t)
		j += 1
	plt.show()



def fit_polynomial_reg(x, t, M, lamb):
	length = len(x)
	phi = []
	for i in range(length):
		phi.append([])
		for j in range(M):	
			phi[i].append(x[i]**j)
	phi = numpy.matrix(phi);
	w = numpy.linalg.inv(lamb * numpy.identity(M) + numpy.transpose(phi) *phi) *numpy.transpose(phi)  		
	return w*numpy.transpose(numpy.matrix(t))


def kfold_indices(N, k):
	all_indices = numpy.arange(N,dtype=int) 
	numpy.random.shuffle(all_indices)
	idx = numpy.floor(numpy.linspace(0,N,k+1)) 
	train_folds = []
	valid_folds = []
	for fold in range(k):
		valid_indices = all_indices[idx[fold]:idx[fold+1]] 
		valid_folds.append(valid_indices) 
		train_folds.append(numpy.setdiff1d(all_indices, valid_indices))
	return train_folds, valid_folds


def cross_validation():
	(x, t) = gen_sinusoidal(9)
	(train_folds, valid_folds) = kfold_indices(9,9)
	error = []
	for M in range(11):
		for j in range(11):
			error1 = []
			for i in range(9):
				xnew = []
				tnew = []
				for k in train_folds[i]:
					xnew.append(x[k])
					tnew.append(t[k])
				w = fit_polynomial_reg(xnew,tnew,M,math.exp(-j))
				xnew = []
				tnew = []
				for k in valid_folds[i]:
					xnew.append(x[k])
					tnew.append(t[k])
				y = fit_predict(xnew, w, M)
				error1.append((y-tnew)**2)			
			error.append(sum(error1)/len(error1))
	i = error.index(min(error))
	index_j = i % 11
	index_M = i//11
	error = numpy.squeeze(error)
	error = numpy.reshape(error,(11,11))
	for i in range(11):
		plt.plot(error[i],label='$M={i}$'.format(i=i))
	plt.xlabel('Lamb')
	plt.ylabel('Error')
	plt.axis([-0.5, 10.5, -0.2, 18])
	plt.legend(loc='best')
	plt.show()
	return index_M, math.exp(-index_j)



def gen_sinusoidal2(N):
	x = []
	t = []
	for i in range(N):
		x.append(numpy.random.uniform(2*math.pi))
		t.append(numpy.random.normal(math.sin(x[i]),0.04))
	return (x,t)
 

def fit_polynomial_bayes(x, t, M, alpha, beta):
	length = len(x)
	xrep = numpy.matlib.repmat(x,M,1)
	for i in range(M):
		xrep[i,:] = numpy.power(xrep[i,:],i)
	phi = numpy.transpose(numpy.matrix(xrep))
	s = numpy.linalg.inv(alpha + beta*numpy.transpose(phi)*phi)
	m = beta*s*numpy.transpose(phi)*numpy.transpose(numpy.matrix(t))  		
	return m,s


def predict_polynomial_bayes(x, m,s,beta):
	length = len(x)
	M = len(m)
	xrep = numpy.matlib.repmat(x,M,1)
	for i in range(M):
		xrep[i,:] = numpy.power(xrep[i,:],i)
	phi = numpy.transpose(numpy.matrix(xrep))
	phi = numpy.matrix(phi)
	y = phi*m
	sigma = 1/beta + phi*s*numpy.transpose(phi)
	return y,sigma

def Q_24():
	(x,t) = gen_sinusoidal2(7)
	M = 5
	alpha = 0.5
	beta = 1/0.2**2
	(m,s) = fit_polynomial_bayes(x,t,M,alpha,beta)
	x = numpy.arange(0,2*math.pi,0.01)
	(y,sigma) = predict_polynomial_bayes(x,m,s,beta)
	x = numpy.array(x)
	y = numpy.array(y)
	sg = numpy.array(sigma.diagonal())
	plt.subplot(2,1,1)
	plt.fill_between(x,y[:,0],sg[0,])
	'''plt.subplot(1,2,1)
	plt.plot(x,y,'ro')
	plt.subplot(1,2,2)
	plt.plot(x,t,'*')'''
	#plt.show()
	w = numpy.random.multivariate_normal(m.flat,s,100)
	x = numpy.arange(0,5,0.1)
	x = numpy.matlib.repmat(x,M,1)
	x = numpy.matrix(x)
	for i in range(M):
		x[i,:] = numpy.power(x[i,:],i)
	w = numpy.matrix(w)
	y = w*x
	plt.subplot(2,1,2)
	x = x[1,]
	for i in range(100): 
		plt.plot(x.flat,y[i,].flat)
	plt.show()


















