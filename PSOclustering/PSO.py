from random import randint, uniform
from copy import deepcopy
from sys import argv
import numpy as np

def Trimdata (inputfile):
	f = open(inputfile,"r")
	x, y = [], []
	for line in f:
		lin = line.strip().split()
		x.append(map(float,lin[:-1]))
		y.append(lin[-1])
	return x,y

class Particle (object):
	def __init__(self,x,k):
		self.position = []
		for i in range (0,k):
			point = x[randint(0,len(x)-1)]
			while point in self.position:
				point = x[randint(0,len(x)-1)]
			self.position.append(point)
		self.velocity = [[0 for coord in cluster] for cluster in self.position]
		self.best = deepcopy(self.position)
		self.bestfit = 0

	#assign each point to a cluster
	def assign(self,x):
		output = []
		for point in x:
			distance = []
			for index,cluster in enumerate(self.position):
				distance.append(np.linalg.norm(np.array(point)-np.array(cluster)))
			output.append(np.argmin(distance))
		return output

	#indexes of points that belong to a given cluster
	def elements (self,cluster,output):
		return np.where(np.array(output)==cluster)[0]

	#intracluster distance (between the points belonging to same cluster)
	def intracluster (self,x,output):
		intra = []
		for index,cluster in enumerate(self.position):
			xi = self.elements(index,output)
			dmax = 0
			for m,point1 in enumerate(xi):
				for point2 in xi[m+1:]:
					d = np.linalg.norm(np.array(x[point1])-np.array(x[point2]))
					if d > dmax:
						dmax = d
			intra.append(dmax)
		return intra

	#intercluster distance for all clusters (between the points belonging to different clusters)
	def intercluster (self):
		inter = []
		for index,cluster1 in enumerate(self.position):
			for cluster2 in self.position[index+1:]:
				inter.append(np.linalg.norm(np.array(cluster1)-np.array(cluster2)))
		return inter

	def fitness (self,x): #Fitness function
		return min(self.intercluster())/max(self.intracluster(x,self.assign(x)))

def PSOPopulationInit (npart,x,k):
	return [Particle(x,k) for i in range (0,npart)]

def PSO (npart,k,in_max,in_min,c1,c2,maxit,inputfile):
	x,y = Trimdata(inputfile) #trimming the dataset
	swarm = PSOPopulationInit(npart,x,k) #initializing population
	fit = [particle.fitness(x) for particle in swarm]
	for particle,value in zip(swarm,fit):
		particle.bestfit = deepcopy(value) #p-best
	rho1 = [uniform(0,1) for i in range(0,len(x[0]))] #rand1
	rho2 = [uniform(0,1) for i in range(0,len(x[0]))] #rand2
	for i in range (0,maxit):
		inertia = (in_max-in_min)*((maxit-i+1)/maxit)+in_min #prob of movement of particle (w)
		
		fit = [particle.fitness(x) for particle in swarm]
		gbest = deepcopy(swarm[np.argmax(fit)]) #g-best (max of all fitness)
		#update best
		for index,particle in enumerate(swarm):
			if fit[index] > particle.bestfit:
				particle.best = deepcopy(particle.position)
				particle.bestfit = deepcopy(fit[index])
		#update velocity and position
		for particle in swarm:
			#V(i+1) = w * Vi + c1 * rand1 * (pbest - Xi) + c2 * rand2 * (gbest - Xi)
			particle.velocity = [map(float,j) for j in inertia*np.array(particle.velocity) + c1*np.array(rho1)*np.array(np.array(particle.best)-np.array(particle.position)) + c2*np.array(rho2)*np.array(np.array(gbest.position)-np.array(particle.position))]
 			#X(i+1) = Xi + Vi
			particle.position = [map(float,j) for j in np.array(particle.position)+np.array(particle.velocity)] 

		print "After Iter %s" % (i+1) ,
		print "\tFitness = %s\n" % gbest.fitness(x) 
		
	fit = [particle.fitness(x) for particle in swarm]
	#best so far
	if max(fit) > gbest.fitness(x):
		gbest = swarm[np.argmax(fit)]

	#return best cluster
	print "\n\nAnd Best Cluster Position is : \n"
	return gbest.position

if __name__ == '__main__':
	a = int(raw_input("Enter No. of Particles "))
	b = int(raw_input("Enter No. of clusters "))
	c = float(raw_input("Enter Max Inertia(0.9) "))
	d = float(raw_input("Enter Min Inertia(0.5) "))
	e = float(raw_input("Enter the (C1) cognitive factor(give 1.5) "))
	f = float(raw_input("Enter the (C2) social factor(give 1.5) "))
	g = int(raw_input("Enter no. of Itterations "))
	h = raw_input("Enter the file name of data set ")
	print PSO(a,b,c,d,e,f,g,h)
