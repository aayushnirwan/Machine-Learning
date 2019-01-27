#PYTHON-2.7

from random import randint, uniform
from copy import deepcopy
from sys import argv
import numpy as np

def BinSearch(prob,p,imin,imax):
	imid = (imin+imax)/2
	if imid == len(prob)-1 or imid == 0:
		return imid
	if p > prob[imid] and p <= prob[imid+1]:
		return imid+1
	elif p < prob[imid]:
		imid = BinSearch(prob,p,imin,imid)
	else:
		imid = BinSearch(prob,p,imid,imax)
	return imid

def Inputs (arqStr):
	arq = open(arqStr,"r")
	x, y = [], []
	for line in arq:
		lin = line.strip().split()
		x.append(map(float,lin[:-1]))
		y.append(lin[-1])
	#print x,y
	return x,y

class Individual (object):
	def __init__(self,x,k,genes):
		self.genes = genes
		if x!= None:
			for i in range (0,k):
				point = x[randint(0,len(x)-1)]
				for coord in point:
					self.genes.append(coord)
			self.dim = len(x[0])
		else:
			self.dim = len(genes)/k

	#assign each point to a cluster
	def assign(self,x):
		output = []
		for point in x:
			distance = []
			for index in range (0,len(self.genes)/self.dim):
				distance.append(np.linalg.norm(np.array(point)-np.array(self.genes[index*self.dim:(index+1)*self.dim])))
			output.append(np.argmin(distance))
		return output

	#indexes of points that belong to a given cluster
	def elements (self,cluster,output):
		return np.where(np.array(output)==cluster)[0]

	#update clusters centers based on assignments
	def update (self,x,output):
		for index in range (0,len(self.genes)/self.dim):
			xi = self.elements(index,output)
			for d in range(index*self.dim,(index+1)*self.dim):
				self.genes[d] = sum([x[item][d%self.dim] for item in xi])/len(xi) if len(xi)!=0 else self.genes[d]
	
	#intracluster distance (between the points belonging to same cluster)
	def intracluster (self,x,output):
		intra = []
		for index in range(0,len(self.genes)/self.dim):
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
		for index in range(0,len(self.genes)/self.dim):
			for j in range (index+1,len(self.genes)/self.dim):
				inter.append(np.linalg.norm(np.array(self.genes[index*self.dim:(index+1)*self.dim])-np.array(self.genes[j*self.dim:(j+1)*self.dim])))
		return inter

	def fitness (self,x): #Fitness function 
		output = self.assign(x)
		self.update(x,output)
		return min(self.intercluster())/max(self.intracluster(x,self.assign(x)))

	def mutation (self,pmut): #Single point Mutation
		for g,gene in enumerate(self.genes):
			if uniform(0,1) <= pmut:
				delta = uniform(0,1)
				if uniform(0,1) <= 0.5:
					self.genes[g] = gene - 2*delta*gene if gene!=0 else -2*delta
				else:
					self.genes[g] = gene + 2*delta*gene if gene!=0 else 2*delta

def GAPopulationInit (npop,x,k): #Inintilizing Population
	return [Individual(x,k,[]) for i in range (0,npop)]

def Crossover (parent1, parent2,k): #Single Point Crossover
	point = randint(1,len(parent1.genes)-2)
	return Individual(None,k,parent1.genes[:point]+parent2.genes[point:]), Individual(None,k,parent2.genes[:point]+parent1.genes[point:])

def RouletteWheel (pop,fit):	#Selection Method
	sumf = sum(fit)
	prob = [(item+sum(fit[:index]))/sumf for index,item in enumerate(fit)]
	return pop[BinSearch(prob,uniform(0,1),0,len(prob)-1)]

def GeneticAlgorithm (npop,k,pcros,pmut,maxit,arqStr):
	x,y = Inputs(arqStr)
	pop = GAPopulationInit(npop,x,k)
	fit = [indiv.fitness(x) for indiv in pop]
	verybest = [pop[np.argmax(fit)],max(fit)]
	for i in range(0,maxit):
		print "After Iter %s " % (i+1) ,
		fit = [indiv.fitness(x) for indiv in pop]
		new = []
		while len(new) < len(pop):
			#selection
			parent1 = RouletteWheel(pop,fit)
			p = uniform(0,1)
			#genetic operators
			if p <= pcros:
				parent2 = RouletteWheel(pop,fit)
				while parent2 == parent1:
					parent2 = RouletteWheel(pop,fit)
				child1, child2 = Crossover(parent1,parent2,k)
				new.append(child1)
				if len(new) < len(pop):
					new.append(child2)
			else:
				child = deepcopy(parent1)
				child.mutation(pmut)
				new.append(child)
		pop = deepcopy(new)
		#elitism (but individual is kept outside population)
		if max(fit)>verybest[1]:
			verybest = [pop[np.argmax(fit)],max(fit)]
		print "\tFitness Value = %s\n" % verybest[1]
	
	#return best cluster
	print "\n\nAnd Best Cluster is : \n"
	return verybest[0].genes

a = int(raw_input("Enter Population "))
b = int(raw_input("Enter no. of clusters "))
c = float(raw_input("Enter Probability of crossover (0.85) "))
d = float(raw_input("Enter Probability of mutation (0.01) "))
e = int(raw_input("Enter no. of Itterations "))
f = raw_input("Enter the file name of data set ")
print GeneticAlgorithm(a,b,c,d,e,f)
