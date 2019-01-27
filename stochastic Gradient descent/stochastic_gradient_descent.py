import numpy as np
import matplotlib.pyplot as plt
import random

def cost(m,b):
    global points
    N = float(len(points))
    error = 0
    for point in points:
        error += (point[1]-(m*point[0] + b)) ** 2
    return error / N

points = np.genfromtxt("data.csv", delimiter=",")
data_X = [p[0] for p in points]
data_Y = [p[1] for p in points]

def step_gradient(m,b,points, alpha = 0.00005):
    N = float(len(points))
    m_descend = 0.0
    b_descend = 0.0
    for point in points:
        m_descend += -(2/N) * point[0]*(point[1]-((m*point[0]) + b))
        b_descend += -(2/N) * (point[1]-((m*point[0]) + b))
    m_descend = m - (m_descend * alpha) 
    b_descend = b - (b_descend * alpha)
    return (m_descend, b_descend)

iterations_number = 100
batch_size = 100
precision = 0.00001
m_old = -2
b_old = 0
    
parameter_its = [(m_old, b_old)]
for iteration in xrange(0, iterations_number):
    B = np.random.randint(10,size=batch_size) # Sampling dataset
    
    m_new,b_new = step_gradient(m_old, b_old, points[B,:])
    parameter_its.append((m_new,b_new))
    if (abs(m_old-m_new)) < precision:
	plt.title("y=%s*x+%s" % (round(m_new,4),round(b_new,4))
        break
    m_old = m_new
    b_old = b_new
    print "Cost = ",cost(m_old,b_old),"-","(",m_old,",",b_old,")"
print(m_new, b_new)

plt.ion()

def plot_points_and_modelfit(m,b,iteration=0):
    plt.clf()
    abline = []
    for x in data_X:
        abline.append(m*x+b)

    plt.scatter(data_X,data_Y)
    plt.plot(data_X, abline, 'b')
    plt.show()
    plt.pause(0.05)
        
for parameter in parameter_its:
    plot_points_and_modelfit(parameter[0],parameter[1])
