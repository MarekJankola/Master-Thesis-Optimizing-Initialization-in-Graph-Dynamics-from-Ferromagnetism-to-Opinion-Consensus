#SA procedure for finding the strategic initializations leading to global concensus on RRG.
import numpy as np
import networkx as nx
import copy

rng = np.random.default_rng()

#we want to store neighboring nodes in a np.array (so nxd array)
def neighbours(G):
    N=np.ones([n,d])
    for i in range(n):
        count=0
        for k in G.neighbors(i):
            N[i][count]=k
            count+=1
    return N.astype(int)     #.astype(int) convertes float values in the array to int values, which we need because we want to use N values as indices

def onestep_majority(N,s0):
    sums=np.sum(s0[N], axis=1)        #in majority/minority dynamics we just need sums of neighbour values for each node
    return ((1 - np.abs(np.sign(sums)))*s0 + np.sign(sums)) #majority dynamics, always-stay

#returns endstate of node values
def s_endstate(N,s0,p,c):
    for k in range(p+c-1):
        s0=onestep_majority(N,s0)
    return(s0)

def E(N,s0,a,b,p,c):
    s_end=s_endstate(N,s0,p,c)
    return((a*np.sum(s0) - b*np.sum(s_end))/n)

def E_delta(N,s0,a,b,p,c,i):    #E(s')-E(s)
    s_end1=s_endstate(N,s0,p,c)
    s0_copy=copy.deepcopy(s0) #so we still have original s0  
    s0_copy[i]=-s0[i]
    s_end2=s_endstate(N,s0_copy,p,c)
    return((-2*a*s0[i] + b*(np.sum(s_end1) - np.sum(s_end2)))/n)

def m(s):
    return(np.sum(s)/n)



n=10000
d=4
p=3
c=1    #***we end the MCMC alg when we reach all ones state after p steps -which is c=1 attractor***

par_a=1.0005  #this is the constant with which we increase "temperatures" a,b
par_b=1.0005

N_stat=5 #number of repetitions
mag_reached=np.zeros(N_stat)
num_steps=np.zeros(N_stat) #number of steps needed for convergence, we stop the algorithm either if it reached sought for m_init(all hom attr) or after a treshold of 2*n^3 steps
conf=np.zeros((N_stat,n))
graphs=np.zeros((N_stat,n,d))

for k in range(N_stat):
    G=nx.random_regular_graph(d,n) #in Networkx documentation (https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.random_regular_graph.html)
    #in notes they say that the algorithm samples in an asymptotically uniform way from the space of random graphs 
    N=neighbours(G)
    graphs[k]=N
    
    #next we initialize node values uniformly at random, in a way where ith element of s0 is the value of ith node in G 
    s = 2*np.random.binomial(n=1, p=0.5, size=[n])-1
    
    a=0.015*n
    b=0.01*n

    t=0
    m_final=m(s_endstate(N,s,p,c))
    while(m_final<1):   
        i = np.random.randint(low=0, high=n)
        delta_H = E_delta(N,s,a,b,p,c,i)
        prob_accept = min([1, np.exp(-delta_H)])
        if (np.random.rand() < prob_accept):
            s[i]=-s[i]
        
        #simmulated annealing part:
        if a<4.5*n: a=par_a*a
        if b<5*n: b=par_b*b
            
        t+=1
        if t>(2*n**3): m_final=2
        else: m_final=m(s_endstate(N,s,p,c))
    mag_reached[k]=m(s)
    num_steps[k]=t
    conf[k]=s

graphs=graphs.astype(int)
        
#np.savez("MCMC_p3_d4.npz", mag_reached=mag_reached, num_steps=num_steps,conf=conf,graphs=graphs)

