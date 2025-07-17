import numpy as np
import torch
import networkx as nx
import itertools
import random
import copy
import time

rng = np.random.default_rng()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

#Ising ferromagnet/majority rule, always stay tie breaking
def atr_condition(xi,xj,rho,p,c):
    if xi[p]==torch.sign(rho[p+c-1]+xj[p+c-1]): return(1)
    elif (rho[p+c-1]+xj[p+c-1])==0 and xi[p]==xi[p+c-1]: return(1)
    else: return(0)
        
def traj_condition(xi,xj,rho,p,c):
    prod=1
    for t in range(0,p+c-1):
        #if x_i[t+1]==-1*np.sign(np.sum(neighbours[:,t])): continue    #minority dynamics
        if xi[t+1]==torch.sign(rho[t]+xj[t]): continue    #majority dynamics
        elif (rho[t]+xj[t])==0 and xi[t+1]==xi[t]: continue #always-stay tie breaking
        #elif np.sum(neighbours[:,t])==0 and x_i[t+1]==-1*x_i[t]: continue #always-change tie breaking
        else: 
            prod=0
            break
    return(prod)

def m_attr_i(x_i,p,c):
    return(torch.sum(x[p:])/c)

def attr_fix(x_i,p,c,attr_value):
    if x_i[p+c-1]==attr_value: return(1)
    else: return(0)

def A_i_sums(xi,xj,rho,p,c,attr_value,lmbd_in):
    return(torch.exp(-lmbd_in*xi[0]/n)*atr_condition(xi,xj,rho,p,c)*traj_condition(xi,xj,rho,p,c)*attr_fix(xi,p,c,attr_value))

#we create edge dictionary with the positions
def order_edge(edge, edge_dict):
    # if we have edge as an array, we need it to convert to tuple: tuple(edge)
    return edge_dict.get(tuple(edge)) 

def order_gpu(x_i, x_j, p, c):
    # changing minus ones to zeros, so we can get number in binary
    #bin_x_i=((x_i+1)//2).astype(int) 
    #bin_x_j=((x_j+1)//2).astype(int)
    #bin_x_i=((x_i+1)/2).int() 
    #bin_x_j=((x_j+1)/2).int() 
    # we concatenate the two trajectories
    #bin_comb = torch.cat((bin_x_i, bin_x_j))
    bin_comb = torch.cat((x_i, x_j))
    # Convert binary tensor to a single integer
    bin_string = ''.join(map(str, bin_comb.tolist()))  # Convert to string
    binary_value = int(bin_string, 2)  # Parse binary string as an integer

    # Compute the final value
    result = num_combs - 1 - binary_value
    return result



#the following is a function, which gives an order of a particular combination of trajectories x_i and x_j for mes chi^ij (we suppose that intertools creates all pos combinations of trajecotires in unchanging order -list and positions in it numbers the particular combination, e.g.: for p=1 c=1: we have combinations like (1,1),(1,1) <->0,(1,1),(1,-1) <->1 ...)
def order(x_i, x_j, p, c):
    # changing minus ones to zeros, so we can get number in binary
    #bin_x_i=((x_i+1)//2).astype(int) 
    #bin_x_j=((x_j+1)//2).astype(int)
    bin_x_i=((np.array(x_i)+1)/2).astype(int) 
    bin_x_j=((np.array(x_j)+1)/2).astype(int)
    # we concatenate the two trajectories
    bin_comb = np.concatenate((bin_x_i, bin_x_j))
    return (num_combs-1 - int("".join(map(str, bin_comb)), 2)) # Interpret the binary sequence as an integer(changing from array to scalar and then do decimal)
             #2**(2*(p+c))-1 - int() because we want all ones correspond to 0
#maybe I dont even want to convert to decimal (and maybe instead of minus ones I want to work with zeros? but then -1 are good for trajectory and atr checks) ??


#we want to store positions of the neighboring edges as in the chi_col system(for every edge (incl reversed) we have 2**2(p+c) mes comb) (after the end of G.edges we cosider list of reversed edges)
#so when the G.edges go as (i1,j1),(i2,j2)... we asign (i1,j1) row of positions (in this list) of edges (k1,i1), (k2,i1)... for k in the neib. of i1 except for j1!   
def neib_edg_pos_chi_mat(G):
    N_edg_pos_chi_mat=np.ones((2*num_edg,d-1), dtype=np.int32) 
    
    for idx,edge in enumerate(G.edges):
        count=0
        for k in G.neighbors(edge[0]):
            if k!=edge[1]:
                N_edg_pos_chi_mat[idx][count] = order_edge((k,edge[0]), edge_dict)
                count+=1
        
        #reversed edges:
        count=0
        for k in G.neighbors(edge[1]):
            if k!=edge[0]:
                N_edg_pos_chi_mat[idx+num_edg][count] = order_edge((k,edge[1]), edge_dict)
                count+=1
    return N_edg_pos_chi_mat.astype(int)   


#random uniform initialization of messages
def mes_init_mat(num_edg,p,c):
    chi_mat=torch.rand((2*num_edg,num_combs),device=device)
    return(chi_mat/torch.sum(chi_mat,axis=1, keepdims=True))


#now we want a matrix of true marginals ((n,2)), where for each node i we have probability of x_i^0=+-1 (0th column for +1 and first column for -1)
#we need positions of neighbouring edges for each node (for every node we have d positions of neighboring  UNORDERED edges):

#returns ALSO neighboring nodes (for i its d neighbors) 
def neighb_edges_pos_AND_nodes(G):
    N_edges_pos=np.ones((n,d), dtype=np.int32) 
    N_nodes=np.ones((n,d), dtype=np.int32)
    
    for i in range(n):
        for idx,k in enumerate(G.neighbors(i)):
            N_edges_pos[i][idx]= order_edge((i,k),edge_dict)/num_combs     #order_edge=order_edge_plain (just order/index in G.edges)*num_combs
            N_nodes[i][idx]=k    
    return np.array(N_edges_pos).astype(int), N_nodes.astype(int) 

def positions_biases(nodes,n,p,c,num_edg):#nodes=nodes_chi(G)
    pos=np.zeros(2*num_edg*num_combs, dtype=np.int32) 
    for i in range(2*num_edg):
        pos[i*num_combs:int(i*num_combs + num_combs/2)]=nodes[i]          #first num_combs/2 positions are positions of node i (node i is i-th row in biases for +1 and (i+n)th row for -1)
        pos[int(i*num_combs + num_combs/2):i*num_combs + num_combs]=nodes[i]+n
    return(pos.astype(int))

#now we make biases into a chi-like array so we can use them efficiently in the BP reinf. update
def new_biases_chi(biases_i,pos_biases): #pos_biases=positions_biases(nodes,n,p,c,num_edges)
    #reshaped = np.vstack((biases_i[:, 0], biases_i[:, 1])).reshape(-1, 1) #we reshape n,2 biases_i into 1D array in a whay that first we have 1. column and then 2.
    reshaped = torch.vstack((biases_i[:, 0], biases_i[:, 1])).reshape(-1, 1)
    #reshaped=reshaped.flatten()
    reshaped=torch.flatten(reshaped)
    return(reshaped[pos_biases])

#similarily as in cedrics paper, eq. (24)
#gives new/updated values of biases WITH CERTAIN PROBABILITY and also RETURNS NEW POSSIBLE SOLUTION {s}
def new_biases_i(biases_i,pie,gamma,marginals,t):
    T= marginals[:, 1] >= marginals[:, 0]     #this is an array of truth values (True if marg[1]>=marg[0] element wise for the two columns of marginals... ,and False otherwise)
    new_bias=torch.ones((n,2),device=device) #we create bias value for all mes i (n) and for two values of nodes (+-1), we update them, but then need to distribute them in the correct shape for messages
    new_bias[T]=torch.tensor([pie,1-pie],device=device)               # when marginal for -1 is bigger than marginal for +1 for nodes i, biases go for 1-pi fo -1 and pi for +1
    new_bias[~T]=torch.tensor([1-pie,pie],device=device)              #and vice versa in the other/complementary case
    prob=torch.rand(n)<1-(1+t)**(-gamma)
    biases_i[prob]=new_bias[prob]
    s=biases_i[:, 0]>biases_i[:, 1]
    return biases_i,(2*s.int()-1)

def marginals_comp(chi_mat,pairs,pji,N_edges_pos,epsilon=torch.tensor(1e-15,device=device)):
    marg=torch.zeros((n,2),device=device)
    
    ZZ = chi_mat[:num_edg]*chi_mat[num_edg:,pairs]   #((E/2,2**2(p+c)))#this creates, for each edge, products of chi^ij_xixj*chi^ji_xjxi in a way that we have right pairs (xi, xj are really the same in both xixj and xjxi)
    ZZZ = torch.vstack((ZZ,ZZ[:,pji])) #lower half of ZZZ are the same products as above we will sum them in a different way (thats why we shuffled these products with pji)
    #we sum them so the SECOND trajectory starts with +-1, thats because when we have for every "forward" edge ij the som of products of mes, we then do the final product k in neigh. of i, and if we only have calcualted final sum for ki (not ik) edge, in this ki edge only xk^0=+-1 (not xi, that we need), so we also do reversed edges  
    
    Z_plus = torch.sum(ZZZ[:,:int(num_combs/2)],dim=1)
    Z_minus = torch.sum(ZZZ[:,int(num_combs/2):],dim=1) # we sum products for all combinations of trajectory xj and only those (first half/second half) with x_i^0 = +1/-1 (plus/minus)
    #normalization
    Z_plus=torch.maximum(Z_plus,epsilon)
    Z_minus=torch.maximum(Z_minus,epsilon)
    zzz=Z_plus+Z_minus
    Z_plus=Z_plus/zzz
    Z_minus=Z_minus/zzz     #we got sum over all traj comb with xi^0=+-1 of chi^ij*chi^ji

    marg[:,0] = torch.prod(Z_plus[N_edges_pos],dim=1) # now we do the product of these sums for all neighbours of i
    marg[:,1] = torch.prod(Z_minus[N_edges_pos],dim=1) #so marginals[0] will be a vector of marginals for s=+1, [1] is for s=-1
    marg = marg/(marg[:,0] + marg[:,1])[:,None] #normalization

    return(marg)

def onestep_majority(N,s0):
    sums=torch.sum(s0[N], dim=1)        #in majority/minority dynamics we just need sums of neighbour values for each node
    return ((1 - torch.abs(torch.sign(sums)))*s0 + torch.sign(sums)) #majority dynamics, always-stay
    
#returns endstate of node values
def s_endstate(N,s0,p,c):
    for k in range(p+c-1):
        s0=onestep_majority(N,s0)
    return(s0)

def m(s):
    return(torch.sum(s)/n)

#here we have two tensors L and LL, but both without dimension for D
def HPr_dp(chi_mat,chi_col,biases_chi,rho_D1,N_edg_pos_chi_mat,d,p,c,attr_value,lmbd_in,damppar):
    L = torch.zeros(([2*num_edg]+[2]*T+[d]*T),device=device)
    LL = torch.zeros(([2*num_edg]+[2]*T+[d]*T),device=device)
    for xi in xi_comb:
        for rho in rho_D1:
            LL[tuple([torch.arange(2*num_edg)] + list(xi) + list(rho))] =(biases_chi*chi_col)[N_edg_pos_chi_mat[:,0]+order_gpu(rho,xi,p,c)]  
    
    for D in range(2,d): #we go with D from 1(above), 2, 3,... to d-1 (number of neighbours except j)
        for xi in xi_comb:
            unique_set = set()
            for rho in rho_D1:
                for xk in xi_comb:
                    L[tuple([torch.arange(2*num_edg)] + list(xi) + list(rho+xk))] += LL[tuple([torch.arange(2*num_edg)] + list(xi) + list(rho))]*(biases_chi*chi_col)[N_edg_pos_chi_mat[:,D-1]+order_gpu(xk,xi,p,c)] 
                    unique_set.add(tuple((rho+xk).tolist())) #we save only unique new values of the vector sum for new D and in the next for loop

        del rho_D1, LL
        torch.cuda.empty_cache()  # Free GPU memory
    
        rho_D1 = torch.tensor(list(unique_set), device=device).int()
        LL = L.detach() 
        del L 
        torch.cuda.empty_cache()
        L = torch.zeros_like(LL)
 

    #final HPr update
    chi_mat2=torch.zeros((2*num_edg,num_combs),device=device)
    for idxij, xixj in enumerate(itertools.product(xi_comb, repeat=2)):
        for rho in rho_D1:
            chi_mat2[:,idxij] += A_i_sums(2*xixj[0]-1,2*xixj[1]-1,2*rho-d+1,p,c,attr_value,lmbd_in)*LL[tuple([torch.arange(2*num_edg)] + list(xixj[0]) + list(rho))]

    #normalizations of mes + dampening
    chi_mat2 = damppar*chi_mat2/torch.sum(chi_mat2,dim=1, keepdims=True) + (1-damppar)*chi_mat
    
    # +from matrix to column
    return(chi_mat2.reshape(-1), chi_mat2)



#History Passing reinforcement (HPr) PyTorch + GPU version
#PARAMETERS ---------------------------------------------------------------------------------------------------------------------
n=10000
d=4
p=1
c=1

damppar=0.4
attr_value=1
lmbd_in=25*n


#"HPr" parameters
pie=0.3
gamma=0.1
TT=10000

#AUXILIARY VARIABLES ---------------------------------------------------------------------------------------------------------------------
num_edg = int(n*d/2)
num_combs = 2**(2*(p+c))
num_combs_neigh = 2**((p+c)*(d-1))

T=p+c
x=torch.tensor([1,0], dtype=torch.int, device=device)
x_repeat=x.repeat(T,1)
xi_comb=torch.cartesian_prod(*x_repeat) #p+c times

xi_comb_cpu=np.array(list(itertools.product([1, -1], repeat=p+c)))

n_rep=1
mag_reached=torch.zeros(n_rep,device=device)
num_steps=torch.zeros(n_rep,device=device)
conf=torch.zeros((n_rep,n), device=device)
graphs=torch.zeros((n_rep,n,d),device=device)

start = time.time()

for kkk in range(n_rep):
    #RANDOM REGULAR GRAPH ---------------------------------------------------------------------------------------------------------------------
    G=nx.random_regular_graph(d,n)
    
    #AUXILIARY CONSTRUCTIONS ---------------------------------------------------------------------------------------------------------------------
    pairs=np.zeros(num_combs) #positions of message pairs chi^ji (to chi^ij, more about it in the marginals creating function)
    pls=0
    mns=0
    pji=np.zeros(num_combs) #positions of combinations xixj where second trajectory xj^0=+-1, we also need Zji with xi^0 =+-1 (so second trajectory)
    for idx, xixj in enumerate(itertools.product(xi_comb_cpu, repeat=2)):
        pairs[idx] = order(xixj[1],xixj[0],p,c)
        if xixj[1][0]==1:
            pji[pls]=idx
            pls+=1
        else:
            pji[mns+int(num_combs/2)]=idx
            mns+=1
    
    edge_dict = {}
    N_nodes_order=np.zeros(2*num_edg) #as nodes go in the G.edges()
    for idx, edge in enumerate(G.edges):        
        #rev_edg= = edge[::-1]  # Reversed edge
        edge_dict[edge] =idx*num_combs
        edge_dict[edge[::-1]] = (idx+num_edg)*num_combs
        
        N_nodes_order[idx] = edge[0]
        N_nodes_order[idx+num_edg] = edge[1]
        #returns first node in the list of edges, after all edges there is list of second nodes in those edges: if G.edges is (0,7), (0,9),(1,3)... we get array of 0 0 1... and E values we have 7 9 3 ... 
        #we will use this to redistribute biases_i (n,2) into chi-like shape (len(chi),), and since each row i of biases_i corresponds to node i, above we can really just store node values
    
    
    #positions of edges needed for BDCM update
    N_edg_pos_chi_mat=neib_edg_pos_chi_mat(G)
    
    
    #position of neighboring edges for each node i (position in G.edges (same position also for reversed edges)) -needed for marginals computation
    N_edges_pos,N_nodes = neighb_edges_pos_AND_nodes(G)
    
    pos_biases=positions_biases(N_nodes_order,n,p,c,num_edg) #positions of biases in chi-like array
    
    
    D=1
    rho_D11=torch.zeros(((D+1)**T,T),device=device)  
    rho_D11=rho_D11.int()
    for idx, xi in enumerate(xi_comb): 
        rho_D11[idx]=xi
    
    #---------------------------------------------------------------------------------------------------------------------
    
    #TRANSFERING TO TENSORS ON CUDA ---------------------------------------------------------------------------------------------------------------------
    pairs = torch.tensor(pairs,device=device)
    pairs = pairs.int()
    pji = torch.tensor(pji,device=device)
    pji = pji.int()
    
    N_nodes_order = torch.tensor(N_nodes_order,device=device)
    N_nodes_order = N_nodes_order.int()
    N_edg_pos_chi_mat = torch.tensor(N_edg_pos_chi_mat,device=device)
    N_edg_pos_chi_mat = N_edg_pos_chi_mat.int()
    
    N_edges_pos = torch.tensor(N_edges_pos,device=device)
    N_edges_pos = N_edges_pos.int()
    N_nodes = torch.tensor(N_nodes,device=device)
    N_nodes = N_nodes.int()
    
    pos_biases = torch.tensor(pos_biases,device=device)
    pos_biases = pos_biases.int()
    
    #RANDOM INITIALIZATIONS OF MESSAGES, BIASES AND TRIAL SOLUTION ---------------------------------------------------------------------------------------------------------------------
    
    chi_mat=mes_init_mat(num_edg,p,c)   #chi_mat is a matrix of messages of a shape (2E,2**2*(p+c)) so for every edge we have 2 combinations of messages (combinations of trajcetories xi xj) (reversed edges are in the second half, order of rows is given by the order in G.edges)
    chi_col=chi_mat.reshape(-1)         #chi_col is a column of messages, where we have edge ij, all its combinations values and another edge i1 j1 and all its combinations ...(second half for reversed edges) 
    
    #for every trajectory and its combination we have bias b^k_xk, k is for variable node/trajectory in the original graph and xk is particular/concrete trajectory (we take into account just x_k^0 value -initial vlaue of the traj)
    #first we initialize biases at random:
    biases_i=torch.rand((n,2),device=device)
    biases_i=biases_i/torch.sum(biases_i,axis=1,keepdims=True)   #normalization
    
    ttt=biases_i[:, 0]>biases_i[:, 1]    #from biases for +-1 values we get first shot for our solution, s=max_+-1 (b_+1,b_-1)
    s=(2*ttt.int()-1)
    #---------------------------------------------------------------------------------------------------------------------
    
    #HPr
    t=0
    m_final=m(s_endstate(N_nodes,s,p,c))
    while(m_final<1):
        biases_chi = new_biases_chi(biases_i,pos_biases) #we get a vector of biases arranged to chi shape from biases_i
    
        rho_D1 = rho_D11.detach().clone().to(device='cuda')
        chi_col,chi_mat = HPr_dp(chi_mat,chi_col,biases_chi,rho_D1,N_edg_pos_chi_mat,d,p,c,attr_value,lmbd_in,damppar)
        
        marginals = marginals_comp(chi_mat,pairs,pji,N_edges_pos,epsilon=torch.tensor(1e-15,device=device))
    
        biases_i,s = new_biases_i(biases_i,pie,gamma,marginals,t) #updates biases with given prob and we also update conf of spins
        
        t+=1
        if t>TT: m_final=2
        else: m_final=m(s_endstate(N_nodes,s,p,c))
    

    mag_reached[kkk]=m(s)
    num_steps[kkk]=t
    conf[kkk]=s
    graphs[kkk]=N_nodes

end = time.time()

mag_reached=mag_reached.cpu().numpy()

num_steps=num_steps.cpu().numpy()

len_time=(end-start)
#len_time=len_time.cpu().numpy()

conf=conf.cpu().numpy()

graphs=graphs.cpu().numpy() 

np.savez("hpr_d4_p1.npz", mag_reached=mag_reached, conf=conf, num_steps=num_steps, graphs=graphs, time=len_time)

