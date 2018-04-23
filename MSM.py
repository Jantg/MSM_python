#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:55:04 2018

@author: jan
"""
import numpy as np
#from starting_vals import MSM_starting_values,MSM_starting_values_pf
from MSM_likelihood import MSM_likelihood,particle_filter,LW_filter
#from MSM_likelihood import LW_filter
#import scipy
import pandas as pd

def T_mat_template(kbar):
    kbar2 = 2**kbar
    A = np.zeros((kbar2,kbar2))
    for i in range(kbar2):
        for j in range(i,kbar2-i):
            A[i,j] = np.bitwise_xor(i,j)
    return(A)     
def MSM_modified(data,kbar,startingvals):
        #kbar = int(kbar)
        A_template = T_mat_template(kbar)
        startingvals, LLs,ordered_parameters = MSM_starting_values(data,startingvals,kbar,A_template)
        bnds = ((1,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
        minimizer_kwargs = dict(method = "L-BFGS-B",bounds = bnds,args = (kbar,dat,A_template,None))
        #res = scipy.optimize.minimize(MSM_likelihood,x0 = startingvals,args = (kbar,dat,A_template,None),method = "L-BFGS-B",
                                      #options = {"disp":True},bounds = bnds)
        res = scipy.optimize.basinhopping(MSM_likelihood,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,niter = 3)
        parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
        #print(parameters)
        LL, LLs = MSM_likelihood(parameters,kbar,data,A_template,None,2)
        LL = -LL
        
        return(LL,LLs,parameters)
def MSM_particle(data,kbar,n_particles,startingvals):
    A_template = T_mat_template(kbar)
    #startingvals, LLs,ordered_parameters = MSM_starting_values_pf(data,startingvals,kbar,A_template,n_particles)
    startingvals, LLs,ordered_parameters = MSM_starting_values(data,startingvals,kbar,A_template)
    bnds = ((1,50),(1,1.99),(1e-3,0.999999),(1e-4,5))
    #LL,LLs,M_mat= particle_filter(startingvals,kbar,data,A_template,n_particles)
    #return(LL,LLs,M_mat)
    res = scipy.optimize.minimize(particle_filter,x0 = startingvals,args = (kbar,dat,A_template,n_particles,None),
                                  method = "L-BFGS-B",options = {"disp":True},bounds = bnds)
    #minimizer_kwargs = dict(method = "SLSQP",bounds = bnds,args = (kbar,dat,A_template,n_particles,None))
    #res = scipy.optimize.minimize(minimizer_kwargs)
    #res = scipy.optimize.basinhopping(particle_filter,x0 = startingvals,minimizer_kwargs = minimizer_kwargs,niter = 1)
    parameters,LL,niters,output = res.x,res.fun,res.nit,res.message
    LL, LLs,M_mat = particle_filter(parameters,kbar,data,A_template,n_particles,None,2)
    LL = -LL
        
    return(LL,LLs,parameters,M_mat)

  
#A_template = T_mat_template(3)
#import pandas as pd
#dat = pd.read_csv("data_demo.csv",header = None)
#dat = np.array(dat)
#MSM_starting_values(dat,None,3,A_template)
        
if __name__ == "__main__":
    T = 1000
    kbar = 3
    g_kbar = 0.5
    b = 5
    m0 = 1.5
    m1 = 2-1.5
    sig = 3
    g_s = np.zeros(kbar)
    M_s = np.zeros((kbar,T))
    g_s[0] = 1-(1-g_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        g_s[i] = 1-(1-g_s[0])**(b**(i))
    
    for j in range(kbar):
        M_s[j,:] = np.random.binomial(1,g_s[j],T)
    dat = np.zeros(T)
    tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
    dat[0] = np.prod(tmp)
    for k in range(1,T):
        for j in range(kbar):
            if M_s[j,k]==1:
                tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])

        dat[k] = np.prod(tmp)
    dat = np.sqrt(dat)*sig* np.random.normal(size = T)
    dat = dat.reshape(-1,1)
    #dat = pd.read_csv("data_demo.csv",header = None)
    #dat = np.array(dat)
    LL_,LLs_,params_= MSM_modified(dat,3,None)
    #kbar = 5
    #kbar2 = 2**kbar
    #LL_,LLs_,params,M_mat = MSM_particle(dat,kbar,1000,None)
    
    #A_template = T_mat_template(kbar)
    #startingvals, LLs,ordered_parameters = MSM_starting_values(dat,None,kbar,A_template)
    startingvals = [1,1.6,0.9,10]
    LL,LLs,M_mat,inputs,w = LW_filter(startingvals,kbar,dat,10000,0.95,8000)
    LLs_out = pd.DataFrame(LLs)
    M_mat_out = pd.DataFrame(M_mat)
    b_out = pd.DataFrame(inputs[0])
    m0_out = pd.DataFrame(inputs[1])
    g_out = pd.DataFrame(inputs[2])
    s_out = pd.DataFrame(inputs[3])
    #param_out = pd.DataFrame(inputs)
    LLs_out.to_csv("LLs.csv",index = False)
    M_mat_out.to_csv("states.csv",index = False)
    b_out.to_csv("b.csv",index = False)
    m0_out.to_csv("m0.csv",index = False)
    g_out.to_csv("g.csv",index = False)
    s_out.to_csv("s.csv",index = False)
    
    #param_out.tocsv("params.csv",index = False)
    #cts = {key: 0 for key in np.linspace(0,kbar2-1,kbar2)}
    #for i,v in enumerate(M_mat[-1,:]):
    #    cts[v] +=1
    #states = np.fromiter(cts.keys(),dtype = "uint8")
    #states = np.unpackbits(states.reshape(-1,1),axis = 1)
    #states = states[:,-kbar:]
        
    
