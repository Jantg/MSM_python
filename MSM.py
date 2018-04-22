#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:55:04 2018

@author: jan
"""
import numpy as np
from starting_vals import MSM_starting_values,MSM_starting_values_pf
from MSM_likelihood import MSM_likelihood,particle_filter,LW_filter
import scipy
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
    
    dat = pd.read_csv("data_demo.csv",header = None)
    dat = np.array(dat)
    #LL,LLs,params = MSM_modified(dat,5,None)
    kbar = 3
    #kbar2 = 2**kbar
    #LL,LLs,params,M_mat = MSM_particle(dat,kbar,1000,None)
    
    A_template = T_mat_template(kbar)
    #startingvals, LLs,ordered_parameters = MSM_starting_values(dat,None,kbar,A_template)
    startingvals = [2,1.5,0.1,0.2]
    LL,LLs,M_mat,inputs = LW_filter(startingvals,kbar,dat,A_template,3000,0.975,1500)
    #cts = {key: 0 for key in np.linspace(0,kbar2-1,kbar2)}
    #for i,v in enumerate(M_mat[-1,:]):
    #    cts[v] +=1
    #states = np.fromiter(cts.keys(),dtype = "uint8")
    #states = np.unpackbits(states.reshape(-1,1),axis = 1)
    #states = states[:,-kbar:]
        
    
