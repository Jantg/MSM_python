# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import scipy
from MSM_likelihood import MSM_likelihood

def MSM_starting_values(data,startingvals,kbar,A_template):
    if startingvals is None:
        print("No starting values entered: Using grid-search")
        
        b = np.array([1.5,3,6,20])
        lb = len(b)
        g = np.array([.1,.5,.9])
        lg = len(g)
        sigma = np.std(data)*np.sqrt(252)
        output_parameters = np.zeros(((lb*lg),3))
        LLs = np.zeros((lb*lg))
        m0_lower = 1.2
        m0_upper = 1.8
        idx = 0
        for i in range(lb):
            for j in range(lg):
                xopt,fval,ierr,numfunc = scipy.optimize.fminbound(MSM_likelihood,
                                                 x1 = m0_lower,x2 = m0_upper,xtol = 1e-3,
                                                 args = (kbar,data,A_template,[b[i],g[j],sigma]),full_output = True)
                m0,LL = xopt,fval
                output_parameters[idx,:] = b[i],g[j],m0
                LLs[idx] = LL
                idx +=1
        idx = np.argsort(LLs)
        LLs = np.sort(LLs)
        startingvals = output_parameters[idx[0],:].tolist()+[sigma]
        output_parameters = output_parameters[idx,:]
        return(startingvals,LLs,output_parameters)