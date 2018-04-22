#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:12:32 2018

@author: jan
"""

import numpy as np

def gofm(inpt,kbar):
    """
    A function that calculates all the possible volatility states
    """
    m0 = inpt[1]
    m1 = 2-m0
    kbar2 = 2**kbar
    g_m1 = np.arange(kbar2)
    g_m = np.zeros(kbar2)
    
    for i in range(kbar2):
        g =1
        for j in range(kbar):
            if np.bitwise_and(g_m1[i],(2**j))!=0:
                g = g*m1
            else:
                g = g*m0
        g_m[i] = g
    return(np.sqrt(g_m))
            
def transition_mat(A,inpt,kbar):
    b = inpt[0]
    gamma_kbar = inpt[2]
    gamma = np.zeros((kbar,1))
    #print(b,gamma_kbar)
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    #print(gamma[0,0])
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,1]
    kbar2 = 2**kbar
    prob = np.ones((kbar2,1))
    
    for i in range(kbar2):
        for m in range(kbar):
            prob[i,0] =prob[i,0] * gamma[kbar-m-1,
                np.unpackbits(np.array([i],dtype = np.uint8))[-(m+1)]]
    #print(prob)
    #print(A[0,0])
    for i in range(2**(kbar-1)):
        for j in range(i,2**(kbar-1)):
            A[kbar2-i-1,j] = prob[np.rint(kbar2 - A[i,j]-1).astype(int),0]
            A[kbar2-j-1,i] = A[kbar2-i-1,j]
            A[j,kbar2-i-1] = A[kbar2-i-1,j]
            A[i,kbar2-j-1] = A[kbar2-i-1,j]
            A[i,j] = prob[np.rint(A[i,j]).astype(int),0]
            A[j,i] = A[i,j].copy()
            A[kbar2-j-1,kbar2-i-1] = A[i,j]
            A[kbar2-i-1,kbar2-j-1] = A[i,j]
    #print(A)       
    return(A)

def transition_prob(inpt,state_t,kbar): 
    kbar2 = 2**kbar
    b = inpt[0]
    gamma_kbar = inpt[2]
    gamma = np.zeros(kbar)
    gamma[0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    probs = np.ones(kbar2)
    #print(gamma[0,0])
    for i in range(1,kbar):
        gamma[i] = 1-(1-gamma[0])**(b**(i))
    for i in range(kbar2):
        b_xor = np.bitwise_xor(state_t,i)
        val = np.unpackbits(np.arange(b_xor,b_xor+1,dtype = np.uint16).view(np.uint8))
        val = np.append(val[8:],val[:8])[-kbar:]
        for j,v in enumerate(val):
            if v == 1:
                probs[i] = probs[i]*gamma[j]
            else:
                probs[i] = probs[i]*(1-gamma[j])
    return(probs)
        
def MSM_likelihood(inpt,kbar,data,A_template,estim_flag,nargout =1):
    if not hasattr(inpt,"__len__"):
        inpt = [estim_flag[0],inpt,estim_flag[1],estim_flag[2]]
        
    sigma = inpt[3]/np.sqrt(252)
    k2 = 2**kbar
    A = transition_mat(A_template.copy(),inpt,kbar)
    g_m = gofm(inpt,kbar)
    T = len(data)
    pi_mat = np.zeros((T+1,k2))
    LLs = np.zeros(T)
    pi_mat[0,:] = (1/k2)*np.ones((1,k2))
    """
    Likelihood Algorithm
    """
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m#np.matlib.repmat(sigma*g_m,T,1)
    w_t = data #np.matlib.repmat(data,1,k2)
    #print(w_t.shape,s.shape,np.sum(s))
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    #print(A)
    for t in range(T):
        #print(pi_mat.shape,A.shape)
        piA = np.dot(pi_mat[t,:],A)
        #print(np.sum(piA))
        C = (w_t[t,:]*piA)
        ft = np.sum(C) # log
        if np.isclose(ft,0):
            pi_mat[t+1,1]=1
        else:
            pi_mat[t+1,:] = C/ft
        #print(ft)
        #print(np.any(w_t[t,:] == np.inf))    
        LLs[t] = np.log(w_t[t,:]@piA)
        
    LL = -np.sum(LLs)
    if np.any(np.isinf(LLs)):
        print("Log-likelihood is inf. Probably due to all zeros in pi_mat.")
    if nargout == 1:
        return(LL)
    else:
        return(LL,LLs)
    
def particle_filter(inpt,kbar,data,A_template,B,estim_flag,nargout =1):
    if not hasattr(inpt,"__len__"):
        inpt = [estim_flag[0],inpt,estim_flag[1],estim_flag[2]]

    sigma = inpt[3]/np.sqrt(252)
    k2 = 2**kbar
    A = transition_mat(A_template.copy(),inpt,kbar)
    g_m = gofm(inpt,kbar)
    Ms = np.arange(len(g_m))
    T = len(data)
    # For storing pi_t
    pi_mat = np.zeros((T,k2))
    pi_mat[0,:] = (1/k2)*np.ones((1,k2))
    M_mat = np.zeros((T,B))
    """
    Likelihood Algorithm
    """
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m#np.matlib.repmat(sigma*g_m,T,1)
    w_t = data #np.matlib.repmat(data,1,k2)
    #print(w_t.shape,s.shape,np.sum(s))
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    
    LLs = np.zeros(T-1)
    M_mat[0,:] = np.random.choice(Ms, size=B, replace=True, p=pi_mat[0,:])
    
    for i in range(T-1):
        M_temp = np.zeros(B)
        weights = np.zeros(B)
        for j,val in enumerate(M_mat[i,:]):
            M_temp[j] = np.random.choice(Ms,size = 1,p = A[val.astype(int),:])
        #print(M_temp)
        for k,val in enumerate(M_temp):
            #print(i,w_t.shape,len(M_temp))
            weights[k] = w_t[i+1,val.astype(int)]/np.sum(w_t[i+1,M_temp.astype(int)])
        M_mat[i+1,:] = np.random.choice(M_temp,size = B,replace = True,p = weights)
        
        LLs[i] = np.mean(w_t[i+1,M_mat[i+1,:].astype(int)])
    LL = np.sum(np.log(LLs))    
    if nargout == 1:
        return(LL)
    else:
        return(LL,LLs,M_mat)
    
def LW_filter(inpt,kbar,data,B,a,nEff):
    # set priors and other variables
    sigma = inpt[3]/np.sqrt(252)
    k2 = 2**kbar
    #A = transition_mat(A_template.copy(),inpt,kbar)
    inputs = []
    T = len(data)
    # For storing weights
    weights = np.zeros((T,B))
    w_t = np.zeros((T,B))
    weights[0,:] = (1/B)
    M_mat = np.zeros((T,B))
    Ms = np.arange(k2)
    """
    Likelihood Algorithm
    """
    pa = (2*np.pi)**(-0.5)
    #s = sigma*g_m#np.matlib.repmat(sigma*g_m,T,1)
    #w_t = data #np.matlib.repmat(data,1,k2)
    #print(w_t.shape,s.shape,np.sum(s))
    #w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    #w_t = w_t + 1e-16
    
    LLs = np.zeros(T-1)
    M_mat[0,:] = np.random.choice(Ms, size=B, replace=True, p=(1/k2)*np.ones(k2))
    s = sigma*gofm(inpt,kbar)
    hoge = pa*np.exp(-0.5*((data[0]/s)**2))/s
    w_t[0,:] = hoge[M_mat[0,:].astype(int)]
    for idx,val in enumerate(inpt):
        tmp = np.zeros((T,B))
        if idx == 0:
            tmp[0,:] = np.random.random(B)*10+1
        elif idx == 1:
            tmp[0,:] = np.random.random(B)+1
        elif idx == 2:
            tmp[0,:] = np.random.random(B)
        else:
            tmp[0,:] = np.random.random(B)*5
        inputs.append(tmp)
    #g_m = gofm(inpt,kbar)
    #sigma = inputs[3][0,:]/np.sqrt(252)
    #s = sigma
    N_eff = np.zeros(T)
    for i in range(T-1):
        muSystem = np.zeros((len(inpt),B))
        sigma2System = np.zeros((len(inpt),B))
        alphaSystem = np.zeros((len(inpt),B))
        betaSystem = np.zeros((len(inpt),B))
        M_temp = np.zeros(B)
        
        for j,val in enumerate(M_mat[i,:]):
            inp_tmp = [inputs[0][i,j],inputs[1][i,j],inputs[2][i,j],inputs[3][i,j]]
            #A = transition_mat(A_template.copy(),inp_tmp,kbar)
            prob = transition_prob(inp_tmp,val.astype(int),kbar)
            #M_temp[j] = np.random.choice(Ms,size = 1,p = A[val.astype(int),:])        
            M_temp[j] = np.random.choice(Ms,size = 1,p = prob) 
        for idx,val in enumerate(inputs):
            meanSystem= np.average(val[i,:],weights = weights[i,:])
            varSystem = np.average((val[i,:]-meanSystem)**2,weights = weights[i,:])
            #print(meanSystem,varSystem,np.max(weights[i,:]))
            muSystem[idx,:] = a*val[i,:]+(1-a)*meanSystem
            sigma2System[idx,:] = (1-(a**2))*varSystem
            alphaSystem[idx,:] = muSystem[idx,:]**2/sigma2System[idx,:]
            betaSystem[idx,:] = muSystem[idx,:]/sigma2System[idx,:]
        for k,val in enumerate(M_temp): # for M_t+1^1 to M_t+1^B
            #weight particles given the likelihood
            inp_tmp = [inputs[0][i,k],inputs[1][i,k],inputs[2][i,k],inputs[3][i,k]] 
            if inp_tmp[1]>2:
                inp_tmp[1] = 1.9999999
            sigma = inp_tmp[3]/np.sqrt(252)
            g_m = gofm(inp_tmp,kbar)
            s = sigma*g_m[val.astype(int)]
            #print(s,inp_tmp ,g_m[val.astype(int)])
            w_t[i+1,k] = pa*np.exp(-0.5*((data[i+1]/s)**2))/s
            w_t[i+1,k] = w_t[i+1,k] + 1e-16
            weights[i+1,k] = w_t[i+1,k]
        weights[i+1,:] = weights[i+1,:]/np.sum(weights[i+1,:])
        Ix = np.random.choice(B,size = B, replace =True,p = weights[i+1,:])
        M_mat[i+1,:] = M_temp[Ix.astype(int)]
        for idx,val in enumerate(inputs):
            inputs[idx][i+1,:] = np.random.gamma(shape = alphaSystem[idx,Ix.astype(int)],
                  scale = 1/betaSystem[idx,Ix.astype(int)],size = B)
            print(inputs[idx][i+1,:],np.mean(1/betaSystem[idx,:]))
            #print(np.max(inputs[idx][i+1,:]))
        #print(inputs[3][i+1,:])
        LLs[i] = np.mean(w_t[i+1,M_mat[i+1,:].astype(int)])
        N_eff[i+1]  =  1 / np.dot(weights[i+1,:],weights[i+1,:])
        print("finished running ",i," times ")
        if (N_eff[i+1] < nEff):
            print(i,"hogeeeeeeeeeeeeeeeeeeeeeeeeee")
            # multinominal resampling
            print(weights[i+1,:])
            tmp = np.random.choice(B, size = B, replace = True, p = weights[i+1,:])
            w_t[i+1,:] = w_t[i+1, tmp.astype(int)]
            #pfOutObsVar[it, ] <- pfOutObsVar[it, tmp]
            for idx,val in enumerate(inputs):
                inputs[idx][i+1,:] = val[i+1,tmp.astype(int)]
            weights[i+1,:] = 1/B
    LL = np.sum(np.log(LLs))    
    
    return(LL,LLs,M_mat,inputs)
"""
def particle_filtering(inpt,kbar,data,A_template,B):
    ## Initialization
    #B = Number of Samples drawn for each T
    sigma = inpt[3]/np.sqrt(252)
    k2 = 2**kbar
    A = transition_mat(A_template.copy(),inpt,kbar)
    g_m = gofm(inpt,kbar)
    T = len(data)
    # For storing pi_t
    pi_mat = np.zeros((T+1,k2))
    # For storing M_t
    M_mat = np.zeros((T+1,B))
    sim_like = np.zeros(T)
    pi_mat[0,:] = (1/k2)*np.ones((1,k2))
    
 
    pa = (2*np.pi)**(-0.5)
    s = sigma*g_m
    w_t = data 
    #w_t has to be 1x8 matrix
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16
    
    # Reference dictionary to match the value of g_m and its corresponding likelihood for resampling
    dict_ref = {val:w_t[key] for key, val in enumerate(g_m)}
    
    
    
    

    M_mat[0,:] = np.random.choice(g_m, size=B, replace=True, p=pi_mat[0,:])
    for i in range(T):
        # 1x8 matrix
        temp_pi = pi_mat[i,:] @ A
        
        # element multiplication
        ## Dimension of w_t and temp_pi has to be (1,k_bar)
        pi_mat[i+1,:] = w_t * temp_pi[None]
        
        # temporary sampling using the updated conditional distribution  (1000 samples)
        temp_M = np.random.choice(g_m, size=B, replace=True, p=pi_mat[i+1,:])
        # storing likelihoods that correspond to one of the values in "g_m"
        temp_like = np.array([dict_ref[val] for val in temp_M])
        # re_calculate the weighted probability.
        w_like = temp_like/np.sum(temp_like)
        M_mat[i+1,:]= np.random.choice(temp_M,size=B,replace=True,p=w_like)
        
        '''
        Simulated Likelihood
        '''
        # Corresponding likelihood for each value in "M_mat" at time t
        cond_dens = np.array([dict_ref[val] for val in M_mat[i+1,:]])
        sim_like[i] = np.sum(cond_dens)/B
    log_likelihood = np.sum(np.log(sim_like))
    return (pi_mat, M_mat, log_likelihood)  
"""