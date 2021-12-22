# (wp) (st,sap) (bl,st)
# (wp) (shp,st) (bl,ps)
#
# How to make Distance map
# What is distance map
# Summary of MSA
# MSA Results
# 

# First write about parser
# And how baker group parses and extracts information
# Write what is MSA and which method and parameters are used and what the parameters are
# Then write how ca matrix is constructed
# Then write how the MSA is generated


# * proteins and chains to do in ffindex
# * ca backbone chain extract baker group methods
# * hhblits methods
# * asm generation methods
# * pair statistics MSA
    # * I think I will borrow Baker's code for this (Since we are using tensorflow)

# * Inter-residue prediction

# * Additional papers

# FUTURE STEPS
# * Baker code direct
#   * Baker code for data processing is directed embedded as tensorflow graph
#   * This makes it difficult to use it anywhere else really
#   * Additional Processing
# * Some details about inv cov dca
# * What to write in documentation?


# * ALPHAFOLD 
#   * How do they handle the seq-seq/seq-msa non-3d data inputs?
#   * How is the former integrated to the 3d invariant tensor network
#   * What attention do they use iteration?

# An implementation of MSA preprocessing from
# Improved protein structure prediction using predicted interresidue orientations (Baker 2019)

import torch
import numpy as np

# Couple of checks
# TODO: Set seq max length
# TODO: Combine this msa_a3m code into one workflow and document

ALPHABET = bytes('-ARNDCQEGHILKMFPSTWYV', 'utf8')

def msa__a3m(fp):
    ''' Given a a3m filepath, converts it to a msa array of strings
    fp: a3m filepath
    return: Array of msa strings
    '''
    lines = open(fp).readlines()

    msa = []
    for l in lines:

        if l.startswith('>'): continue

        l = l.translate( {ord(i):None for i in 'abcdefghijklmnopqrstuvwxyz'} ) # remove lower case
        l = l.strip()
        msa.append(l)

    return msa

def nseq__seq(s):
    ''' Given a string amino sequence, converts it to a numerical sequence
    s: string sequence
    returns: numerical sequence
    '''
    s = bytes(s, 'utf8')
    s = np.array(list(s))

    for i,a in enumerate(ALPHABET):
        s[s==a] = i

    s[s>20] = 0 # set unknown characters to gap
    return s 


def msa_a3m(fp):
    msa = msa__a3m(fp)
    msa = np.array( [nseq__seq(s) for s in msa] )
    return msa

def msa1hot_a3m(fp):
    ''' Given a a3m filepath, converts it to a one-hot encoding of msa
    fp: a3m filepath
    return: 1 hot encoding of msa. NxLx21 (Number seq)x(seq Length)x(Amino+gap)
    '''
    # Parse Files
    msa = msa_a3m(fp)
    # Onehot
    msa = torch.tensor(msa)
    msa = torch.nn.functional.one_hot(msa, num_classes=len(ALPHABET))
    msa = msa.float()
    return msa

def PSSM(msa): 
    ''' Given a msa converts it to a pssm
    msa: multiple sequence alignment of shape NxLx21 (Number seq)x(seq Length)x(Amino+gap)
    return: ppsm (Lx21) + entropy matrix (Lx1) = (Lx22)
    '''
    # Note: Baker uses a weighting scheme based on w (N shaped tensor)
    # where torch.sum(w*msa,0)/( torch.sum(w)+1e-9 )
    pssm = torch.sum(msa,0)
    entropy = torch.sum( -pssm * torch.log(pssm), 1, keepdim=True)

    return torch.cat([pssm, entropy], 1)

def W(msa, T=.8):
    ''' Constructs the weights for each sequence based on number of msa seqs that share T% identity with itself
    msa: multiple sequence alignment of shape NxLx21 (Number seq)x(seq Length)x(Amino+gap)
    T: Threshold value
    return: weight of shape (N)
    '''
    L = msa.shape[1]
    w = torch.tensordot(msa,msa,[(1,2),(1,2)])
    w = w>(L*T)
    w = 1/w.sum(1)
    return w

def DCA(msa, penalty):
    '''
    msa: multiple sequence alignment of shape NxLx21 (Number seq)x(seq Length)x(Amino+gap)
    penalty:
    return: DCA matrix of shape (L)x(L)x(21*21+1)
    '''
    N,L,A = msa.shape #(Number seq)x(seq Length)x(Amino+gap)
    w = W(msa).view(N,1)
    Meff = torch.sum(w)

    # Covariance is c-ab where c is f(i,j,A,B) and a,b are f(i,A), f(j,B) 
    # where i,j are sequence index/ A,B are amino acids
    # eq 1 (Baker 2019)
    x = (msa.reshape(N,L*A)*w).sum(0, keepdim=True) #1x(L*21)
    ab = x.T @ x # (L*21)x(L*21)
 
    x = msa.reshape(N,L*A)* torch.sqrt(w) #Nx(L*21)
    c = x.T @ x # (L*21)x(L*21)
    cov = (c-ab)/Meff

    # Inverse Covariance
    K = penalty/torch.sqrt(Meff)
    icov = torch.inverse( cov + K*torch.eye(L*A) )
    icov = icov.reshape(L,A,L,A) #Lx21xLx21
    x = icov
    icov = icov.permute(0,2,1,3).reshape(L,L,A*A) #LxLx441
    
    # APC average product correction score eq 4 (Baker 2019)
    zI = 1-torch.eye(L)
    x = torch.sqrt( torch.sum(x[:,:20,:,:20]**2, [1,3]) ) * zI #LxL
    correction = x.sum(1,keepdim=True) @ x.sum(0,keepdim=True)/x.sum() #LxL
    apc = (x-correction) * zI
    apc = apc.unsqueeze(2) #LxLx1
    
    return torch.cat([icov, apc], 2) #LxLx(441+1)



'''
msa = msa1hot_a3m('/raid0/ychnh/a3m/4GUL_A.a3m')
# 1D features
seq= msa[0,:,:20]
pssm= PSSM(msa)
F1 = torch.cat([seq,pssm],1) #Lx20+Lx22 = Lx42

L = msa.shape[1]

a = F1.unsqueeze(0).repeat(L,1,1) #LxLx42
b = F1.unsqueeze(1).repeat(1,L,1) #LxLx42

# DCA features
dca = DCA(msa, 4.5) #LxLx442
print(dca)
'''
