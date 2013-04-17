''' This file:

1) Reads the number of simulations (NumAgents) from the dictionary input file.

2) Divides it equally across the number of processors available.
[For simplicity, we just round the number of simulations per processor to an integer, so 
the total number of simulations may fall slightly short of the original intended. To make
sure it works, one can always request a number of processors such that NumAgents is a
multiple of it.]

3) Broadcasts the number of simulations per processor to each processor available, using
mpi4py's MPI.

4) Reduces the result of the simulations back to the processor ranked 0.

5) Export the result.

6) Runs the grmToolbox.estimate() as in the template.
'''

# standard library
import os
import sys

# edit pythonpath
dir_ = os.path.realpath(__file__).replace('ParallelRun.py','')
sys.path.insert(0, dir_)

# project library
import grmToolbox
''' 
===========
Simulation 
===========
Was simply grmToolbox.simulate()
'''
import grmReader, numpy as np
from mpi4py import MPI


comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nprocs = comm.Get_size()

''' Read number of agents to simulate and name of the file to be exported
'''
totalNumAgents = grmReader.read()['numAgents']
fileName       = grmReader.read()['fileName']

''' Distribute, run and gather the simulations using available processors
'''
if rank == 0:
    numAgents = int(np.floor(totalNumAgents/nprocs))

else:
    numAgents = None

numAgents = comm.bcast(numAgents, root=0)

for rank in range(nprocs):
    simOutput = grmToolbox.simulate(numAgents)

gatheredOutputs = comm.gather(simOutput, root=0)

''' Organize the data
'''

numCovarsOut = np.array(grmReader.read()['Y1_beta']).shape[0]  # numCovarOut~dimX
numCovarsCost = np.array(grmReader.read()['D_gamma']).shape[0]  # numCovarCost~dimZ

Y = np.tile(np.nan, (totalNumAgents,))
D = np.tile(np.nan, (totalNumAgents,))
X = np.tile(np.nan, (totalNumAgents,numCovarsOut))
Z = np.tile(np.nan, (totalNumAgents,numCovarsCost))
treatments = np.tile(np.nan, (nprocs,3))

for sourceRank in range(nprocs):
        Y[sourceRank*numAgents : (sourceRank+1)*(numAgents)]           = gatheredOutputs[sourceRank]['variables'][:,0]
        D[sourceRank*numAgents : (sourceRank+1)*(numAgents)]           = gatheredOutputs[sourceRank]['variables'][:,1]
        X[sourceRank*numAgents : (sourceRank+1)*(numAgents),:]         = gatheredOutputs[sourceRank]['variables'][:,2:(2+numCovarsOut)]
        Z[sourceRank*numAgents : (sourceRank+1)*(numAgents),:]         = gatheredOutputs[sourceRank]['variables'][:,-numCovarsCost:]
        treatments[sourceRank,:]                                               = gatheredOutputs[sourceRank]['treatments']

ATE  = np.mean(treatments[:,0])
ATT  = np.mean(treatments[:,1])
ATUT = np.mean(treatments[:,2])

'''Checks
'''


assert (np.all(np.isfinite(Y)))
assert (np.all(np.isfinite(D)))

assert (Y.shape  == (totalNumAgents, ))
assert (D.shape  == (totalNumAgents, ))

assert (Y.dtype == 'float')
assert (D.dtype == 'float')

assert ((D.all() in [1.0, 0.0]))

assert (ATE.dtype == 'float')
assert (ATT.dtype == 'float')
assert (ATUT.dtype == 'float')


''' Output everything to files
'''

np.savetxt(fileName, np.column_stack((Y, D, X, Z)), fmt= '%8.3f')

np.savetxt('treatments.txt', np.column_stack((ATE,ATT,ATUT)), fmt= '%8.3f')


'''
===========
Estimation
===========
'''
grmToolbox.estimate()

