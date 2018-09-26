import h5manager
import numpy as np
from mpi4py import MPI
import h5py
import os

# initialize MPI communicator
comm = MPI.COMM_WORLD
rank=comm.rank
size=comm.size

h5io = h5manager.h5IO('./data/prova.hdf5', comm)

# initialize data (np array)
nx=12
ny=3
nz=9

# ###### CHECK h5io._get_first_last  ############
frst, lst= h5io._get_first_last(nx)
nxL = lst-frst

tot_events_pproc=nxL*ny*nz
begin=rank*tot_events_pproc
end=(rank+1)*tot_events_pproc

print( "RANK = {} --- nx = {} --- nxL = {} ---  frst = {} --- lst = {} --- ny = {} --- nz = {} --- tot_events_pproc = {} --- begin = {} --- end = {}".format(rank, nx, nxL,frst,lst, ny,nz, tot_events_pproc, begin, end) )

data=np.arange(start=begin,stop=end, dtype=np.int64).reshape(nxL,ny,nz)

print( "RANK = {}\n data =\n{}".format(rank, data) )

try:
    ds0 = h5io.add_dataset('group0', 'ds0', data, (nx,ny,nz), np.int64, 'x', sample_rate=3, sample_rate_u='Hz', ciao=3.14)
except NameError as e:
    print (e, flush=True)
finally:
    if(rank==0):
        os.system("h5dump -A "+h5io.h5filepath)

comm.Barrier()
ds0_read_from_h5 = h5io.get_dataset('/group0/ds0', 'x', nx)
assert data.all() == ds0_read_from_h5.all(), "Wrong read of ds0_read_from_h5."


try:
    ds0_wrong_name = h5io.get_dataset('/group0/ds0XXXXXXXXXXX', 'x', nx)
except NameError as e:
    print (e, flush=True)

try:
    h5io_fail = h5manager.h5IO('./data/fail.hdf5', comm)
    ds0_wrong_h5filename = h5io_fail.get_dataset('/group0/ds0XXXXXXXXXXX', 'x', nx)
except OSError as e:
    print (e, flush=True)





# print("END OF TEST FILE", flush=True)






#
