# -*- coding: utf-8 -*-

# module h5manager

import numpy as np
import h5py
# from mpi4py import MPI
import os

__all__ = ['h5IO']

class h5IO():
    def __init__(self, h5namefile, comm):
        self.h5filepath=h5namefile
        self.h5folderpath=os.path.split(self.h5filepath)[0]
        os.makedirs(self.h5folderpath, exist_ok=True)
        self.comm = comm

    def _get_first_last(self, sliced_axis_len):
        # evaluate range of pics to be written per process
        ppp=sliced_axis_len//self.comm.size # frontal 256 x 256 slices per process
        rem=sliced_axis_len%self.comm.size # remainder
        # with open("firstlast"+str(self.comm.Get_rank())+".txt", "w") as f:
        #     f.write("ppp={}  -- rem={}".format(ppp, rem))
        if(self.comm.rank>=rem):
            frst=self.comm.rank*ppp + rem
            lst=frst+ppp
        else:
            frst=self.comm.rank*ppp + self.comm.rank
            lst=frst+ppp+1
        return frst, lst


    def add_dataset(self, groupname, ds_name, data, ds_shape, ds_dtype, slicing_axis, **kwargs):
        assert slicing_axis=='x' or slicing_axis=='z', "Allowed slicing policies are either 'x' or 'z'."
        # add group to the hdf5
        def _add_group(group_name, h5file):
            # with open("XX"+str(self.comm.Get_rank())+".txt", "w") as f:
            #     f.write("\nself.h5filepath={} --- group_name ={} --- group_name in self.h5filepath = {}".format( self.h5filepath, group_name , group_name in self.h5filepath))

            if ( not group_name in h5file):
                group = h5file.create_group(group_name)
            else:
                group = h5file[group_name]
            return group

        # add the dataset
        with h5py.File(self.h5filepath,
                       'a',
                       driver='mpio',
                       comm=self.comm) as f:
            # create group or check if exists
            group = _add_group(groupname, f)
            # create dataset or raise exception if already exists
            if (not ds_name in group):
                # with open("XXds"+str(self.comm.Get_rank())+".txt", "w") as f:
                #     f.write("\nself.h5filepath={} --- ds_name ={} --- group_name in self.h5filepath = {}".format(self.h5filepath, ds_name ,  ds_name in group))

                ds = group.create_dataset(ds_name, shape=ds_shape, dtype=ds_dtype)
                if (slicing_axis=='x'):
                    first, last = self._get_first_last(ds_shape[0])
                    ds[first:last,:,:] = data
                elif (slicing_axis=='z'):
                    first, last = self._get_first_last(ds_shape[2])
                    ds[:,:,first:last] = data
                # initialize dataset's attributes
                for k in kwargs.keys():
                    ds.attrs[k] = kwargs[k]
                # return dataset
                return ds
            # if a dataset with the same name is present: raise exception
            else:
                raise NameError( "\nDataset {} already present in {} group!\n".format(ds_name, groupname) )


    def get_dataset(self, ds_name, slicing_axis, sliced_axis_len):
        assert slicing_axis=='x' or slicing_axis=='z', "Allowed slicing policies are either 'x' or 'z'."
        # returns the tuple first:last (i.e. sentinel)
        # MAIN: try to open the file to read the dataset
        try:
            with h5py.File(self.h5filepath,
                           'r',
                           driver='mpio',
                           comm=self.comm) as f:

                if ( not ds_name in f):
                    raise NameError("\nDataset " + ds_name + " not present in file " + self.h5filepath + ".\n")

                first, last = self._get_first_last(sliced_axis_len)
                if (slicing_axis=='x'):
                    data=np.array(f.get(ds_name)[first:last,:,:] )
                elif (slicing_axis=='z'):
                    data=np.array(f.get(ds_name)[:,:,first:last] )
                return data

        except OSError:
            raise OSError("\nFile " + self.h5filepath + " not present.\n")



#
