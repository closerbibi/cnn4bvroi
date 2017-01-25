import os
import numpy as np

def write_file(option, path):
    with open(option+'_loc.txt', 'w') as f:
        filelist = os.listdir(path.split('/')[1])
        filelist = np.sort(filelist)
        for fname in filelist:
            f.write('data/' + path+'/'+fname+'\n')
        f.close()



def write_lmdbtxt(option, path):
    with open('val.txt', 'w') as f:
        for fname in os.listdir(path.split('/')[1]):
            f.write('data/' + path+'/'+fname+'\n')
        f.close()

#write_file('../train', 'hdf5/train_dir')
#write_file('../test', 'hdf5/test_dir')

