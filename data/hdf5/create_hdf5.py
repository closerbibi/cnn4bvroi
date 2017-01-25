import numpy as np
import lmdb
import caffe
import os
import random
import h5py
import pdb
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import write_train_test_loc as wloc
import shutil

def loadtoXY(option, classdict):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    # shuffle the data
    content = os.listdir(path)
    filelist = random.sample(content, len(content))
    N = len(filelist)
    X = np.zeros((N, 1, 32, 32), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    count = 0
    for fname in filelist:
        X[count] = np.load(path+'/'+fname)['data']
        Y[count] = classdict[str(np.load(path+'/'+fname)['classname'])]
        count += 1
    return X, Y, N

def loadtoXY_mat(option, classdict):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    # shuffle the data
    content = os.listdir(path)
    filelist = random.sample(content, len(content))
    N = len(filelist)
    X = np.zeros((N, 1, 112, 112), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    count = 0
    for fname in filelist:
        tmpX = sio.loadmat(path+'/'+fname)['target_grid']
        tmpX = np.resize(tmpX,(112,112,1))
        X[count] = tmpX
        Y[count] = classdict[str(fname.split('.')[0].split('_')[-1])]
        count += 1
        print 'file %s done' % fname
    return X, Y, N

def write_lmdbtxt(filelist, label, pilotlen):
    count = 0
    with open('val.txt', 'w') as f:
        for fname in filelist:
            f.write(fname+' '+str(label[count]) +'\n')
            if count == (pilotlen-1):
                break
            count += 1
        f.close()

def loadtoXY_imagenet(option):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    N = 100
    # shuffle the data
    filelist = os.listdir(path)
    filelist = np.sort(filelist)[:N]
    #N = len(filelist) # too big to store
    X = np.zeros((N, 3, 224, 224), dtype=np.uint8)
    Y = np.zeros(N, dtype=np.int64)
    countx = 0
    county = 0
    text_file = test_path + '/../ILSVRC2012_validation_ground_truth.txt'
    lines = [line.rstrip('\n') for line in open(text_file)]
    lines = map(int, lines)
    Y = np.asarray(lines)[:N]
    write_lmdbtxt(filelist, np.asarray(lines), len(filelist))
    for fname in filelist:
        tmpX = cv2.imread(path+'/'+fname)
        #bb, gg, rr = cv2.split(tmpX)
        #tmpX = cv2.merge([rr,gg,bb])
        tmpX = cv2.resize(tmpX,(224,224))
        tmpX = np.swapaxes(tmpX,0,2)
        tmpX = np.swapaxes(tmpX,1,2)
        X[countx] = tmpX
        countx += 1
        if countx == N:
            break
        print 'file %s done' % fname
    '''
    for f in gtfile:
        Y[county] = f.readlines()
        county += 1
    '''
    return X, Y, N

def compute_mean(X):
    data_mean = np.mean(X, axis=0)
    return data_mean

def shift_data(data, mean):
    #new_data = data - mean
    mean = np.array([104,117,123])
    new_data = data - mean[:,np.newaxis,np.newaxis]
    return new_data


def creat_hdf5(option, X, Y, N, batsz):
    for i in range(N):
        if (i % batsz) == 0:
            filestr = option+'_dir/'+option+'_%07d'%i+'.h5'
            with h5py.File(filestr,'w') as f:
                try:
                    f['data'] = X[i:i+batsz-1,:,:,:]
                    f['label'] = Y[i:i+batsz-1]
                    print('batch %d finished'%i)
                except:
                    f['data'] = X[i:,:,:,:]
                    f['label'] = Y[i:]
                    print('batch %d finished'%i)

#train_path = '../picture_roi'
train_path = '/home/closerbibi/dataset/imagenet/ILSVRC2012_img_val'
test_path = '/home/closerbibi/dataset/imagenet/ILSVRC2012_img_val'

# bed=157, chair=5, table=19, sofa=83, toilet=124
classdict = {
        'chair': 1,
        'table': 2,
        'sofa': 3,
        'toilet': 4,
        'bed': 5,
        }
if os.path.exists('train_dir'):
    shutil.rmtree('train_dir')
if os.path.exists('test_dir'):
    shutil.rmtree('test_dir')
os.makedirs('train_dir')
os.makedirs('test_dir')
'''
#training
X,Y,N = loadtoXY_mat('train', classdict)
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('train', new_X, Y, N, 10)
'''
#testing
X,Y,N = loadtoXY_imagenet('test')
data_mean = compute_mean(X)
new_X = shift_data(X, data_mean)
creat_hdf5('test', X, Y, N, 10)
creat_hdf5('train', X, Y, N, 10)
wloc.write_file('../train', 'hdf5/train_dir')
wloc.write_file('../test', 'hdf5/test_dir')
