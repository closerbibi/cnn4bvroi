import numpy as np
import os
import random
import caffe
import pdb

def loadtoXY(option):
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path

    # shuffle the data
    content = os.listdir(path)
    filelist = random.sample(content, len(content))
    N = len(filelist)
    X = np.zeros((N, 1, 32, 32), dtype=np.int64) # from uint8 turn into int64
    count = 0
    for fname in filelist:
        X[count] = np.load(path+'/'+fname)['data']
        count += 1
    return X, N

def compute_mean(X):
    data_mean = np.mean(X, axis=0)
    np.save('data_mean', data_mean)
    return data_mean

def tobinaryproto(data_mean, option):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.channels, blob.height, blob.width = data_mean.shape
    blob.data.extend(data_mean.astype(float).flat)
    binaryproto_file = open('mean/'+'%s_modelNet10_data_mean.binaryproto'%option, 'wb' )
    binaryproto_file.write(blob.SerializeToString())
    binaryproto_file.close()


train_path = '/home/closerbibi/3D/understanding/rankpooling/python/train_dir'
test_path = '/home/closerbibi/3D/understanding/rankpooling/python/test_dir'

X,N = loadtoXY('train')
data_mean = compute_mean(X)
tobinaryproto(data_mean, 'train')

X,N = loadtoXY('test')
data_mean = compute_mean(X)
tobinaryproto(data_mean, 'test')
