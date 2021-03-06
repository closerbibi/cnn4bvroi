import numpy as np
import lmdb
import caffe
import os
import pdb
import cv2
import shutil

def get_N(option):
    if option == 'train':
        return 50000
    elif option == 'test':
        return 20000

def loadtoXY4voxeldata(option, classdict):
    # Let's pretend this is interesting data
    N = get_N(option)
    X = np.zeros((N, 1, 32, 32), dtype=np.int64) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path
    count = 0
    for fname in os.listdir(path):
        X[count] = np.load(path+'/'+fname)['data']
        Y[count] = classdict[str(np.load(path+'/'+fname)['classname'])]
        count += 1
    return X, Y, N
def loadtoXY(option, classdict):
    # Let's pretend this is interesting data
    if option == 'train':
        path = train_path
    elif option == 'test':
        path = test_path
    filelist = np.sort(os.listdir(path))
    N = 1000#len(filelist)#get_N(option)
    X = np.zeros((N, 3, 256, 256), dtype=np.uint8) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)
    # loading label
    text_file = 'ILSVRC2012_validation_ground_truth.txt'
    #text_file = '../../caffe/examples/images/fuck.txt'
    lines = [line.rstrip('\n') for line in open(text_file)]
    lines = map(int, lines)
    Y = np.asarray(lines)[:N]

    # loading data
    count = 0
    for fname in filelist:
        tmpX = cv2.imread(path+'/'+fname)
        tmpX = cv2.resize(tmpX,(256,256))
        tmpX = np.swapaxes(tmpX,0,2)
        tmpX = np.swapaxes(tmpX,1,2)
        X[count] = tmpX
        print 'file %s done' % fname
        count += 1
        if count == N:
            break
    return X, Y, N

def creat_lmdb(option, X, Y, N):
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10

    env = lmdb.open(option+'_lmdb', map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(Y[i])
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


#train_path = '/home/closerbibi/3D/understanding/rankpooling/python/train_dir'
#test_path = '/home/closerbibi/3D/understanding/rankpooling/python/test_dir'
test_path = '../../../../dataset/ILSVRC2012_img_val'
#test_path = '../../caffe/examples/images'


if os.path.exists('train_lmdb'):
    shutil.rmtree('train_lmdb')
if os.path.exists('test_lmdb'):
    shutil.rmtree('test_lmdb')
os.makedirs('train_lmdb')
os.makedirs('test_lmdb')


classdict = {
        'bathtub': 1,
        'bed': 2,
        'chair': 3,
        'desk': 4,
        'dresser': 5,
        'monitor': 6,
        'night_stand': 7,
        'sofa': 8,
        'table': 9,
        'toilet': 10
        }
#X,Y,N = loadtoXY('train', classdict)
#creat_lmdb('train', X, Y, N)
X,Y,N = loadtoXY('test', classdict)
creat_lmdb('test', X, Y, N)
