# shows the misclassified samples and the confusion matrix.
# Launched like this: python ../src/convnet_test_lmdb.py --proto lenet.prototxt --model snapshots/lenet_mnist_v3-id_iter_1000.caffemodel --lmdb ../caffe/examples/mnist/mnist_test_lmdb/
# python tools/convnet_test_lmdb.py --proto models/alexnet/train_val.prototxt --model data/models/alexnet/bvlc_alexnet.caffemodel --lmdb data/lmdb/test_lmdb

import sys
import caffe
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import argparse
from collections import defaultdict
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lmdb', type=str, required=True)
    args = parser.parse_args()

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    lmdb_env = lmdb.open(args.lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load('data/ilsvrc12/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)
    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)

        #out = net.forward_all(data=np.asarray([image]))
        #########################
        # transpose 1st time
        image = np.swapaxes(image,0,2)
        image = np.swapaxes(image,0,1)        
        #plt.imshow(image)
        #plt.show()
        #pdb.set_trace()
        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        net.blobs['data'].reshape(1,        # batch size
                                  3,         # 3-channel (BGR) images
                                  224, 224)  # image size is 227x227
        # create transformer for the input called 'data'
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformed_image = transformer.preprocess('data', image)

        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image
        
        pdb.set_trace()
        ### perform classification
        out = net.forward()
        #########################
        plabel = int(out['prob'][0].argmax(axis=0))
        probthis = out['prob'][0].max(axis=0)

        count = count + 1
        iscorrect = label == plabel
        correct = correct + (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("\rError: key=%s, expected %i but predicted %i, prob is %.3f" \
                    % (key, label, plabel, probthis))

        sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
        sys.stdout.flush()

    print(str(correct) + " out of " + str(count) + " were classified correctly")

    '''
    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
    '''
