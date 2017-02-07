# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import pdb
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../caffe_origin/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
#caffe.set_mode_cpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

#model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
#model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

model_def = 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'

#model_def = 'models/vgg/vgg_train_val.prototxt'
model_weights = 'models/vgg/VGG_ILSVRC_16_layers.caffemodel'

if os.path.isfile(model_weights):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

for k in xrange(5):
	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	net.blobs['data'].reshape(50,        # batch size
				  3,         # 3-channel (BGR) images
				  224, 224)  # image size is 227x227
	image = caffe.io.load_image('../../dataset/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG' % (k+1))
	transformed_image = transformer.preprocess('data', image)
	#plt.imshow(image)

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

	print 'predicted class is:', output_prob.argmax()

	# load ImageNet labels
	labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
	if not os.path.exists(labels_file):
	    print 'ehhh...'#!../data/ilsvrc12/get_ilsvrc_aux.sh
	    
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	print 'output label:', labels[output_prob.argmax()]

	# sort top five predictions from softmax output
	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	print 'probabilities and labels:'
	zip(output_prob[top_inds], labels[top_inds])
	print('image: %d' % (k+1))
	print output_prob[top_inds]
	print labels[top_inds], '\n'