# vgg
#./caffe/build/tools/caffe train \
#--gpu 0 \
#--solver=models/vgg/solver.prototxt \
#--weights=models/vgg/VGG_ILSVRC_16_layers.caffemodel

# alexnet
./caffe/build/tools/caffe test --gpu 0 --solver=models/alexnet/solver.prototxt --weights=data/models/alexnet/bvlc_alexnet.caffemodel 2>&1 | tee logfile/val_100img.log


