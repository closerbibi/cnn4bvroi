# vgg
#./caffe/build/tools/caffe train \
#--gpu 0 \
#--solver=models/vgg/solver.prototxt \
#--weights=models/vgg/VGG_ILSVRC_16_layers.caffemodel

# alexnet
./caffe/build/tools/caffe test -gpu 0 -model models/alexnet/train_val.prototxt -weights data/models/alexnet/bvlc_alexnet.caffemodel 2>&1 | tee logfile/gotohell.log
#./caffe/build/tools/caffe test -gpu 0 -model models/alexnet/deploy.prototxt -weights data/models/alexnet/bvlc_alexnet.caffemodel 2>&1 | tee logfile/gotohell.log

#./tools/test_net.py --gpu 0 --iters 248000  --solver models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt     --weights data/faster_rcnn_models/ZF_faster_rcnn_fina
