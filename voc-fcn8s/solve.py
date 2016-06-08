import caffe
import surgery
import sys
import numpy as np
import os
import log_training
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '../voc-fcn16s/fcn16s-heavy-pascal_transplanted.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#train = np.loadtxt('/data02/bioinf/treml/pascal/SBDD/dataset/train_debug.txt', dtype=str)
#val = np.loadtxt('/data02/bioinf/treml/pascal/VOC2011/ImageSets/Segmentation/seg11valid_debug.txt', dtype=str)

iter_steps = 1
for _ in range(1000*1111):
    solver.step(iter_steps)
    log_training.diagnose_training(solver)

#    score_alt.diagnose_training(solver, train, val, save_format=False, layer='score', gt='label')
