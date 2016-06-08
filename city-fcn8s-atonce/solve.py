import caffe
import surgery
import sys
import os
import log_training
import setproctitle
import time as ti
from threading import Thread
setproctitle.setproctitle(os.path.basename(os.getcwd()))

folder = os.path.join('/data02/bioinf/treml/monitoring/city-fcn8-atonce', 'MyTEST')
if not os.path.isdir(folder):
    os.mkdir(folder)
    os.mkdir(os.path.join(folder, 'snapshot'))
else:
    raise Exception("Log-folder '", folder, "'exists")

weights = 'vgg16fc-CITYSCAPES.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver_steps = 10
N_train = 10
epochs = 200

for _ in range(N_train*epochs):
    
    t0 = ti.time()
    solver.step(solver_steps)
    print(solver_steps, 'solver steps take: ', round(float(ti.time()-t0),2))

    # Compute loss & accuracy
    t0 = ti.time()
    log_training.observe_loss_and_acc(solver, folder)
    print('diagnose_training takes: ', round(float(ti.time()-t0),2))
    
    # Get current weights
    t0 = ti.time()
    log_training.observe_weights(solver, folder)
    print('observe_weights takes: ', round(float(ti.time()-t0),2))
