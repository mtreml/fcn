from __future__ import division
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import montrain
from collections import OrderedDict
import parse_log as pl
import csv

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    """Computes a pixelwise histogram over all n_cl classes
       n_ij in hist (n_cl x n_cl): number of pixels of class i in groundtruth predicted to be class j
       Also computes loss
    """
    n_cl = net.blobs[layer].channels
#    print('channels in score layer (n_cl): ', n_cl)
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    print('compute loss in score_alt.py ...')
    for idx in dataset:
        print('idx: ', idx)
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, test_dict_list, layer='score', gt='label'):
    """Evaluates segmentation metrics on testset and appends them to a list of dicts
       Every metric corresponds to a key in the dict
       Saves all data to csv
    """
#    print('>>>', datetime.time(datetime.now()), 'Begin seg tests for : ' + dataset)
    solver.test_nets[0].share_with(solver.net)
    loss, pix_acc, mean_acc, iu, fwavacc = do_seg_tests(solver.test_nets[0],
							solver.iter,
							save_format,
							dataset,
							layer,
							gt)

    row = OrderedDict([
		('NumIters', solver.iter),
		('loss', loss),
		('PixelAccuracy', pix_acc),
		('MeanAccuracy', mean_acc),
		('IU', iu),
		('FreqWeighMeanAcc', fwavacc)
		])
    
    test_dict_list.append(row)

    return test_dict_list	
    
def do_seg_tests(net, iteration, save_format, dataset, layer='score', gt='label'):
    """Specify the metrics you want to evaluate here       
    """

    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iteration)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'loss', loss)
    # overall accuracy
    pix_acc = np.diag(hist).sum() / hist.sum()
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'overall accuracy', pix_acc)
    # per-class accuracy
    mean_acc = np.nanmean(np.diag(hist) / hist.sum(1))
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'mean accuracy', mean_acc)
    # per-class IU
    iu_ = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iu = np.nanmean(iu_)
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'mean IU', iu)
    # freq weighted acc    
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu_[freq > 0]).sum()
#    print('>>>', datetime.time(datetime.now()), 'Iteration', iteration, 'fwavacc', fwavacc)
    
    return loss, pix_acc, mean_acc, iu, fwavacc


def diagnose_training(solver, train, val, save_format, layer='score', gt='label'):
    """Invoke in solver.py to write diagnostic data to csv and plots
    """
     
    # get diagnosis for TESTING data
    if 'test_dict_list' not in globals():
       global test_dict_list
       test_dict_list = []
    if 'train_dict_list' not in globals():
       global train_dict_list
       train_dict_list = []

#    test_dict_list = seg_tests(solver, save_format, val, test_dict_list, layer='score', gt='label')
    train_dict_list = seg_tests(solver, save_format, train, train_dict_list, layer='score', gt='label')

#    save_csv_file('logs', 'csv', test_dict_list, dataset='test', delimiter=',', verbose=False)
    save_csv_file('logs', 'csv', train_dict_list, dataset='train', delimiter=',', verbose=False) 

    plot_diagnostics(train_dict_list, test_dict_list) 


	    


def plot_loss(train_dict_list, test_dict_list)
    # initiate plots            
    if 'trainloss_fig' not in globals():
        global comparetraintestloss_fig
        loss_fig = montrain.LossFig(title='Loss')
    
    loss_fig.update(train_dict_list)


def plot_diagnostics(train_dict_list, test_dict_list):            
    """Plot different diagnostics whilst training

    Initiates the plot-class if it does not exists
    Updates data live
    """
    
    # initiate plots            
    if 'trainloss_fig' not in globals():
        global trainloss_fig, testloss_fig, trainmetrics_fig, testmetrics_fig, comparetraintestloss_fig, comparetraintestmetrics_fig
        trainloss_fig = montrain.LossFig(title='Train')
        testloss_fig = montrain.LossFig(title='Test')
        trainmetrics_fig = montrain.MetricsFig(title='Train')
        testmetrics_fig = montrain.MetricsFig(title='Test')
        comparetraintestloss_fig = montrain.CompareLossFig(title='Loss')
        comparetraintestmetrics_fig = montrain.CompareMetricsFig(title='Metrics')
    
    trainloss_fig.update(train_dict_list)
    testloss_fig.update(test_dict_list)
    testmetrics_fig.update(test_dict_list)
    trainmetrics_fig.update(train_dict_list)
    comparetraintestloss_fig.update(train_dict_list, test_dict_list)
    comparetraintestmetrics_fig.update(train_dict_list, test_dict_list)

def save_csv_file(logfile_path, output_dir, dict_list, dataset, delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)

    test_filename = os.path.join(output_dir, log_basename + '.' + dataset)
    write_csv(test_filename, dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print('Wrote %s' % output_filename)
