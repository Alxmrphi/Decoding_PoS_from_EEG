# These were run on the University of Birmingham's High-Performance Computing Cluster (BlueBEAR)

import numpy as np
import glob
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import argparse
import os

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-n_avg', help='number of averaging (1/3/10)', required=True)
my_parser.add_argument('-window_size', help='side of sliding window in time points', required=True)
my_parser.add_argument('-category', help='which confound-corrected data? (len/freq/lexgram)', required=True)
my_parser.add_argument('-seed', help='integer for the random seed', required=True)
my_parser.add_argument('-folder', help='folder containing the confound-corrected data', required=True)
my_parser.add_argument('-bc_type', help='either -ss- or -mnn-', required=True)
my_parser.add_argument('-ica_type', help='fo', required=True) # used mainly in filename saving
args = my_parser.parse_args()

# ica_type = [ica_weak_sent_bc / ica_weak_epoch_bc / ica_weak_no_bc]

# Extract arguments
category = str(args.category)
n_avg = int(args.n_avg)
folder = str(args.folder)
data_folder = Path(folder)
window_size = int(args.window_size)
n_timepoints = 176
seed = int(args.seed)
# Explicitly set seed
np.random.seed(seed)
ica_type = str(args.ica_type)
bc_type = str(args.bc_type)
category2 = category+'_'+str(ica_type)
standard_scaler = bc_type == "ss" 

BATCH_SIZE = 256
N_ELECTRODES = 64
N_CLASSES = 2

# Make folder to store the results if not already present, ignore if exists already
Path(f'./exp1_confound_corrected/{category}_results/').mkdir(parents=True, exist_ok=True)

print('INFO:')
print('category = {}'.format(category))
print('n_avg = {}'.format(n_avg))
print('window_size = {}'.format(window_size))
print('n_timepoints= {}'.format(n_timepoints))
print('seed = {}'.format(seed))
print('bc_type = {}'.format(bc_type))
print('ica_type = {}'.format(ica_type))
print('folder = {}'.format(folder))

# Load data
train_data = np.load(data_folder / Path('{}_avg{}_train_data.npy'.format(category2, n_avg)))
train_labels = np.load(data_folder / Path('{}_avg{}_train_labels.npy'.format(category2, n_avg)))

dev_data = np.load(data_folder / Path('{}_avg{}_dev_data.npy'.format(category2, n_avg)))
dev_labels = np.load(data_folder / Path('{}_avg{}_dev_labels.npy'.format(category2, n_avg)))

test_data = np.load(data_folder/ Path('{}_avg{}_test_data.npy'.format(category2, n_avg)))
test_labels = np.load(data_folder / Path('{}_avg{}_test_labels.npy'.format(category2, n_avg)))

print(f'train_data shape: {train_data.shape}')
print(f'dev_data shape: {dev_data.shape}')
print(f'test_data shape: {test_data.shape}')

def run(X_train, y_train, X_dev, y_dev, X_test, y_test,
        window_size, start_idx, end_idx, seed):
    
    bestmodel_loc = os.path.join(f'exp1_confound_corrected',f'{category}_results','bestmodel',
        f'SVM_{category}_avg{n_avg}_bestmodel_ws{window_size}_win{start_idx}-{end_idx}_seed{seed}_{ica_type}.pkl')

    SGD_clf = SGDClassifier(loss='hinge', tol=1e-3, n_jobs=-1, alpha=0.75)
    classes = [0,1]
    best_score = 0
    best_model = -1

    for epoch_i in range(4):

        # Shuffle every epoch
        idx = np.arange(len(y_train))
        np.random.shuffle(idx)
        X_train_ = X_train[idx]
        y_train_ = y_train[idx]
        
        # train over batches
        for batch_i in range(len(X_train_) // BATCH_SIZE):
            
            X_train_batch = X_train_[batch_i : batch_i+BATCH_SIZE]
            y_train_batch = y_train_[batch_i : batch_i+BATCH_SIZE]

            SGD_clf.partial_fit(X_train_batch, y_train_batch, classes=classes)
            
            # check dev score after each update and save highest-scoring model
            score_dev = SGD_clf.score(X_dev, y_dev)

            if score_dev > best_score:
                best_score = score_dev
                pickle.dump(SGD_clf, open(bestmodel_loc, "wb"))

    model = pickle.load(open(bestmodel_loc, "rb"))
    score_train = model.score(X_train_, y_train_) # over entire train set
    score_dev = model.score(X_dev, y_dev) # over entire train set
    score_test = model.score(X_test, y_test)
    print('SVM: Train set score: {}, Test score: {}'.format(score_train, score_test))
    
    return (score_dev, score_test, model)


def main(X_train, y_train, X_dev, y_dev, X_test, y_test, window_size, seed, std_scaler=False):

    n_timepoints = X_train.shape[2]
    assert n_timepoints == 176
    dev_scores = []
    test_scores = []

    # 0-160, shift by 1 each time
    start_idxs = range(0,n_timepoints-window_size,1) 

    for start_idx in start_idxs:

        end_idx = start_idx + window_size
        print('start = {}, end = {}'.format(start_idx, end_idx))

        # Isolate the temporal window in third axis
        X_train_subset = X_train[:,:, start_idx:end_idx]
        X_dev_subset   = X_dev[:,:,   start_idx:end_idx]
        X_test_subset  = X_test[:,:,  start_idx:end_idx]

        # Flatten to 2D to be compatible with SVM
        X_train_flat = X_train_subset.reshape((len(X_train), -1))
        X_dev_flat   = X_dev_subset.reshape((len(X_dev), -1))
        X_test_flat  = X_test_subset.reshape((len(X_test), -1))
        
        # Use StandardScaler if requested (otherwise assume data already scaled)
        if std_scaler:
            X_train_flat = StandardScaler().fit_transform(X_train_flat)
            X_dev_flat = StandardScaler().fit_transform(X_dev_flat)
            X_test_flat = StandardScaler().fit_transform(X_test_flat)

        results = run(X_train_flat, train_labels,
                             X_dev_flat, dev_labels,
                             X_test_flat, test_labels,
                             window_size, start_idx, end_idx, seed)

        assert len(results) == 3
        dev_score,  test_score, clf = results
        dev_scores.append(dev_score)
        test_scores.append(test_score)

    dev_filename_scores = f"exp1_confound_corrected/{category}_results/SVM_{category}_avg{n_avg}_window_size{window_size}_dev_seed{seed}_{ica_type}_{bc_type}_scores.pkl"

    test_filename_scores = f"exp1_confound_corrected/{category}_results/SVM_{category}_avg{n_avg}_window_size{window_size}_test_seed{seed}_{ica_type}_{bc_type}_scores.pkl"

    pickle.dump(dev_scores, open(dev_filename_scores, "wb"))
    pickle.dump(test_scores, open(test_filename_scores, "wb"))

    print(dev_scores)
    print(test_scores)

main(train_data, train_labels, dev_data, dev_labels, test_data, test_labels, window_size=window_size,
   seed=seed, std_scaler=standard_scaler)