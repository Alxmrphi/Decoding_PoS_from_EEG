import glob
import numpy as np
import os
import pickle
#from multivariate_normalisation import multivariate_noise_norm
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


classes = ['NOUN', 'ADP', 'VERB', 'PRON', 'AUX', 'ADV']
BATCH_SIZE = 512
N_ELECTRODES = 64
model_type = 'SVM'
filename_stem = "unigram_6class"
misc = "small_"

def load_data(N_AVG, filename_stem):

    train_data_ = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_train_{misc}data.npy')
    train_labels = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_train_{misc}labels.npy')

    dev_data_ = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_dev_small_data.npy')
    dev_labels = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_dev_small_labels.npy')

    test_data_ = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_test_small_data.npy')
    test_labels = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_test_small_labels.npy')

    if N_AVG > 1:
        test_wr_data = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_test_wr_small_data.npy')
        test_wr_labels = np.load(f'../data/ica_mid/unigram_6class/restricted/{filename_stem}_avg{N_AVG}_test_wr_small_labels.npy')
    else:
        test_wr_data = None
        test_wr_labels = None

    print(f'train_data shape: {train_data_.shape}, len(train_labels) == {len(train_labels)}')
    print(f'dev_data shape: {dev_data_.shape}, len(dev_labels) == {len(dev_labels)}')
    print(f'test_data shape: {test_data_.shape}, len(test_labels) == {len(test_labels)}')
    print(f'test_wr_data shape: {test_wr_data.shape if N_AVG > 1 else 0}')
        
    return (train_data_, train_labels, dev_data_, dev_labels,
           test_data_, test_labels, test_wr_data, test_wr_labels)
           
def preprocess_data(N_AVG, train_data, train_labels, dev_data, dev_labels, test_data,
                    test_labels, start, end, test_wr_data=None, test_wr_labels=None):
    
    # Collapse along electrode & temporal dims for (n_samples, n_features)
    n_samples, n_electrodes, n_timepoints = train_data.shape
    train_data = train_data[:,:,start:end]

    n_samples, n_electrodes, n_timepoints = dev_data.shape
    dev_data = dev_data[:,:,start:end]
    
    n_samples, n_electrodes, n_timepoints = test_data.shape
    test_data = test_data[:,:,start:end]

    if N_AVG > 1:
        n_samples, n_electrodes, n_timepoints = test_wr_data.shape
        test_wr_data = test_wr_data[:,:,start:end]
    
    
    train_data_flat = train_data.reshape((len(train_data), -1))
    dev_data_flat = dev_data.reshape((len(dev_data), -1))
    test_data_flat = test_data.reshape((len(test_data), -1))
    test_wr_data_flat = test_wr_data.reshape((len(test_wr_data), -1))
        
    # Print shapes
    print(f'train_data shape: {train_data_flat.shape}')
    print(f'dev_data shape: {dev_data_flat.shape}')
    print(f'test_data shape: {test_data_flat.shape}')
    print(f'test_wr_data shape: {test_wr_data_flat.shape if N_AVG > 1 else 0}')
        
    # Return
    return (train_data_flat, train_labels, dev_data_flat, dev_labels,
           test_data_flat, test_labels, test_wr_data_flat, test_wr_labels)
           
           
           
def run_model(n_iter, n_epoch, N_AVG, train_data, train_labels, dev_data, dev_labels,
             test_data, test_labels, test_wr_data=None, test_wr_labels=None,
              filename_stem=None, n_classes=None, start=None, end=None):
    
    classes = [i for i in range(n_classes)]
    category = filename_stem
    best_global_run = -1
    best_global_dev = -1
    best_test = -1
    #print('train_data.shape = ', train_data.shape, ' train_labels.shape = ', train_labels.shape)
    
    for run_num in range(n_iter):

        SGD_clf = SGDClassifier(loss='hinge', tol=1e-3, alpha=0.75)
        score = 0
        best_score = 0
        dev_scores = []

        for i in range(n_epoch):
        
            # Shuffle
            idx = np.arange(len(train_labels))
            np.random.shuffle(idx)
            train_data = train_data[idx]
            train_labels = train_labels[idx]
            
            n_steps = len(train_data) // BATCH_SIZE
            
            for i in range(0, len(train_data)+1, BATCH_SIZE):
                train_data_ = train_data[i:i+BATCH_SIZE]
                train_labels_ = train_labels[i:i+BATCH_SIZE]
                
                #print('train_data_.shape = ', train_data_.shape, ' train_labels_.shape = ',
                # train_labels_.shape)

                SGD_clf.partial_fit(train_data_, train_labels_, classes=classes)
                y_pred = SGD_clf.predict(dev_data)
                dev_score = sum(y_pred == dev_labels) / len(dev_data)
                print(f'{model_type} : epoch ({i}), score = {dev_score*100}%')
                dev_scores.append(dev_score)

                if dev_score > best_score:
                    best_score = dev_score
                    #print(f'saving new best_model (run {run_num})')
                    pickle.dump(SGD_clf, open(os.path.join("saved_models",
                    "{}_{}_avg{}_start{}_end{}_{}run{}.pkl".format(model_type, category, N_AVG,
                                                                    start, end, misc, run_num)), "wb"))

                if dev_score > best_global_dev:
                    best_global_run = run_num
                    best_global_dev = dev_score
                    best_test = SGD_clf.score(test_data, test_labels)
                
        model = pickle.load(open(os.path.join("saved_models",
                "{}_{}_avg{}_start{}_end{}_{}run{}.pkl".format(model_type, category, N_AVG,
                                                               start, end, misc, run_num)), "rb"))
        test_score = model.score(test_data, test_labels)
        print('......')
        print('{}: Test set score (run {}): {}'.format(model_type, run_num, test_score))
        predictions = model.predict(test_data)
        binary_scores = predictions == test_labels
        to_write = np.array([binary_scores, predictions, test_labels]).T
        np.savetxt(os.path.join("saved_predictions",
            f"{model_type}_{category}_avg{N_AVG}_start{start}_end{end}_{misc}run{run_num}_bin_vec.txt"), to_write)

        if N_AVG > 1:
            test_score = model.score(test_wr_data, test_wr_labels)
            print('{}: Test WR set score (run {}): {}'.format(model_type, run_num, test_score))
            print('......')
            predictions = model.predict(test_wr_data)
            binary_scores = predictions == test_wr_labels
            to_write = np.array([binary_scores, predictions, test_wr_labels]).T
            np.savetxt(os.path.join("saved_predictions",
                f"{model_type}_{category}_avg{N_AVG}_start{start}_end{end}_{misc}run{run_num}_wr_bin_vec.txt"), to_write)
            
    return best_global_run, best_global_dev, best_test
    
    
def run_experiment(N_AVG, start, end, filename_stem, n_classes):
    
    (train_data, train_labels,
     dev_data, dev_labels,
     test_data, test_labels,
    test_wr_data, test_wr_labels) = load_data(N_AVG, filename_stem)
    
    (train_data_scaled, train_labels,
     dev_data_scaled, dev_labels,
     test_data_scaled, test_labels,
     test_wr_data_scaled, test_wr_labels) = preprocess_data(N_AVG,
                                        train_data, train_labels,
                                        dev_data, dev_labels,
                                        test_data, test_labels,
                                        start, end, test_wr_data,
                                        test_wr_labels)
                                        
    print('train_data.shape = ', train_data_scaled.shape, ' train_labels.shape = ', train_labels.shape)
    
    r, d, t = run_model(20, 1, N_AVG, train_data_scaled, train_labels,
                                  dev_data_scaled, dev_labels,
                                  test_data_scaled, test_labels,
                                  test_wr_data_scaled, test_wr_labels,
                                  filename_stem, n_classes, start=start, end=end)
    
    print(f'Best run: {r}, best dev score: {d}, corresponding test: {t}')
    
run_experiment(N_AVG=10, start=0, end=176, filename_stem="unigram_6class", n_classes=6)