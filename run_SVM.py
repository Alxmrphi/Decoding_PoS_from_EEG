import glob
import numpy as np
import os
import pickle
from multivariate_normalisation import multivariate_noise_norm
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


classes = ['NOUN', 'ADP', 'VERB', 'PRON', 'AUX', 'ADV']
BATCH_SIZE = 256
N_ELECTRODES = 64
model_type = 'SVM'
filename_stem = "bigram_6class"

def load_data(N_AVG: int, filename_stem: str) -> tuple:
    """ Load train, dev & test data (optionally test_wr), print shapes and return tuple
    
    Args:
        N_AVG:           Pseudotrial averaging number (for correct filename identification)
        filename_stem:   Each experiment has its own designated naming convention, for filenames
    Returns:
        loaded_data:     A tuple containing all the loaded data
    """

    train_data_ = np.load(f'{filename_stem}_avg{N_AVG}_train_data.npy')
    train_labels = np.load(f'{filename_stem}_avg{N_AVG}_train_labels.npy')

    dev_data_ = np.load(f'{filename_stem}_avg{N_AVG}_dev_data.npy')
    dev_labels = np.load(f'{filename_stem}_avg{N_AVG}_dev_labels.npy')

    test_data_ = np.load(f'{filename_stem}_avg{N_AVG}_test_data.npy')
    test_labels = np.load(f'{filename_stem}_avg{N_AVG}_test_labels.npy')

    if N_AVG > 1:
        test_wr_data = np.load(f'{filename_stem}_avg{N_AVG}_test_wr_data.npy')
        test_wr_labels = np.load(f'{filename_stem}_avg{N_AVG}_test_wr_labels.npy')
    else:
        test_wr_data = None
        test_wr_labels = None

    print(f'train_data shape: {train_data_.shape}, len(train_labels) == {len(train_labels)}')
    print(f'dev_data shape: {dev_data_.shape}, len(dev_labels) == {len(dev_labels)}')
    print(f'test_data shape: {test_data_.shape}, len(test_labels) == {len(test_labels)}')
    print(f'test_wr_data shape: {test_wr_data.shape if N_AVG > 1 else 0}')
        
    loaded_data = (train_data_, train_labels, dev_data_, dev_labels,
                   test_data_, test_labels, test_wr_data, test_wr_labels)
    
    return loaded_data


def preprocess_data(N_AVG, train_data, train_labels, dev_data, dev_labels, test_data,
                    test_labels, start, end, test_wr_data=None, test_wr_labels=None) -> tuple:
    """ Take previously loaded data (already MNN-corrected externally), extract time point of interest
    and then flatten data, before returning """
    
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
    """ Run the SVM model using specified data and save / print results during training

    Args:
        n_iter:             How many repetitions to perform
        n_epoch:            How many epochs to trial a single analysis for
        N_AVG:              Number of pseudotrials for corresponding data
        ....                
        filename_stem:      Filename specifier for saving corresponding results
        n_classes:          How many classes for the problem (required for SGDClassifier)
        start:              Start point of analysis window for EEG window
        end:                End point of analysis window for EEG window
    """
    
    classes = [i for i in range(n_classes)]
    category = filename_stem
    best_global_run = -1
    best_global_dev = -1
    best_test = -1
    
    for run_num in range(n_iter):

        SGD_clf = SGDClassifier(loss='hinge', tol=1e-3, alpha=0.75)
        score = 0
        best_score = 0

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

                SGD_clf.partial_fit(train_data_, train_labels_, classes=classes)
                y_pred = SGD_clf.predict(dev_data)
                score = sum(y_pred == dev_labels) / len(dev_data)
                print(f'{model_type} : epoch ({i}), score = {score*100}%')

                if score > best_score:
                    best_score = score
                    #print(f'saving new best_model (run {run_num})')
                    pickle.dump(SGD_clf, open(os.path.join("saved_models", "{}_{}_avg{}_start{}_end{}_run{}.pkl".format(model_type, category, N_AVG, start, end, run_num)), "wb"))

                if score > best_global_dev:
                    best_global_run = run_num
                    best_global_dev = score
                    best_test = SGD_clf.score(test_data, test_labels)
                
        model = pickle.load(open(os.path.join("saved_models", "{}_{}_avg{}_start{}_end{}_run{}.pkl".format(model_type, category, N_AVG, start, end, run_num)), "rb"))
        score = model.score(test_data, test_labels)
        print('......')
        print('{}: Test set score (run {}): {}'.format(model_type, run_num, score))
        predictions = model.predict(test_data)
        binary_scores = predictions == test_labels
        #to_write = np.array([binary_scores, predictions, test_labels]).T
        #np.savetxt(os.path.join("saved_predictions", f"{model_type}_{category}_avg{N_AVG}_start{start}_end{end}_run{run_num}_bin_vec.txt"), to_write)

        if N_AVG > 1:
            score = model.score(test_wr_data, test_wr_labels)
            print('{}: Test WR set score (run {}): {}'.format(model_type, run_num, score))
            print('......')
            predictions = model.predict(test_wr_data)
            binary_scores = predictions == test_wr_labels
            #to_write = np.array([binary_scores, predictions, test_wr_labels]).T
            #np.savetxt(os.path.join("saved_predictions", f"{model_type}_{category}_avg{N_AVG}_start{start}_end{end}_run{run_num}_wr_bin_vec.txt"), to_write)
            
    return best_global_run, best_global_dev, best_test

def run_experiment(N_AVG, start, end, filename_stem, n_classes) -> None:
    """ This ties together all the functions specified above and runs the model fitting & eval

    Args:  
        N_AVG:              Number of pseudotrials for corresponding data
        start:              Start point of analysis window for EEG window
        end:                End point of analysis window for EEG window
        filename_stem:      Filename specifier for saving corresponding results
        n_classes:          How many classes for the problem (required for SGDClassifier)
    """
    
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
    
    r, d, t = run_model(10, 5, N_AVG, train_data_scaled, train_labels,
                                  dev_data_scaled, dev_labels,
                                  test_data_scaled, test_labels,
                                  test_wr_data_scaled, test_wr_labels,
                                  filename_stem, n_classes, start=start, end=end)
    
    print(f'Best run: {r}, best dev score: {d}, corresponding test: {t}')


# --- Examples
# run_experiment(N_AVG=10, start=0, end=76, filename_stem="bigram_6class", n_classes=6)
# run_experiment(N_AVG=10, start=76, end=151, filename_stem="bigram_6class", n_classes=6)
# run_experiment(N_AVG=10, start=0, end=151, filename_stem="bigram_6class", n_classes=6)
# run_experiment(N_AVG=10, start=0, end=76, filename_stem="bigram_6class", n_classes=6)