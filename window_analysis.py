# This code is primarily for BlueBEAR analysis, with windows starting at t=0 

import numpy as np
import glob
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from multivariate_normalisation import multivariate_noise_norm
import pickle
import os
import argparse

np.random.seed() # Pick random new seed for scikit-learn (uses NumPy's seed)
my_parser = argparse.ArgumentParser()

my_parser.add_argument('-n_avg', help='a first argument', required=True)
my_parser.add_argument('-model_type', help='a second argument', required=True)
my_parser.add_argument('-window_size', help='a second argument', required=True)
#my_parser.add_argument('-run_num', help='a second argument', required=True) # run 10 times by default now
args = my_parser.parse_args()

print(args.n_avg)
print(args.model_type)
print(args.window_size)
#print(args.run_num)

# Settings
N_AVG = int(args.n_avg)
category = os.path.join("bigram_6class", "500k_train")
data_folder = Path('../data/{}'.format(category))
END = int(args.window_size)
window_size = int(args.window_size)
model_type = args.model_type
n_timepoints = 226
#run_num = args.run_num

print('INFO:')
print('n_avg = {}'.format(N_AVG))
print('window_size = {}'.format(window_size))
print('model_type = {}'.format(model_type))
print('n_timepoints= {}'.format(n_timepoints))
#print('run num = {}'.format(run_num))

def preprocess_data(N_AVG, train_data, train_labels, dev_data, dev_labels, test_data,
                    test_labels, start, end, test_wr_data=None, test_wr_labels=None):
    
    # Collapse along electrode & temporal dims for (n_samples, n_features)
    n_samples, n_electrodes, n_timepoints = train_data.shape
    train_data = train_data[:,:,start:end]
    #train_data_flat = train_data_scaled.reshape((n_samples, -1))

    n_samples, n_electrodes, n_timepoints = dev_data.shape
    dev_data = dev_data[:,:,start:end]
    #dev_data_flat = dev_data_scaled.reshape((n_samples, -1)) 

    n_samples, n_electrodes, n_timepoints = test_data.shape
    test_data = test_data[:,:,start:end]
    #test_data_flat = test_data_scaled.reshape((n_samples, -1)) 

    if N_AVG > 1:
        n_samples, n_electrodes, n_timepoints = test_wr_data.shape
        test_wr_data = test_wr_data[:,:,start:end]
        #test_wr_data_flat = test_wr_data_scaled.reshape((n_samples, -1)) 
    
    train_data_scaled = multivariate_noise_norm(train_data, train_labels)
    dev_data_scaled = multivariate_noise_norm(dev_data, dev_labels)
    test_data_scaled = multivariate_noise_norm(test_data, test_labels)
    test_wr_data_scaled = multivariate_noise_norm(test_wr_data, test_wr_labels)
        
    # Print shapes
    print(f'train_data shape: {train_data_scaled.shape}')
    print(f'dev_data shape: {dev_data_scaled.shape}')
    print(f'test_data shape: {test_data_scaled.shape}')
    print(f'test_wr_data shape: {test_wr_data.shape if N_AVG > 1 else 0}')
        
    # Return
    return (train_data_scaled, train_labels, dev_data_scaled, dev_labels,
           test_data_scaled, test_labels, test_wr_data_scaled, test_wr_labels)



BATCH_SIZE = 256
N_ELECTRODES = 64
N_CLASSES = 6
START = 0

print(f'N_CLASSES = {N_CLASSES}, N_AVG = {N_AVG}')
data_folder      = Path('../data/no_ica/{}'.format(category))

#category_name = "bigram_10class"
category_name = "bigram_6class"

train_data = np.load(data_folder / Path('{}_avg{}_train_data.npy'.format(category_name, N_AVG)))
train_labels = np.load(data_folder / Path('{}_avg{}_train_labels.npy'.format(category_name, N_AVG)))
#np.random.shuffle(train_labels)

dev_data = np.load(data_folder / Path('{}_avg{}_dev_data.npy'.format(category_name, N_AVG)))
dev_labels = np.load(data_folder / Path('{}_avg{}_dev_labels.npy'.format(category_name, N_AVG)))
#np.random.shuffle(dev_labels)

test_data_type = "test_wr" if N_AVG > 1 else "test"
test_data = np.load(data_folder/ Path('{}_avg{}_{}_data.npy'.format(category_name, N_AVG, test_data_type)))
test_labels = np.load(data_folder / Path('{}_avg{}_{}_labels.npy'.format(category_name, N_AVG, test_data_type)))
#np.random.shuffle(test_labels)

print(f'train_data shape: {train_data.shape}')
print(f'dev_data shape: {dev_data.shape}')
print(f'test_data shape: {test_data.shape}')

               
(train_data_scaled, train_labels,
 dev_data_scaled, dev_labels,
 test_data_scaled, test_labels,
 test_wr_data_scaled, test_wr_labels) = preprocess_data(N_AVG,
                                    train_data, train_labels,
                                    dev_data, dev_labels,
                                    test_data, test_labels,
                                    START, END, test_data,
                                    test_labels)


def run(X_train, y_train, X_dev, y_dev, X_test, y_test, window_size):
    
    # Set up model
    model_type = 'SVM_noica'
    category = 'bigram'
    classes = [i for i in range(10)] #[0,1,2,3]
    scores = []
    best_global_run = -1
    best_global_dev = -1
    best_test = -1

    for run_num in range(10):
        SGD_clf = SGDClassifier(loss='hinge', tol=1e-3, alpha=0.75)
        score = 0
        best_score = 0

        for i in range(10):
            # Shuffle
            idx = np.arange(len(y_train))
            np.random.shuffle(idx)
            X_train = X_train[idx]
            y_train = y_train[idx]

            SGD_clf.partial_fit(X_train, y_train, classes=classes)
            y_pred = SGD_clf.predict(X_dev)
            score = sum(y_pred == y_dev) / len(X_dev)
            print('{} : epoch ({}), score = {}%'.format(model_type, i, score*100))

            if score > best_score:
                best_score = score
                #print('saving new best_model')
                model_filename = os.path.join('saved_models',"{}_{}_avg{}_bestmodel_start{}_end{}_run{}.pkl".format(model_type, category, N_AVG, START, END, run_num) )
                pickle.dump(SGD_clf, open(model_filename, "wb"))
                
            if score > best_global_dev:
                best_global_run = run_num
                best_global_dev = score
                best_test = SGD_clf.score(X_test, y_test)

        model_filename = os.path.join('saved_models',"{}_{}_avg{}_bestmodel_start{}_end{}_run{}.pkl".format(model_type, category, N_AVG, START, END, run_num) )
        model = pickle.load(open(model_filename, "rb"))
        score = model.score(X_test, y_test)
        dev_score = model.score(X_dev, y_dev)
        print('{}: Run {}:  Dev set score: {}, Test set score: {}'.format(model_type, run_num, dev_score, score))
        scores.append(score)
        
        predictions = model.predict(X_test)
        svm_bin_vec = predictions == y_test
        to_write = np.array([svm_bin_vec, predictions, y_test]).T
        print(to_write.shape)
        bin_vec_filename = os.path.join('saved_predictions', '{}_{}_avg{}_start{}_end{}_run{}_bin_vec.txt'.format(model_type, category, N_AVG, START, END, run_num))
        np.savetxt(bin_vec_filename, to_write)
        
        print('{}-{}: Best run: {}, best dev score: {:.3f}, corresponding test: {:.3f} (avg test: {:.3f})'.format(START, END, best_global_run, (best_global_dev*100), (best_test*100), np.mean(scores)))
    return scores

def main(X_train, y_train, X_dev, y_dev, X_test, y_test, window_size):

    scores = run(X_train, y_train,
                         X_dev, y_dev,
                         X_test, y_test,
                         window_size)