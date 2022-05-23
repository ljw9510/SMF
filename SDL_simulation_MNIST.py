import re
import os
import numpy as np
import pandas as pd

from tqdm import trange
import matplotlib.pyplot as plt
from src.SNMF import SNMF, update_code_within_radius
from src.LMF import LMF
from src.SDL_SVP import SDL_SVP

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import SparseCoder
from sklearn.metrics import roc_curve
from scipy.spatial import ConvexHull

from sklearn.datasets import fetch_openml
from PIL import Image, ImageOps
from pneumonia_dataprocess import process_path

import seaborn as sns
sns.set_theme()

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    #"font.size"   : 15,
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
        "\\usepackage{units}",  # load additional packages
        "\\usepackage{metalogo}",
        "\\usepackage{unicode-math}",  # unicode math setup
        r"\setmathfont{xits-math.otf}",
        r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ])
})


def coding(X, W, H0,
          r=None,
          a1=0, #L1 regularizer
          a2=0, #L2 regularizer
          sub_iter=[5],
          stopping_grad_ratio=0.0001,
          nonnegativity=True,
          subsample_ratio=1):
    """
    Find \hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) within radius r from H0
    Use row-wise projected gradient descent
    """
    if H0 is None:
        H0 = np.random.rand(W.shape[1],X.shape[1])

    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[1])
    if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
        idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
    A = W.T @ W ## Needed for gradient computation

    grad = W.T @ (W @ H0 - X)
    while (i < np.random.choice(sub_iter)):
        step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
        H1 -= step_size * grad
        if nonnegativity:
            H1 = np.maximum(H1, 0)  # nonnegativity constraint
        i = i + 1
        # print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H1)**2))
    return H1

def ALS(X,
        n_components = 10, # number of columns in the dictionary matrix W
        n_iter=100,
        a0 = 0, # L1 regularizer for H
        a1 = 0, # L1 regularizer for W
        a12 = 0, # L2 regularizer for W
        H_nonnegativity=True,
        W_nonnegativity=True,
        compute_recons_error=False,
        subsample_ratio = 10):

        '''
        Given data matrix X, use alternating least squares to find factors W,H so that
                                || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
        is minimized (at least locally)
        '''

        d, n = X.shape
        r = n_components

        #normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
        #print('!!! avg entry of X', normalization)
        #X = X/normalization

        # Initialize factors
        W = np.random.rand(d,r)
        H = np.random.rand(r,n)
        # H = H * np.linalg.norm(X) / np.linalg.norm(H)
        for i in trange(n_iter):
            #H = coding_within_radius(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            #W = coding_within_radius(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            H = coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
            W = coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
            if compute_recons_error and (i % 10 == 0) :
                print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
        return W, H

# sigmoid and logit function
def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
def generate_Y(H, Beta, n):
    Y = np.zeros(shape=[1,n])
    p = sigmoid(Beta @ H - np.mean(Beta @ H))
    # print('p')
    # print('p.shape', p.shape)
    for i in range(n):
        U = np.random.rand()
        if U < p[0,i]:
            Y[0,i] = 1
    print('proportion of 1s:', np.sum(Y)/n)
    return Y
"""
def generate_Y(H, Beta, n):

    # n x 1 vector

    Y = np.zeros(n)
    prob = sigmoid(H @ Beta - np.mean(H @ Beta))

    for i in range(n):
        U = np.random.rand()
        if U < prob[i]:
            Y[i] = 1
    print('proportion of 1s:', np.sum(Y)/n)
    return Y

def compute_accuracy_metrics(Y_test, P_pred, train_data=None, verbose=False):
    # y_test = binary label
    # P_pred = predicted probability for y_test
    # train_data = [X_train, ]
    # compuate various binary classification accuracy metrics
    # Compute classification statistics

    if train_data is not None:
        Y_train, P_train = train_data
        fpr, tpr, thresholds = metrics.roc_curve(Y_train, P_train, pos_label=None)
        mythre = thresholds[np.argmax(tpr - fpr)]
        myauc = round(metrics.auc(fpr, tpr), 4)
        print('threshold from training set used:', mythre)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, P_pred, pos_label=None)
        mythre = thresholds[np.argmax(tpr - fpr)]
        myauc_test = round(metrics.auc(fpr, tpr), 4)
        print('!!! test AUC:', myauc_test)

    threshold = round(mythre, 4)

    Y_pred = P_pred.copy()
    Y_pred[Y_pred < threshold] = 0
    Y_pred[Y_pred >= threshold] = 1

    mcm = confusion_matrix(Y_test, Y_pred)
    tn = mcm[0, 0]
    tp = mcm[1, 1]
    fn = mcm[1, 0]
    fp = mcm[0, 1]

    accuracy = round( (tp + tn) / (tp + tn + fp + fn), 4)
    misclassification = 1 - accuracy
    sensitivity = round(tp / (tp + fn), 4)
    specificity = round(tn / (tn + fp), 4)
    precision = round(tp / (tp + fp), 4)
    recall = round(tp / (tp + fn), 4)
    fall_out = round(fp / (fp + tn), 4)
    miss_rate = round(fn / (fn + tp), 4)
    F_score = round(2 * precision * recall / ( precision + recall ), 4)

    # Save results
    results_dict = {}
    results_dict.update({'Y_test': Y_test})
    results_dict.update({'Y_pred': Y_pred})
    results_dict.update({'AUC': myauc})
    results_dict.update({'Opt_threshold': mythre})
    results_dict.update({'Accuracy': accuracy})
    results_dict.update({'Sensitivity': sensitivity})
    results_dict.update({'Specificity': specificity})
    results_dict.update({'Precision': precision})
    #results_dict.update({'Fall_out': fall_out})
    #results_dict.update({'Miss_rate': miss_rate})
    results_dict.update({'F_score': F_score})


    if verbose:
        for key in [key for key in results_dict.keys() if key not in ['Y_test', 'Y_pred']]:
            print('% s ===> %.3f' % (key, results_dict.get(key)))
    return results_dict

def list2onehot(y, list_classes):
    """
    y = list of class lables of length n
    output = n x k array, i th row = one-hot encoding of y[i] (e.g., [0,0,1,0,0])
    """
    Y = np.zeros(shape = [len(y), len(list_classes)], dtype=int)
    for i in np.arange(Y.shape[0]):
        for j in np.arange(len(list_classes)):
            if y[i] == list_classes[j]:
                Y[i,j] = 1
    return Y

def onehot2list(y, list_classes=None):
    """
    y = n x k array, i th row = one-hot encoding of y[i] (e.g., [0,0,1,0,0])
    output =  list of class lables of length n
    """
    if list_classes is None:
        list_classes = np.arange(y.shape[1])

    y_list = []
    for i in np.arange(y.shape[0]):
        idx = np.where(y[i,:]==1)
        idx = idx[0][0]
        y_list.append(list_classes[idx])
    return y_list

def sample_MNIST(X, y, list_digits = ['1', '2'], basis_size=None):

    # get subset of data from MNIST of given digits
    # basis_size = [r1, r2]

    Y = list2onehot(y.tolist(), list_digits)

    ## Sampling
    idx = []
    for j in range(len(list_digits)):
      idx0 = [i for i in range(len(y)) if y[i] == list_digits[j]]
      if basis_size is None:
        idx = idx + idx0
      if basis_size is not None:
        idx = idx + list(np.random.choice(idx0, basis_size[j], replace = False))

    X0 = X[idx,:]
    y0 = Y[idx,:]

    return X0, y0

def sample_MNIST_old(list_digits=['0','1', '2'], full_MNIST=None, padding_thickness=0, sample_size=None):
    # get train and test set from MNIST of given digits
    # e.g., list_digits = ['0', '1', '2']
    # pad each 28 x 28 image with zeros so that it has now "padding_thickness" more rows and columns
    # The original image is superimposed at a uniformly chosen location
    if full_MNIST is not None:
        X, y = full_MNIST
    else:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.
        X = np.asarray(X) # it might load as pandas dataframe
    Y = list2onehot(y.tolist(), list_digits)

    idx = [i for i in np.arange(len(y)) if y[i] in list_digits] # list of indices where the label y is in list_digits
    if sample_size is not None:
        idx = np.random.choice(idx, sample_size, replace=False)

    X01 = X[idx,:]
    y01 = Y[idx,:]

    X_train = []
    y_train = []
    for i in trange(X01.shape[0]):
        # for each example i, make it into train set with probabiliy 0.8 and into test set otherwise
        # U = np.random.rand() # Uniform([0,1]) variable
        img_padded = random_padding(X01[i,:].reshape(28,28), thickness=padding_thickness)
        img_padded_vec = img_padded.reshape(1,-1)
        X_train.append(img_padded_vec[0,:].copy())
        y_train.append(y01[i,:].copy())

    X_train = np.asarray(X_train).T
    y_train = np.asarray(y_train).T
    return X_train,  y_train

## Data Generation (Beat separate NMF + logistic regression by joint optimization)

def sim_data_gen_MNIST(r = [2, 2], n = 1000,
                       digits_X = ['2', '7'],
                       digits_Y = ['1', '4'],
                       noise_std = 0,
                       random_seed = 1):

    np.random.seed(random_seed)

    # Load data
    X, y = fetch_openml('mnist_784', version = 1, return_X_y = True) # pandas dataframe
    X = X / 255.
    X = np.asarray(X)

    W_true, temp = sample_MNIST(X, y, list_digits = digits_X, basis_size = r)
    W_true_Y, y0 = sample_MNIST(X, y, list_digits = digits_Y, basis_size = r)
    p = W_true.shape[1]

    Beta_true = (2*y0[:,0]-1) # 1 if Y = first digit, -1 if Y = second digit
    print('Beta_true', Beta_true)

    Sigma = ((noise_std)**2) * np.eye(p)
    Noise = np.random.multivariate_normal(np.zeros(p),  Sigma, n)

    H_true = np.random.rand(n, sum(r))
    X = H_true @ W_true

    Y = generate_Y(X @ W_true.T, Beta_true, n) # true labels
    X += Noise # corrupt true signal

    X_train, X_test, Y_train, Y_test, H_train, H_test = train_test_split(X, Y, H_true, test_size = 0.2)

    X, X_train, X_test = X.T, X_train.T, X_test.T
    H_true, H_train, H_test = H_true.T, H_train.T, H_test.T
    Y, Y_train, Y_test = Y[np.newaxis,:], Y_train[np.newaxis,:], Y_test[np.newaxis,:]
    W_true, W_true_Y, Beta_true = W_true.T, W_true_Y.T, Beta_true[np.newaxis,:]

    print('X_train.shape', X_train.shape)
    print('Y_train.shape', Y_train.shape)
    print('X_test.shape', X_test.shape)
    print('Y_test.shape', Y_test.shape)

    return X_train, X_test, Y_train, Y_test, W_true, W_true_Y, H_true, H_train, H_test, Beta_true


def sim_data_gen_MNIST_old(r=2, n=1000,
                       digits_X = ['2', '7'],
                       digits_Y=['1', '4'],
                       full_MNIST = None,
                       noise_std = 0.5,
                       random_seed=1,
                       padding_thickness=0):
    np.random.seed(random_seed)

    W_true, y_train = sample_MNIST(list_digits=digits_X,
                                   sample_size=r,
                                   padding_thickness=padding_thickness,
                                   full_MNIST=full_MNIST)
    p = W_true.shape[0]

    W_true_Y, y_train = sample_MNIST(list_digits=digits_Y,
                                     sample_size=r,
                                     padding_thickness=padding_thickness,
                                     full_MNIST=full_MNIST)
    Beta_true = (2*y_train[0,:]-1) # 1 if the first digit, -1 otherwise
    Beta_true = Beta_true[np.newaxis, :]
    print('Beta_true', Beta_true)

    #print('y_train \n', y_train)
    Sigma = ((noise_std)**2) * np.eye(p)
    Noise = np.random.multivariate_normal(np.zeros(p),  Sigma, n).T

    H_true = np.random.rand(r, n)
    X = W_true @ H_true

    Y = generate_Y(W_true_Y.T @ X, Beta_true, n) # true labels

    X += Noise # corrupt true signal

    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2)
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T

    print('X_train.shape', X_train.shape)
    print('Y_train.shape', Y_train.shape)
    print('X_test.shape', X_test.shape)
    print('Y_test.shape', Y_test.shape)

    return X_train, X_test, Y_train, Y_test, W_true, H_true, X, Y, Beta_true

## Data Generation (Beat separate NMF + logistic regression by joint optimization)

def sim_data_gen(p=200, r=2, n=1000, noise_std=0, random_seed=1, use_separate_W=False):
    np.random.seed(random_seed)

    W_true = 1-2*np.random.rand(p, r)
    sum_of_cols = np.sum(abs(W_true), axis=1)
    W_true /= sum_of_cols[:, np.newaxis]


    #H_true = np.random.rand(r, n)
    #sum_of_cols = np.sum(abs(H_true), axis=1)
    #H_true /= sum_of_cols[:, np.newaxis]

    H_true = np.random.rand(r, n)
    H_true[:2,:] *= 10
    # W_true[0,:] *= 0.5
    # Beta_true = 2*(np.random.rand(1, r)-0.5)
    # Beta_true = np.zeros(shape=[1,r])
    # Beta_true[0,0] = 10
    # Beta_true *= 10
    # Beta_true = np.zeros(shape=[1,r])
    Beta_true = (1-2*np.random.rand(1,r))
    Beta_true[0,0] = 2
    Beta_true[0,1] = -2


    print('Beta_true', Beta_true)
    print('noise std', noise_std)
    Sigma = ((noise_std)**2) * np.eye(p)
    Noise = np.random.multivariate_normal(np.zeros(p),  Sigma, n).T
    print('Noise', np.mean(Noise))

    if use_separate_W:
        #W_true_Y = 1-2*np.random.rand(p, r)
        #sum_of_cols = np.sum(abs(W_true_Y), axis=1)
        #W_true_Y /= sum_of_cols[:, np.newaxis]

        W_true_Y = np.eye(p,r)
    else:
        W_true_Y = W_true

    X = W_true @ H_true
    # Y = generate_Y(W_true.T @ X, Beta_true, n)
    Y = generate_Y(W_true_Y.T @ X, Beta_true, n)
    #print('H_true', H_true)
    X = W_true @ H_true + Noise

    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.8)
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T

    print('X_train.shape', X_train.shape)
    print('Y_train.shape', Y_train.shape)
    print('X_test.shape', X_test.shape)
    print('Y_test.shape', Y_test.shape)

    #X1_train = np.vstack((np.ones(X_train.shape[1]), X_train))
    #X1_test = np.vstack((np.ones(X_test.shape[1]), X_test))
    #print('X1_train', np.linalg.norm(X1_train))

    return X_train, X_test, Y_train, Y_test, W_true, H_true, X, Y, Beta_true


def run_methods(data, n_components,
                xi_list = [0, 0.1, 1, 5, 10],
                beta_list = [1, None],
                iteration = 200, iter_avg = 2,
                save_path = None):
    # data  = [X_train, X_test, Y_train, Y_test]
    ## Cross validation plot --- MF + LR, SNMF, LR

    X_train, X_test, Y_train, Y_test = data
    r = n_components
    p = X_train.shape[0]
    results_dict_list = []


    # LR
    print('X_train.T.shape', X_train.T.shape)
    print('Y_train[0,:].shape', Y_train[0,:].shape)

    clf = LogisticRegression(random_state=0).fit(X_train.T, Y_train[0,:])
    P_train = clf.predict_proba(X_train.T)
    P_pred = clf.predict_proba(X_test.T)
    results = compute_accuracy_metrics(Y_test[0], P_pred[:,1], train_data = [Y_train[0], P_train[:,1]],
                                       verbose=True)

    results.update({'method': 'LR'})
    results.update({'xi': None})
    results.update({'beta': None})
    results.update({'Relative_reconstruction_loss (test)': 1})
    LR_AUC = results.get('Accuracy')
    results.update({'Accuracy': results.get('Accuracy')})
    results_dict_list.append(results.copy())

    # SDL_SVP_filter
    data_scale=10
    for j in range(len(xi_list)):
        xi = xi_list[j]
        for i in range(iter_avg):
            SDL_SVP_class = SDL_SVP(X=[X_train/data_scale, Y_train],  # data, label
                                    X_test=[X_test/data_scale, Y_test],
                                    #X_auxiliary = None,
                                    n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                    # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                    # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                    # ini_code = H_true,
                                    xi=xi,  # weight on label reconstruction error
                                    L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                    L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                    full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression

            results_dict_new = SDL_SVP_class.fit(iter=iteration, subsample_size=None,
                                                    beta = 0,
                                                    nu = 2,
                                                    search_radius_const=0.05,
                                                    update_nuance_param=False,
                                                    SDL_option = 'filter',
                                                    prediction_method_list = ['filter'],
                                                    fine_tune_beta = False,
                                                    if_compute_recons_error=True, if_validate=False)
            results_dict_new.update({'method': 'SDL-conv-filt'})
            results_dict_new.update({'beta': None})
            results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy (filter)')})
            results_dict_new.update({'F_score': results_dict_new.get('F_score (filter)')})
            results_dict_list.append(results_dict_new.copy())

    # SDL_SVP_feature
    data_scale=10
    prediction_method_list = ['naive']
    for j in range(len(xi_list)):
        xi = xi_list[j]
        for i in range(iter_avg):
            data_scale=500
            SDL_SVP_class = SDL_SVP(X=[X_train/data_scale, Y_train],  # data, label
                                    X_test=[X_test/data_scale, Y_test],
                                    #X_auxiliary = None,
                                    n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                    # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                    # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                    # ini_code = H_true,
                                    xi=xi,  # weight on label reconstruction error
                                    L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                    L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                    full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression

            results_dict_new = SDL_SVP_class.fit(iter=iteration, subsample_size=None,
                                                beta = 0,
                                                nu = 2,
                                                search_radius_const=0.01,
                                                update_nuance_param=False,
                                                SDL_option = 'feature',
                                                #prediction_method_list = ['naive', 'exhaustive'],
                                                prediction_method_list = prediction_method_list,
                                                fine_tune_beta = False,
                                                if_compute_recons_error=True, if_validate=False)

            for pred_type in prediction_method_list:
                results_dict_new.update({'method': 'SDL-conv-feat ({})'.format(str(pred_type))})
                results_dict_new.update({'beta': None})
                results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy ({})'.format(str(pred_type)))})
                results_dict_new.update({'F_score': results_dict_new.get('F_score ({})'.format(str(pred_type)))})
                results_dict_list.append(results_dict_new.copy())


    # LMF (SDL-filter)
    prediction_method_list = ['naive']
    for beta in beta_list:
        for j in range(len(xi_list)):
            xi = xi_list[j]
            for i in range(iter_avg):
                LMF_class_new = LMF(X=[X_train, Y_train],  # data, label
                                X_test=[X_test, Y_test],
                                #X_auxiliary = None,
                                n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                # ini_code = H_true,
                                xi=xi,  # weight on label reconstruction error
                                L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                                full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression

                results_dict_new = LMF_class_new.train_logistic(iter=iteration, subsample_size=None,
                                                        beta = beta,
                                                        search_radius_const=np.linalg.norm(X_train),
                                                        fine_tune_beta=True,
                                                        update_nuance_param=False,
                                                        prediction_method_list = prediction_method_list,
                                                        if_compute_recons_error=False, if_validate=False)

                for pred_type in prediction_method_list:
                    results_dict_new.update({'method': 'SDL-feat ({})'.format(str(pred_type))})
                    results_dict_new.update({'beta': beta})
                    results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy ({})'.format(str(pred_type)))})
                    results_dict_new.update({'F_score': results_dict_new.get('F_score ({})'.format(str(pred_type)))})
                    results_dict_list.append(results_dict_new.copy())


                if save_path is not None:
                    np.save(save_path, results_dict_list)


    # MF --> LR
    print('MF->LR')
    for i in range(iter_avg):
        W, H = ALS(X_train,
                    n_components = r, # number of columns in the dictionary matrix W
                    n_iter=iteration,
                    a0 = 0, # L1 regularizer for H
                    a1 = 0, # L1 regularizer for W
                    a12 = 0, # L2 regularizer for W
                    H_nonnegativity=True,
                    W_nonnegativity=True,
                    compute_recons_error=False,
                    subsample_ratio = 1)

        clf = LogisticRegression(random_state=0).fit((W.T @ X_train).T, Y_train[0,:])
        P_train = clf.predict_proba((W.T @ X_train).T)
        P_pred = clf.predict_proba((W.T @ X_test).T)
        results = compute_accuracy_metrics(Y_test[0], P_pred[:,1], train_data=[Y_train[0], P_train[:,1]], verbose=True)
        results.update({'method': 'MF-LR'})

        coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=0, transform_algorithm='lasso_lars', positive_code=True)
        H1 = coder.transform(X_test.T).T
        error_data = np.linalg.norm((X_test - W @ H1).reshape(-1, 1), ord=2)**2
        rel_error_data = error_data / np.linalg.norm(X_test.reshape(-1, 1), ord=2)**2
        results.update({'Relative_reconstruction_loss (test)': rel_error_data})
        results.update({'xi': None})
        results.update({'beta': None})
        results_dict_list.append(results.copy())


    # SNMF
    for beta in beta_list:
        for j in range(len(xi_list)):
            xi = xi_list[j]
            for i in range(iter_avg):
                SNMF_class_new = SNMF(X=[X_train, Y_train],  # data, label
                                X_test=[X_test, Y_test],
                                #X_auxiliary = None,
                                n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                # ini_loading=None,  # Initialization for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                # ini_code = H_true,
                                xi=xi,  # weight on label reconstruction error
                                L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                                full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression

                results_dict_new = SNMF_class_new.train_logistic(iter=iteration, subsample_size=None,
                                                        beta = beta,
                                                        search_radius_const=np.linalg.norm(X_train),
                                                        update_nuance_param=False,
                                                        if_compute_recons_error=False, if_validate=False)
                results_dict_new.update({'method': 'SDL-filt'})
                results_dict_new.update({'beta': beta})
                results_dict_list.append(results_dict_new.copy())
                # print('Beta_learned', results_dict.get('loading')[1])

    if save_path is not None:
        np.save(save_path, results_dict_list)

    return results_dict_list


    if save_path is not None:
        np.save(save_path, results_dict_list)

    return results_dict_list




def get_avg_stats_old(input_list, metric = "Accuracy"):
    method_xi_list = []
    for i in np.arange(len(input_list)):
        method = input_list[i].get("method")
        xi = input_list[i].get("xi")
        if [method, xi] not in method_xi_list:
            method_xi_list.append([method, xi])

    results_list = []
    method_xi_list_new = []
    for method_xi in method_xi_list:
        method, xi = method_xi[0], method_xi[1]
        avg_results = {}
        avg_results.update({"method":method})
        avg_results.update({"xi":xi})
        avg_acc_list = []
        avg_rec_list = []
        for i in np.arange(len(input_list)):
            if (method == input_list[i].get("method")) and (xi == input_list[i].get("xi")):
                avg_acc_list.append(input_list[i].get(metric))
                avg_rec_list.append(input_list[i].get("Relative_reconstruction_loss (test)"))
                avg_results.update({"avg_acc_list": avg_acc_list})
                avg_results.update({"avg_rec_list": avg_rec_list})
        if [method, xi] not in method_xi_list_new:
            results_list.append(avg_results)
            method_xi_list_new.append([method, xi])

    return sorted(results_list, key=lambda d: d['method'])


def get_avg_stats(input_list, metric = "Accuracy"):
    method_xi_list = []
    for i in np.arange(len(input_list)):
        method = input_list[i].get("method")

        xi = input_list[i].get("xi")
        beta = input_list[i].get("beta")
        method_beta = [method, str(beta)]
        # print('_'.join([method, beta]))
        if [method_beta, xi] not in method_xi_list:
            method_xi_list.append([method_beta, xi])
        #if method not in method_xi_list:
        #    method_xi_list.append([method, xi])
    print(method_xi_list)

    results_list = []
    method_xi_list_new = []
    for method_xi in method_xi_list:
        method_beta, xi = method_xi[0], method_xi[1]
        avg_results = {}
        avg_results.update({"method":method_beta})
        avg_results.update({"xi":xi})

        avg_acc_list = []
        avg_rec_list = []
        for i in np.arange(len(input_list)):
            if (method_beta == [input_list[i].get("method"), str(input_list[i].get("beta"))]) and (xi == input_list[i].get("xi")):
                avg_acc_list.append(input_list[i].get(metric))
                print('!!! method, ACCU', [method_beta, input_list[i].get(metric)])
                avg_rec_list.append(input_list[i].get("Relative_reconstruction_loss (test)"))
                avg_results.update({"avg_acc_list": avg_acc_list})
                avg_results.update({"avg_rec_list": avg_rec_list})
        if method_xi not in method_xi_list_new:
            results_list.append(avg_results)
            method_xi_list_new.append(method_xi)

    return sorted(results_list, key=lambda d: d['method'])

def plot_accuracy(results_dict_list, save_path, metric="Accuracy", beta_list_plot=[1], ylim=[0,1], title=None):
    beta_list_plot = [str(beta) for beta in beta_list_plot]
    print('beta_list_plot', beta_list_plot)
    ncols = 1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7,5])

    avg_results_list = get_avg_stats(results_dict_list, metric=metric)

    color_dict = {}
    color_dict.update({'LR':'gray'})
    color_dict.update({'MF-LR':'k'})
    color_dict.update({'SDL-filt':'b'})
    color_dict.update({'SDL-feat (naive)':'r'})
    color_dict.update({'SDL-feat (exhaustive)':'b'})
    color_dict.update({'SDL-conv-filt':'g'})
    color_dict.update({'SDL-conv-feat (naive)':'r'})
    color_dict.update({'SDL-conv-feat (exhaustive)':'g'})
    marker_dict = {}
    marker_dict.update({'LR':'+'})
    marker_dict.update({'MR-->LR':">"})
    marker_dict.update({'SDL-filt':'*'})
    marker_dict.update({'SDL-feat (naive)':'x'})
    marker_dict.update({'SDL-feat (exhaustive)':'^'})
    marker_dict.update({'SDL-conv-filt':''})
    marker_dict.update({'SDL-conv-feat (naive)':''})
    marker_dict.update({'SDL-conv-feat (exhaustive)':''})

    # Get list of hyperparameters
    method_list = []
    xi_list = []

    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        xi = result_dict.get('xi')
        method = [ str(key) for key in result_dict.get('method')]
        if (xi is not None) and (xi not in xi_list):
            xi_list.append(xi)
        if method not in method_list:
            method_list.append(method)

    print("method_list",method_list)

    xi_list = sorted(xi_list)
    for method_beta in method_list:
        method, beta = method_beta[0], method_beta[1]
        print('method_beta', method_beta)

        if (method in ["LR", "MF-LR"]) or (beta in beta_list_plot):

            color = color_dict.get(method)
            marker = marker_dict.get(method)

            accuracy_array = []
            for j in np.arange(len(xi_list)):
                xi = xi_list[j]
                for i in np.arange(len(avg_results_list)):
                    results_dict = avg_results_list[i]
                    xi0 = results_dict.get('xi')
                    if (method == results_dict.get('method')[0]) and (beta == str(results_dict.get('method')[1])) and ((xi0 == xi) or (xi0==None)):
                        accuracy_array.append(results_dict.get('avg_acc_list').copy())
                        break

            accuracy_array = np.asarray(accuracy_array).T
            accuracy_mean = np.sum(accuracy_array, axis=0) / accuracy_array.shape[0]  ### axis-0 : trials
            accuracy_std = np.std(accuracy_array, axis=0)

            if method == "LR":
                ax.axhline(y=accuracy_mean[0], color=color, linestyle='--', label="LR")
            else:
                linestyle = "-"
                if (method in ["LR", 'MF-LR']):
                    linestyle = "--"
                elif method in ["SDL-conv-filt", "SDL-conv-feat (naive)", "SDL-conv-feat (exhaustive)"]:
                    linestyle = "--"
                elif (beta != "None"):
                    print(method, beta)
                    #linestyle = "-"
                    s = method.split(" ")
                    if len(s) == 1:
                        method = s[0] #+ " (DR)"
                    else:
                        method = s[0] #+ " (DR) " + s[1]


                markers, caps, bars = ax.errorbar(xi_list, accuracy_mean, yerr=accuracy_std,
                                                       fmt=color, marker=marker, linestyle=linestyle, label=method, errorevery=5, markersize=10)
                ax.fill_between(xi_list, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, facecolor=color, alpha=0.1)


    #ax.title.set_text("[p, r, n, noise_std] = [%i, %i, %i, %.2f]" % (p,r,n, noise_std))
    ax.set_xlabel(r"$ \xi$", fontsize=12)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(ylim)
    ax.legend()
    if title is not None:
        plt.title(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(save_path)

def plot_pareto(results_dict_list, save_path,
                 metric="Accuracy", xlim=[0,1], ylim=[0,1],
                 beta_list_plot=[1],
                 title=None):
    "making pareto optimality plot.."

    ncols = 1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,5])

    avg_results_list = get_avg_stats(results_dict_list, metric=metric)

    color_dict = {}
    color_dict.update({'LR':'gray'})
    color_dict.update({'MF-LR':'k'})
    color_dict.update({'SDL-filt':'b'})
    color_dict.update({'SDL-feat (naive)':'r'})
    color_dict.update({'SDL-feat (exhaustive)':'b'})
    color_dict.update({'SDL-conv-filt':'g'})
    color_dict.update({'SDL-conv-feat (naive)':'r'})
    color_dict.update({'SDL-conv-feat (exhaustive)':'g'})
    marker_dict = {}
    marker_dict.update({'LR':'+'})
    marker_dict.update({'MR-->LR':">"})
    marker_dict.update({'SDL-filt':'*'})
    marker_dict.update({'SDL-feat (naive)':'x'})
    marker_dict.update({'SDL-feat (exhaustive)':'^'})
    marker_dict.update({'SDL-conv-filt':'o'})
    marker_dict.update({'SDL-conv-feat (naive)':'|'})
    marker_dict.update({'SDL-conv-feat (exhaustive)':'<'})

    xi_list = []
    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        xi = result_dict.get("xi")
        xi_list.append(xi)
    xi_list0 = list(set(xi_list))
    xi_list = [xi for xi in xi_list0 if xi is not None]
    xi_min = min(xi_list)
    print('xi_min', xi_min)

    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        method, beta = result_dict.get('method')[0], result_dict.get('method')[1]
        print(method, beta)
        xi = None
        if not ((method not in ["LR", "MF-LR"]) and (beta in beta_list_plot)):

            if method in ['SDL-filt', 'SDL-feat (naive)', 'SDL-feat (exhaustive)', 'SDL-conv-filt', 'SDL-conv-feat (naive)', 'SDL-conv-feat (exhaustive)']:
                xi = result_dict.get('xi')
            #print('xi', xi)

            rel_recons_error = np.mean(result_dict.get('avg_rec_list'))
            accuracy = np.mean(result_dict.get('avg_acc_list'))

            #print('rel_recons_error', rel_recons_error)
            #print('accuracy', accuracy)

            color = color_dict.get(method)
            marker = marker_dict.get(method)
            method0 = method
            if (beta != "None"):
                s = method.split(" ")

            print("method0", method0)
            print("rel_recons_error", rel_recons_error)
            print("accuracy", accuracy)

            if (xi is not None) and (xi>xi_min):
                ax.scatter(rel_recons_error, accuracy, s=100, c=color, alpha=1, marker=marker)
            else:
                ax.scatter(rel_recons_error, accuracy, s=100, c=color, alpha=1, label=method0, marker=marker)
            if (method in ['SDL-filt', 'SDL-feat (exhaustive)']) and (xi in [0, 0.001, 10]):
                x_len = xlim[1]-xlim[0]
                y_len = ylim[1]-ylim[0]
                ax.annotate(r" $\xi={}$".format(xi), (rel_recons_error-(0.07*x_len), accuracy+0.02*(y_len)), fontsize=9)

            results_prev = avg_results_list[i-1]
            if (xi is not None) and (xi>0) and (method == results_prev.get('method')[0]) and (beta == str(results_prev.get('method')[1]) ):
                #print(method, results_prev.get('method'), results_prev.get('xi'), xi)

                linestyle = "--"
                if (beta != "None"):
                    linestyle = "-"

                rel_recons_error_prev = np.mean(results_prev.get('avg_rec_list'))
                accuracy_prev = np.mean(results_prev.get('avg_acc_list'))
                line = [rel_recons_error_prev, accuracy_prev, rel_recons_error_prev, accuracy_prev]

                ax.plot([rel_recons_error_prev, rel_recons_error], [accuracy_prev, accuracy],
                        'k-', linestyle = linestyle, color = color)

            ax.set_xlabel('Reconstruction error', fontsize=12)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend()
    if title is not None:
        plt.title(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(save_path)


def main(data_type = "MNIST",
         n_components = 2,
         xi_list = [0, 0.1, 1, 5, 10],
         beta_list = [0.5, None],
         noise = 1,
         iteration = 200,
         iter_avg=2,
         plot_only=False):

    folder_name = "SDL_sim_J"
    file_name = folder_name + "_" + str(data_type)

    if plot_only:
        save_path = "Output_files/" + folder_name + "/results_" + file_name + ".npy"
        results_dict_list = np.load(save_path, allow_pickle=True)
        for metric in ["Accuracy", "F_score"]:
            plot_accuracy(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_accuracy_" + metric + "_" + file_name + ".pdf",
                          title=data_type)
            plot_pareto(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_pareto_" + metric + "_" + file_name + ".pdf",
                          title=data_type)
    else:
        if data_type == "fakejob":
            path1 = "Data/fake_job_postings_v9.csv"
            data1 = pd.read_csv(path1, delimiter=',')

            path2 = "Data/results_data_description2.csv"
            data2 = pd.read_csv(path2, delimiter=',')

            d1 = data1['fraud']
            Y = np.asarray(d1) # indicator of fraud postings
            Y = Y[np.newaxis,:]
            print('Y.shape', Y.shape)
            print('number of fraud postings:', np.sum(Y))
            print('ratio of fraud postings:', np.sum(Y)/Y.shape[1])

            d2 = data1.get(data1.keys()[1:73]) # covariates
            d3 = data1.get(data1.keys()[73:]) # word frequencies

            #X = d3.values
            X = data2.values
            X = X - np.min(X) # word frequency array
            X = X.T
            print('X.shape', X.shape) # words x docs

            X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2)
            X_train, X_test = X_train.T, X_test.T
            Y_train, Y_test = Y_train.T, Y_test.T
            print('X_train.shape', X_train.shape)
            print('Y_train.shape', Y_train.shape)
            print('X_test.shape', X_test.shape)
            print('Y_test.shape', Y_test.shape)
            print('number of fraud postings in Y_test:', np.sum(Y_test))


        if data_type == "pneumonia":
            print('fetching Pneumonia X-ray dataset ...')
            IMAGE_SIZE = [200, 200]  # low performance with size [50, 50]
            subsample_ratio = 10

            # Load training set for optimal parameter (only for Lasso or Ridge)
            train_path_normal = "Data/chest_xray/train/NORMAL/"
            train_path_pneumonia = "Data/chest_xray/train/PNEUMONIA/"

            X_train, Y_train, train_n0, train_n1 = process_path(train_path_normal, train_path_pneumonia, IMAGE_SIZE=IMAGE_SIZE,
                                                                vector_label=False, subsample_ratio=subsample_ratio)

            # dim(X) = p x n
            # dim(Y) = 1 x n


            # Load test set for evaluation
            test_path_normal = "Data/chest_xray/test/NORMAL/"
            test_path_pneumonia = "Data/chest_xray/test/PNEUMONIA/"

            X_test, Y_test, test_n0, test_n1 = process_path(test_path_normal, test_path_pneumonia, IMAGE_SIZE=IMAGE_SIZE,
                                                                vector_label=False, subsample_ratio=subsample_ratio)

        if data_type == "MNIST":
            print('fetching MNIST ...')
            '''
            # Load data from https://www.openml.org/d/554
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # X = X.values  ### Uncomment this line if you are having type errors in plotting. It is loading as a pandas dataframe, but our indexing is for numpy array.
            X = X / 255.

            print('X.shape', X.shape)
            print('y.shape', y.shape)


            Each row of X is a vectroization of an image of 28 x 28 = 784 pixels.
            The corresponding row of y holds the true class label from {0,1, .. , 9}.


            n=500
            noise_std = 0.5
            r0 = 10
            X_train, X_test, Y_train, Y_test, W_true, W_true_Y, H_true, H_train, H_test, Beta_true = sim_data_gen_MNIST(r = [r0, r0],
                                                                                                                        n = n,
                                                                                                                        digits_X = ['2', '5'],
                                                                                                                        digits_Y = ['4', '7'],
                                                                                                                        noise_std = noise_std,
                                                                                                                        random_seed = 1)

        '''

        X_train, X_test, Y_train, Y_test, W_true, W_true_Y, H_true, H_train, H_test, Beta_true = sim_data_gen_MNIST(r = [int(n_components/2), int(n_components/2)],
                                                                                                                    n = 1000,
                                                                                                                    digits_X = ['2', '5'],
                                                                                                                    digits_Y = ['4', '7'],
                                                                                                                    noise_std = noise,
                                                                                                                    random_seed = 1)
        data = [X_train, X_test, Y_train, Y_test]

        results_dict_list = run_methods(data, n_components=n_components,
                                        #xi_list = [0, 0.001, 1,  3, 5, 10],
                                        xi_list = xi_list,
                                        beta_list = beta_list,
                                        iteration = iteration,
                                        iter_avg = iter_avg,
                                        save_path = "Output_files/" + folder_name + "/results_noise_" + str(noise*10) + "_" + file_name + ".npy")

        #print('results_dict_list', results_dict_list)

        for metric in ["Accuracy", "F_score"]:
            plot_accuracy(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_noise_" + str(noise*10) + "_accuracy_" + metric + "_" + file_name + ".pdf",
                          title=data_type)

            plot_pareto(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_noise_" + str(noise*10) + "_pareto_" + metric + "_" + file_name + ".pdf",
                          title=data_type)


if __name__ == '__main__':
    main(data_type="MNIST",
         n_components=2,
         noise = 1,
         xi_list = [0, 0.1, 1, 5, 10],
         beta_list = [0.5],
         iteration = 200,
         iter_avg=5,
         plot_only=False)
