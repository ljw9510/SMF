import re
import os
import numpy as np
import pandas as pd

from tqdm import trange
import matplotlib.pyplot as plt
from src.SDL_SVP import SDL_SVP
from src.SDL_BCD import SDL_BCD

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d

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

from src.plotting import plot_accuracy, plot_pareto, plot_benchmark_errors, get_avg_stats
#from pneumonia_dataprocess import process_path

#import seaborn as sns
#sns.set_theme()

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
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
        mythre_test = thresholds[np.argmax(tpr - fpr)]
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
    results_dict.update({'Fall_out': fall_out})
    results_dict.update({'Miss_rate': miss_rate})
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

def run_methods(data,
                n_components,
                data_aux = None,
                xi_list = [0.001, 0.01, 0.1,  1, 5, 10],
                beta_list = [1, None],
                iteration=200, iter_avg=2,
                methods_list = ["LR", "MF-LR", "SDL-filt", "SDL-feat", "SDL-conv-filt", "SDL-conv-feat"],
                save_path = None):
    # data  = [X_train, X_test, Y_train, Y_test]
    ## Cross validation plot --- MF + LR, SNMF, LR
    print("methods_list", methods_list)

    X_train, X_test, Y_train, Y_test = data
    if data_aux is not None:
        covariate_train, covariate_test = data_aux  # auxiliary covariate data scaling
        covariate_train = covariate_train/10
        covariate_test = covariate_test/10

    r = n_components
    p = X_train.shape[0]
    results_dict_list = []
    full_result_list = []


    # LR
    if "LR" in methods_list:

        if data_aux is not None:
            X0_train = np.vstack([X_train, covariate_train])
            X0_test = np.vstack([X_test, covariate_test])
            print('X0_train.T.shape', X0_train.T.shape)
            clf = LogisticRegression(random_state=0).fit(X0_train.T, Y_train[0,:])
            P_train = clf.predict_proba(X0_train.T)
            P_pred = clf.predict_proba(X0_test.T)
        else:
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


    # MF --> LR
    if "MF-LR" in methods_list:
        for i in range(iter_avg):
            print('MF-LR')
            print("X_train.shape", X_train.shape)

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

            if data_aux is not None:
                X0_train = np.vstack([W.T @ X_train, covariate_train])
                X0_test = np.vstack([W.T @ X_test, covariate_test])
                print('X0_train.T.shape', X0_train.T.shape)
                clf = LogisticRegression(random_state=0).fit((X0_train).T, Y_train[0,:])
                P_train = clf.predict_proba((X0_train).T)
                P_pred = clf.predict_proba((X0_test).T)
            else:
                print('X_train.T.shape', X_train.T.shape)
                print('Y_train[0,:].shape', Y_train[0,:].shape)
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
            results.update({'W': W})
            results.update({'beta_regression': clf.coef_})
            results_dict_list.append(results.copy())


    # (SDL-filter)

    X_auxiliary = None
    X_test_aux = None
    if data_aux is not None:
        X_auxiliary = covariate_train
        X_test_aux = covariate_test

        print("!!! X_auxiliary.shape", X_auxiliary.shape)
        print("!!! X_test_aux.shape", X_test_aux.shape)
    else:
        print("X_auxiliary is None")

    if "SDL-filt" in methods_list:

        for beta in beta_list:
            for j in range(len(xi_list)):
                xi = xi_list[j]
                for i in range(iter_avg):
                    print("SDL-filt..")
                    print("X_train.shape", X_train.shape)
                    print("X_auxiliary.shape", X_auxiliary.shape)
                    print("X_test_aux.shape", X_test_aux.shape)

                    SDL_BCD_class = SDL_BCD(X=[X_train, Y_train],  # data, label
                                    X_test=[X_test, Y_test],
                                    X_auxiliary = X_auxiliary,
                                    X_test_aux = X_test_aux,
                                    n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                    # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                    # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                    # ini_code = H_true,
                                    xi=xi,  # weight on label reconstruction error
                                    L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                    L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                    nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                                    full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression


                    results_dict_new = SDL_BCD_class.fit(iter=iteration, subsample_size=None,
                                                            beta = beta,
                                                            option = "filter",
                                                            search_radius_const=iteration*np.linalg.norm(X_train),
                                                            update_nuance_param=False,
                                                            if_compute_recons_error=True, if_validate=False)

                    results_dict_new.update({'method': 'SDL-filt'})
                    results_dict_new.update({'beta': beta})
                    results_dict_new.update({'time_error': results_dict_new.get('time_error')})
                    results_dict_list.append(results_dict_new.copy())
                    # print('Beta_learned', results_dict.get('loading')[1])



    # (SDL-feature)
    if "SDL-feat" in methods_list:
        prediction_method_list = ['naive']
        for beta in beta_list:
            for j in range(len(xi_list)):
                xi = xi_list[j]
                for i in range(iter_avg):
                    print("SDL-feat..")
                    print("X_train.shape", X_train.shape)
                    SDL_BCD_class = SDL_BCD(X=[X_train, Y_train],  # data, label
                                    X_test=[X_test, Y_test],
                                    X_auxiliary = X_auxiliary,
                                    X_test_aux = X_test_aux,
                                    n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                    # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                    # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                    # ini_code = H_true,
                                    xi=xi,  # weight on label reconstruction error
                                    L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                    L2_reg = [0,0,0], # L2 regularizer for code H, dictionary W[0], reg param W[1]
                                    nonnegativity=[True,True,False], # nonnegativity constraints on code H, dictionary W[0], reg params W[1]
                                    full_dim=False) # if true, dictionary is Id with full dimension --> Pure regression

                    results_dict_new = SDL_BCD_class.fit(iter=iteration, subsample_size=None,
                                                            beta = beta,
                                                            option = "feature",
                                                            search_radius_const=iteration*np.linalg.norm(X_train),
                                                            update_nuance_param=False,
                                                            #prediction_method_list = prediction_method_list,
                                                            if_compute_recons_error=True, if_validate=False)

                    for pred_type in prediction_method_list:
                        #results_dict_new.update({'method': 'SDL-feat ({})'.format(str(pred_type))})
                        results_dict_new.update({'method': 'SDL-feat'})
                        results_dict_new.update({'beta': beta})
                        results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy')})
                        results_dict_new.update({'F_score': results_dict_new.get('F_score')})
                        #results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy ({})'.format(str(pred_type)))})
                        #results_dict_new.update({'F_score': results_dict_new.get('F_score ({})'.format(str(pred_type)))})
                        results_dict_new.update({'time_error': results_dict_new.get('time_error')})
                        results_dict_list.append(results_dict_new.copy())


                    if save_path is not None:
                        np.save(save_path, results_dict_list)


    # SDL_SVP_filter
    if "SDL-conv-filt" in methods_list:
        data_scale=10
        if data_aux is not None:
            X_auxiliary /= data_scale
            X_test_aux /= data_scale

        for j in range(len(xi_list)):
            xi = xi_list[j]
            list_full_timed_errors = []
            for i in range(iter_avg):
                print("SDL-conv-filt..")
                SDL_SVP_class = SDL_SVP(X=[X_train/data_scale, Y_train],  # data, label
                                        X_test=[X_test/data_scale, Y_test],
                                        X_auxiliary = X_auxiliary,
                                        X_test_aux = X_test_aux,
                                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                        # ini_code = H_true,
                                        xi=xi,  # weight on label reconstruction error
                                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                        L2_reg = [0,0,0]) # L2 regularizer for code H, dictionary W[0], reg param W[1]


                results_dict_new = SDL_SVP_class.fit(iter=iteration, subsample_size=None,
                                                        beta = 0,
                                                        nu = 2,
                                                        search_radius_const=0.01,
                                                        update_nuance_param=False,
                                                        SDL_option = 'filter',
                                                        prediction_method_list = ['filter'],
                                                        fine_tune_beta = False,
                                                        if_compute_recons_error=True, if_validate=False)

                results_dict_new.update({'method': 'SDL-conv-filt'})
                results_dict_new.update({'beta': None})
                results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy (filter)')})
                results_dict_new.update({'F_score': results_dict_new.get('F_score (filter)')})
                results_dict_new.update({'time_error': results_dict_new.get('time_error')})
                results_dict_list.append(results_dict_new.copy())


    # SDL_SVP_feature
    if "SDL-conv-feat" in methods_list:
        data_scale=10
        if data_aux is not None:
            X_auxiliary /= data_scale
            X_test_aux /= data_scale

        prediction_method_list = ['naive']
        for j in range(len(xi_list)):
            xi = xi_list[j]
            for i in range(iter_avg):
                print("SDL-conv-feat..")
                data_scale=500
                SDL_SVP_class = SDL_SVP(X=[X_train/data_scale, Y_train],  # data, label
                                        X_test=[X_test/data_scale, Y_test],
                                        X_auxiliary = X_auxiliary,
                                        X_test_aux = X_test_aux,
                                        n_components=r,  # =: r = number of columns in dictionary matrices W, W'
                                        # ini_loading=None,  # Initializatio for [W,W'], W1.shape = [d1, r], W2.shape = [d2, r]
                                        # ini_loading=[W_true, np.hstack((np.array([[0]]), Beta_true))],
                                        # ini_code = H_true,
                                        xi=xi,  # weight on label reconstruction error
                                        L1_reg = [0,0,0], # L1 regularizer for code H, dictionary W[0], reg param W[1]
                                        L2_reg = [0,0,0]) # L2 regularizer for code H, dictionary W[0], reg param W[1]


                results_dict_new = SDL_SVP_class.fit(iter=iteration, subsample_size=None,
                                                    beta = 0,
                                                    nu = 2,
                                                    search_radius_const=0.01,
                                                    update_nuance_param=False,
                                                    SDL_option = 'feature',
                                                    #prediction_method_list = ['naive', 'exhaustive'],
                                                    prediction_method_list = prediction_method_list,
                                                    if_compute_recons_error=True, if_validate=False)

                for pred_type in prediction_method_list:
                    results_dict_new.update({'method': 'SDL-conv-feat ({})'.format(str(pred_type))})
                    results_dict_new.update({'beta': None})
                    results_dict_new.update({'Accuracy': results_dict_new.get('Accuracy ({})'.format(str(pred_type)))})
                    results_dict_new.update({'F_score': results_dict_new.get('F_score ({})'.format(str(pred_type)))})
                    results_dict_new.update({'time_error': results_dict_new.get('time_error')})
                    results_dict_list.append(results_dict_new.copy())

    if save_path is not None:
        np.save(save_path, results_dict_list)

    return results_dict_list


    if save_path is not None:
        np.save(save_path, results_dict_list)

    return results_dict_list

def main(data_type = "MNIST",
         use_data_aux = True,
         n_components = 20,
         xi_list = [0, 0.001, 1, 5, 10],
         beta_list = [0.5, None],
         iteration = 200,
         iter_avg=1,
         plot_only=False,
         methods_list = ["LR", "MF-LR", "SDL-filt", "SDL-feat", "SDL-conv-filt", "SDL-conv-feat"],
         folder_name = "SDL_sim3",
         error_plot_method_list = ["SDL-conv-feat", "SDL-conv-filt"]):

    file_name = folder_name + "_" + str(data_type)

    if plot_only:
        save_path = "Output_files/" + folder_name + "/results_" + file_name + ".npy"
        results_dict_list = np.load(save_path, allow_pickle=True)
        #avg_results_list = get_avg_stats(results_dict_list, metric="time_error")

        print(" !!! results_dict_list", len(results_dict_list))


        for metric in ["Accuracy", "F_score"]:
            plot_accuracy(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_accuracy_" + metric + "_" + file_name + ".pdf",
                          title=None)

            plot_pareto(results_dict_list, metric=metric, xlim=[0, 1.02], ylim=[0,0.9], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_pareto_" + metric + "_" + file_name + ".pdf",
                          title=None)

        avg_results_list = get_avg_stats(results_dict_list, metric="time_error")
        plot_benchmark_errors(avg_results_list,
                              save_path = "Output_files/" + folder_name + "/plot_error_" + file_name + ".pdf",
                              method_list = error_plot_method_list,
                              xi_list_custom = [0.01, 0.1, 1, 5],
                              fig_size=[6,6])


    else:
        #covariate_train = None
        #covariate_test = None

        if data_type == "fakejob":

            data = pd.read_csv('final_fake_job_postings.csv', delimiter=',')
            Y = data['fraudulent']
            print(sum(Y)/len(Y)) # prop : 5%

            covariate = data.get(data.keys()[1:73]) # covariates
            text = data.get(data.keys()[73:]) # text

            total_variable = list(data.keys()[1:])
            #X = np.hstack((covariate, text))

            Y = np.asarray(Y) # indicator of fraud postings
            print('Y.shape', Y.shape)

            text = text.values
            text = text - np.min(text) # word frequency array
            print('text.shape', text.shape) # words x docs

            covariate = covariate.values
            covariate = covariate - np.min(covariate)
            print('covariate.shape', covariate.shape)

            np.random.seed(1)
            Y_train, Y_test, text_train, text_test, covariate_train, covariate_test = train_test_split(Y, text, covariate,
                                                                                                       test_size = 0.2)
            print('ratio of fraud postings in train set:', np.sum(Y_train)/Y_train.shape)
            print('ratio of fraud postings in test set:', np.sum(Y_test)/Y_test.shape)

            text_train, text_test = text_train.T, text_test.T
            covariate_train, covariate_test = covariate_train.T, covariate_test.T
            Y_train, Y_test = Y_train[np.newaxis,:], Y_test[np.newaxis,:]
            X_train, X_test = text_train, text_test

            print(Y_train.shape)
            print(X_train.shape)
            print(text_train.shape)
            print(covariate_train.shape)

            #covariate_train = None ############
            #covariate_test = None #############

        data = [X_train, X_test, Y_train, Y_test]
        data_aux = None
        if use_data_aux:
            data_aux = [covariate_train, covariate_test]


        results_dict_list = run_methods(data,
                                        data_aux = data_aux,
                                        n_components=n_components,
                                        #xi_list = [0, 0.001, 1,  3, 5, 10],
                                        xi_list = xi_list,
                                        beta_list = beta_list,
                                        iteration = iteration,
                                        iter_avg = iter_avg,
                                        methods_list = methods_list,
                                        save_path = "Output_files/" + folder_name + "/results_" + file_name + ".npy")

        #print('results_dict_list', results_dict_list)

        for metric in ["Accuracy", "F_score"]:
            plot_accuracy(results_dict_list, metric=metric, ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_accuracy_" + metric + "_" + file_name + ".pdf",
                          title=data_type)

            plot_pareto(results_dict_list, metric=metric, xlim=[0,1.1], ylim=[0,1], beta_list_plot=[None]+beta_list,
                          save_path = "Output_files/" + folder_name + "/plot_pareto_" + metric + "_" + file_name + ".pdf",
                          title=data_type)

        avg_results_list = get_avg_stats(results_dict_list, metric="time_error")
        plot_benchmark_errors(avg_results_list,
                              save_path = "Output_files/" + folder_name + "/plot_error_" + metric + "_" + file_name + ".pdf",
                              #method_list = error_plot_method_list,
                              fig_size=[7,8])


if __name__ == '__main__':

    main(data_type="fakejob",
         use_data_aux = True,
         n_components=20,
         #xi_list = [0.001],
         xi_list = [0.001, 0.01, 0.1, 1, 5, 10],
         beta_list = [1],
         iteration = 100,
         iter_avg=4,
         plot_only=False,
         methods_list = ["LR", "MF-LR", "SDL-filt", "SDL-feat", "SDL-conv-filt", "SDL-conv-feat"],
         #methods_list = ["LR", "MF-LR"],
         #methods_list = ["SDL-feat"],
         #methods_list = ["SDL-filt"],
         folder_name = "SDL_fakejob1",
         error_plot_method_list = ["SDL-feat"])
