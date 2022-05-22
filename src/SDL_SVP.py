# (setq python-shell-interpreter "./venv/bin/python")


# import tensorflow as tf
import numpy as np
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time
from tqdm import trange
from sklearn.metrics import roc_curve
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.sparse as sp
from sklearn.decomposition import SparseCoder
from sklearn.linear_model import LogisticRegression
from scipy.linalg import block_diag
from sklearn.decomposition import TruncatedSVD





DEBUG = False


class SDL_SVP():
    # Supervised Dictionary Learning in convex formulation

    def __init__(self,
                 X,  # [X = [X,Y] : data (d1 x n), label (d2 x n)]
                 X_auxiliary=None, # auxiliary data (d3 x n) that is subject to LR but not to NMF
                 X_test=None, # [X_test = [X_test, Y_test]] test set
                 X_test_aux=None, # aux test data (d3 x n)
                 n_components=100,  # =: r = number of columns in dictionary matrices W, W'
                 iterations=500,
                 ini_loading=None,  # Initialization for [W1, W2, W3] = [(dict), (reg. coeff), (reg. coeff for aux var.)]
                 #W1.shape = [d1, r], W2.shape = [d2, r], W3.shape = [d3, r]
                 ini_code=None,
                 xi = None, # weight for dim reduction vs. prediction trade-off
                 L1_reg=[0,0,0], # L1 regularizer for code H, dictioanry W[0], and regression params W[1]
                 L2_reg=[0,0,0], # L2 regularizer for code H, dictioanry W[0], and regression params W[1]
                 full_dim=False): # if true, dictionary matrix W[0] is Id with size d1 x d1 -- no dimension reduction

        self.X = X
        self.X_auxiliary = X_auxiliary
        self.d3 = 0 # auxiliary data dim
        if X_auxiliary is not None:
            self.d3 = X_auxiliary.shape[0]

        self.X_test = X_test
        self.X_test_aux = X_test_aux
        self.n_components = n_components
        self.iterations = iterations
        self.ini_code = ini_code
        if ini_code is None:
            self.ini_code = np.random.rand(n_components, X[0].shape[1])

        self.loading = ini_loading
        if ini_loading is None:
            d1, n = X[0].shape
            d2, n = X[1].shape
            r = n_components
            self.loading = [np.random.rand(X[0].shape[0], r), 1-2*np.random.rand(X[1].shape[0], r + 1 + self.d3)]  # additional first column for constant terms in Logistic Regression
            # add additional d3 columns of regression coefficients for the auxiliary variables
        print('initial loading beta', self.loading[1])

        self.xi = xi
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.code = np.zeros(shape=(n_components, X[0].shape[1]))
        self.full_dim = full_dim
        self.result_dict = {}
        self.result_dict.update({'xi' : self.xi})
        self.result_dict.update({'L1_reg' : self.L1_reg})
        self.result_dict.update({'L2_reg' : self.L2_reg})
        self.result_dict.update({'n_components' : self.n_components})


    def rank_r_projection(self, X, rank):
        svd = TruncatedSVD(n_components=rank, n_iter=7, random_state=42)
        X_reduced = svd.fit_transform(X) #
        u = X_reduced.dot(np.linalg.inv(np.diag(svd.singular_values_)))
        s = svd.singular_values_
        vh = svd.components_
        r = rank
        u0 = u[:,:r]
        s0 = s[:r]
        v0 = vh[:r,:]
        recons = u0 @ np.diag(s0) @ v0
        return u0, s0, v0, recons

    def unfactored2factored(self, A, B, Beta1, rank, option='filter'): # or 'feature')
        if option == 'filter':
            C = np.hstack((A,B))
            u0, s0, v0, recons = self.rank_r_projection(C, rank=rank)
            s = np.diag(s0)
            W0 = u0 @ np.sqrt(s)
            W_norm = np.linalg.norm(W0)
            W0 /= W_norm
            D = W_norm * (np.sqrt(s) @ v0)
            Beta = D[:,:1].T
            W = [W0, np.hstack((Beta1[:,:1], Beta, Beta1[:,1:]))]
            H = D[:,1:]

        elif option == 'feature':
            C = np.vstack((A,B))
            u0, s0, v0, recons = self.rank_r_projection(C, rank=rank)
            s = np.diag(s0)
            H = np.sqrt(s) @ v0
            D = u0 @ np.sqrt(s)
            #H_norm = np.linalg.norm(H)
            #D *= H_norm
            W_norm = np.linalg.norm(D[1:,:])
            W0 = D[1:,:]/W_norm
            Beta = D[:1,:]/W_norm
            H *= W_norm
            W = [W0, np.hstack((Beta1[:,:1], Beta, Beta1[:,1:]))]

        return W, H


    def step_SVP(self, A, B, Beta1, tau=1, nu=0.1, option='filter'): # or 'feature'):
        # A = W[0] @ W[1][1:1+self.n_components].T    (p x 1)
        # B = W @ H    (p x n)
        # Beta1 = regression coefficients for auxiliary variables (including the bias term)  (1 x q) = (1 x (1+self.d3))

        X = self.X
        r = self.n_components
        n = X[0].shape[1]

        grad_B = 2 * self.xi * (B - X[0])
        Z = np.ones(shape=[1,X[0].shape[1]]) # auxiliary covariates
        if self.d3>0:
            Z = np.vstack((Z, self.X_auxiliary))

        if option == 'filter':
            D = A.T @ X[0] + Beta1 @ Z
            P = 1 / (1 + np.exp(-D))
            grad_A = X[0] @ (P - X[1]).T + nu * A
            grad_Beta1 = Z @ (P - X[1]).T + nu * Beta1
            if self.d3>0:
                grad_Beta1 = (P - X[1]) @ Z.T + nu * Beta1

        elif option == 'feature':
            D = A + Beta1 @ Z
            P = 1 / (1 + np.exp(-D))
            grad_A = (P - X[1]) + nu * A
            grad_Beta1 = Z @ (P - X[1]).T + nu * Beta1
            if self.d3>0:
                grad_Beta1 = (P - X[1]) @ Z.T + nu * Beta1

        # gradient descent step
        A -= tau * grad_A
        B -= tau * grad_B
        Beta1 -= tau * grad_Beta1

        # singular value projection on rank-r matrices
        if option == 'filter':
            C = np.hstack((A, B))
            u0, s0, v0, recons = self.rank_r_projection(C, rank=self.n_components)
            A_new = recons[:, :A.shape[1]]
            B_new = recons[:, A.shape[1]:]
            ### TODO: Maybe add column normalization step for W

        elif option == 'feature':
            C = np.vstack((A, B))
            u0, s0, v0, recons = self.rank_r_projection(C, rank=self.n_components)
            A_new = recons[:A.shape[0], :]
            B_new = recons[A.shape[0]:, :]

        return A_new, B_new, Beta1

    def step_feature(self, A, B, Beta1, tau=1, nu=0.1):
        # A = W[1][1:1+self.n_components] @ H    (1 x n)
        # B = W @ H    (p x n)
        # Beta1 = regression coefficients for auxiliary variables (including the bias term)  (1 x q) = (1 x (1+self.d3))

        X = self.X
        r = self.n_components
        n = X[0].shape[1]

        grad_B = 2 * self.xi * (B - X[0])
        Z = np.ones(shape=[1,X[0].shape[1]]) # auxiliary covariates
        if self.d3>0: #####################################
            Z = np.vstack((Z, self.X_auxiliary)) #########################

        D = A + Beta1 @ Z

        # P = probability matrix, same shape as X1
        P = 1 / (1 + np.exp(-D))
        grad_A = (P - X[1]) + nu * A
        grad_Beta1 = Z @ (P - X[1]).T + nu * Beta1
        if self.d3>0: ##########################################################
            grad_Beta1 = (P - X[1]) @ Z.T + nu * Beta1 #########################

        # gradient descent step
        A -= tau * grad_A
        B -= tau * grad_B
        Beta1 -= 10*tau * grad_Beta1

        # singular value projection on rank-r matrices
        C = np.vstack((A, B))
        u0, s0, v0, recons = self.rank_r_projection(C, rank=self.n_components)
        A_new = recons[:A.shape[0], :]
        B_new = recons[A.shape[0]:, :]

        return A_new, B_new, Beta1

    def fit(self,
            iter=100,
            beta=0.1,
            nu = 0.1, # L2 regularizer for regression coefficients
            dict_update_freq=1,
            subsample_size=None,
            subsample_ratio_code=None,
            search_radius_const=1000,
            if_compute_recons_error=False,
            update_nuance_param=False,
            if_validate=False,
            fine_tune_beta = False,
            SDL_option = 'filter',
            prediction_method_list=['filter']): # or 'feature'
        '''
        Given input X = [data, label] and initial loading dictionary W_ini, find W = [dict, beta] and code H
        by projected gradient descent in an unfactored formulation
        '''
        X = self.X
        r = self.n_components
        n = X[0].shape[1]

        #H = np.random.rand(r, n)
        #W = [np.random.rand(X[0].shape[0], r), np.random.rand(X[1].shape[0], r + 1 + self.d3)]  # additional first column for constant terms in Logistic Regression
        H = self.ini_code
        W = self.loading

        time_error = np.zeros(shape=[0, 3])
        elapsed_time = 0

        self.result_dict.update({'iter': iter})
        self.result_dict.update({'n_components': self.n_components})
        self.result_dict.update({'dict_update_freq' : dict_update_freq})

        # set up unfactored variable
        if SDL_option == 'filter':
            A = W[0] @ W[1][:,1:1+r].T
            B = W[0] @ H
        elif SDL_option == 'feature':
            A = W[1][:,1:1+r] @ H
            B = W[0] @ H


        Beta1 = np.zeros(shape=[X[1].shape[0], 1+ self.d3])
        Beta1[:,0] = W[1][:,0]
        Beta1[:,1:] = W[1][:,r+1:]

        for step in trange(int(iter)):
            start = time.time()

            # search_radius = search_radius_const * (float(step + 10)) ** (-beta) / np.log(float(step + 2))
            search_radius = search_radius_const * (float(step + 10)) ** (-beta)
            #print('search_radius', search_radius)
            # search_radius = 0.0001
            A, B, Beta1 = self.step_SVP(A, B, Beta1, tau=search_radius, nu=0, option=SDL_option)

            #print('Beta1', Beta1)
            end = time.time()
            elapsed_time += end - start

            if (step % 10) == 0:
                if if_compute_recons_error:
                    W, H = self.unfactored2factored(A, B, Beta1, rank=self.n_components, option=SDL_option)
                    #W /= np.linalg.norm(W[0])
                    #H *= np.linalg.norm(W[0])
                    #print('Beta', W[1])
                    # print the error every 50 iterations

                    error_data = np.linalg.norm((X[0] - W[0] @ H).reshape(-1, 1), ord=2)**2
                    rel_error_data = error_data / np.linalg.norm(X[0].reshape(-1, 1), ord=2)**2

                    print('*** rel_error_data train', rel_error_data)

                    if SDL_option == 'filter':
                        X0_comp = W[0].T @ X[0]
                        X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))
                        if self.d3>0:
                            X0_ext = np.vstack((X0_ext, self.X_auxiliary))


                    elif SDL_option == 'feature':

                        X0_ext = np.vstack((np.ones(X[1].shape[1]), H))
                        if self.d3>0:
                            H_ext = np.vstack((np.ones(X[1].shape[1]), H)) ###############
                            X0_ext = np.vstack((H_ext, self.X_auxiliary)) ###############
                        """
                        X0_comp = W[0].T @ X[0]
                        X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))
                        if self.d3>0:
                            X0_ext = np.vstack((X0_ext, self.X_auxiliary))
                        """

                    ### Additional fitting of logistic regression
                    #if fine_tune_beta:
                    #    print('beta_old', W[1][0,:])s
                    #    clf = LogisticRegression(random_state=0).fit(X0_comp.T, self.X[1][0,:])
                    #    W[1][0,1:] = clf.coef_[0]
                    #    W[1][0,0] = clf.intercept_[0]
                    #    print('beta_new', W[1][0,:])

                    P_pred = np.matmul(W[1], X0_ext)
                    #P_pred -= np.min(P_pred)
                    P_pred = 1 / (np.exp(-P_pred) + 1)
                    # print('Y - P_pred', np.linalg.norm(X[1] - P_pred))
                    # print('!!! error norm', np.linalg.norm(X[1][0, :]-P_pred[0,:])/X[1].shape[1])
                    fpr, tpr, thresholds = metrics.roc_curve(X[1][0, :], P_pred[0,:], pos_label=None)
                    mythre = thresholds[np.argmax(tpr - fpr)]
                    myauc = metrics.auc(fpr, tpr)
                    self.result_dict.update({'Training_threshold':mythre})
                    self.result_dict.update({'Training_AUC':myauc})
                    print('--- Training --- [threshold, AUC] = ', [np.round(mythre,3), np.round(myauc,3)])

                    error_label = np.sum(np.log(1+np.exp(W[1] @ X0_ext))) - X[1] @ (W[1] @ X0_ext).T
                    error_label = error_label[0][0]

                    time_error = np.append(time_error, np.array([[elapsed_time, error_data, error_label]]), axis=0)
                    print('--- Iteration %i: Training loss --- [Data, Label, Total] = [%f.3, %f.3, %f.3]' % (step, error_data, error_label, self.xi * error_data+error_label))

                    self.result_dict.update({'Relative_reconstruction_loss (training)': rel_error_data})
                    self.result_dict.update({'Classification_loss (training)': error_label})
                    self.result_dict.update({'loading': W})
                    self.result_dict.update({'code': H})
                    self.result_dict.update({'time_error': time_error.T})
                    self.loading = W
                    self.code = H
                    print('error_time', np.asarray(time_error).shape)

                if if_validate and (step>1):
                    self.validation(result_dict = self.result_dict, verbose=True,
                                    SDL_option=SDL_option,
                                    prediction_method_list=prediction_method_list)
                    threshold = self.result_dict.get('Training_threshold')
                    ACC_list = [self.result_dict.get('Accuracy ({})'.format(pred_type)) for pred_type in prediction_method_list]
                    ACC = max(ACC_list)
                    print('Training_threshold', np.round(threshold,3))
                    print('ACC', ACC)
                    if ACC>0.99:
                        # terminate the training as soon as AUC>0.9 in order to avoid overfitting
                        print('!!! --- Validation (Stopped) --- [threshold, ACC] = ', [np.round(threshold,3), np.round(ACC,3)])
                        break



        W, H = self.unfactored2factored(A, B, Beta1, rank=self.n_components, option=SDL_option)
        self.result_dict.update({'loading': W})
        self.result_dict.update({'code': H})
        self.loading = W
        self.code = H

        self.validation(result_dict = self.result_dict,
                        verbose=True,
                        SDL_option=SDL_option,
                        prediction_method_list=prediction_method_list)
        #threshold = self.result_dict.get('Opt_threshold')
        #AUC = self.result_dict.get('AUC')
        #print('!!! FINAL [threshold, AUC] = ', [np.round(threshold,3), np.round(AUC,3)])

        return self.result_dict

    def sparse_code(self, X, W, sparsity=0, nonnegativity=False):
        # Same function as OMF

        '''
        Given data matrix X and dictionary matrix W, find
        code matrix H such that W*H approximates X

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        '''



        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H
        coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                            transform_alpha=sparsity, transform_algorithm='lasso_lars', positive_code=nonnegativity)
        # alpha = L1 regularization parameter.
        H = coder.transform(X.T)

        # transpose H before returning to undo the preceding transpose on X
        # print('!!! sparse_code: Start')
        return H.T

    def update_code_joint_logistic(self, X, W, H0, r,
                                   a1=0, a2=0, sub_iter=2,
                                   stopping_diff=0.1, nonnegativity=True,
                                   xi = 0,
                                   subsample_size=None):
        '''
        X = [X0, X1]
        W = [W0, W1+W2]
        Find \hat{H} = argmin_H ( xi * || X0 - W0 H||^2 + alpha|H| + Logistic_Loss(X1, [W1|W2], H)) within radius r from H0
        Use row-wise projected gradient descent
        '''

        if H0 is None:
            H0 = np.random.rand(W[0].shape[1], X[0].shape[1])
            # print('!!! H0.shape', H0.shape)

        if not self.full_dim:
            A = W[0].T @ W[0]
            B = W[0].T @ X[0]

        H1 = H0.copy()
        i = 0
        dist = 1
        idx = np.arange(X[0].shape[1])
        while (i < sub_iter) and (dist > stopping_diff):
            H1_old = H1.copy()
            for k in np.arange(H1.shape[0]):
                if subsample_size is not None:
                    idx = np.random.randint(X[0].shape[1], size=subsample_size)

                H1_ext = np.vstack((np.ones(len(idx)), H1[:,idx]))
                if self.X_auxiliary is not None:
                    H1_ext = np.vstack((H1_ext, self.X_auxiliary[:,idx]))
                    # add additional rows for the auxiliary explanatory variables

                # P = probability matrix, same shape as X1
                D = W[1] @ H1_ext
                P = 1 / (1 + np.exp(-D))

                if self.full_dim:
                    grad = np.diag(W[1][:,k]) @ (P-X[1][:,idx])
                    H1[k, idx] = H1[k, idx] - (1 / (((i + 10) ** (0.5)) * (0 + 1))) * grad
                else:
                    grad_MF = (np.dot(A[k, :], H1[:,idx]) - B[k, idx])
                    #print('W[1].shape', W[1].shape)
                    grad_pred = np.diag(W[1][:,k+1]) @ (P-X[1][:,idx])
                    grad =  xi * grad_MF + grad_pred + a1 * np.sign(H1[k,idx])*np.ones(len(idx)) + a2 * H1[k, idx]
                    H1[k, idx] = H1[k, idx] - (1 / (((i + 10) ** (0.5)) * (A[k, k] + 1))) * grad

                if nonnegativity:
                    H1[k, idx] = np.maximum(H1[k, idx], np.zeros(shape=(len(idx),)))  # nonnegativity constraint

                if r is not None:  # usual sparse coding without radius restriction
                    d = np.linalg.norm(H1 - H0, 2)
                    H1 = H0 + (r / max(r, d)) * (H1 - H0)
                H0 = H1

            dist = np.linalg.norm(H1 - H1_old, 2) / np.linalg.norm(H1_old, 2)
            # print('!!! dist', dist)
            # H1_old = H1
            i = i + 1
            # print('!!!! i', i)  # mostly the loop finishes at i=1 except the first round


        return H1

    def validation(self,
                    result_dict=None,
                    X_test = None,
                    X_test_aux = None,
                    sub_iter=100,
                    verbose=False,
                    stopping_grad_ratio=0.0001,
                    SDL_option = 'filter', # or 'feature'
                    prediction_method_list = ['filter', 'naive', 'alt', 'exhaustive']):
        '''
        Given input X = [data, label] and initial loading dictionary W_ini, find W = [dict, beta] and code H
        by two-block coordinate descent: [dict, beta] --> H, H--> [dict, beta]
        Use Logistic MF model
        '''
        if result_dict is None:
            result_dict = self.result_dict
        if X_test is None:
            X_test = self.X_test
        if X_test_aux is None:
            X_test_aux = self.X_test_aux

        test_X = X_test[0]
        test_Y = X_test[1]

        W = result_dict.get('loading')
        beta = W[1].T
        # pred_threshold = result_dict.get('Opt_threshold (training)')
        # prediction threshold learned from training data

        for pred_type in prediction_method_list:
            print('!!! pred_type', pred_type)

            P_pred, H_test, Y_pred = self.predict(X_test = test_X,
                                            X_test_aux=X_test_aux,
                                            W=W,
                                            pred_threshold = None,
                                            SDL_option = SDL_option,
                                            method=pred_type) #or 'exhaust' or # naive

            fpr, tpr, thresholds = metrics.roc_curve(test_Y[0, :], P_pred, pos_label=None)
            mythre_test = thresholds[np.argmax(tpr - fpr)]
            myauc_test = metrics.auc(fpr, tpr)

            result_dict0 = compute_accuracy_metrics(test_Y[0,:], Y_pred, pred_type)
            self.result_dict.update(result_dict0)

        if verbose:
            fpr, tpr, thresholds = metrics.roc_curve(test_Y[0, :], P_pred, pos_label=None)
            mythre = thresholds[np.argmax(tpr - fpr)] # optimal prediction threshold for validation
            myauc = metrics.auc(fpr, tpr)
            ACC_list = [self.result_dict.get('Accuracy ({})'.format(pred_type)) for pred_type in prediction_method_list]
            #F_score_list = [self.result_dict.get('F_score ({})'.format(pred_type)) for pred_type in prediction_method_list]
            print("!!! ACC_list",ACC_list)
            #print("!!! F_score_list", F_score_list)
            ACC = np.max(np.asarray(ACC_list))
            rel_error_data =self.result_dict.get('Relative_reconstruction_loss (test)')
            # print('--- Validation --- [threshold, AUC, accuracy] = ', [np.round(mythre,3), np.round(myauc,3), np.round(accuracy, 3)])
            print('--- Validation ({}) --- [threshold, AUC, accuracy, rel_error_data] = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(pred_type, np.round(mythre,3), np.round(myauc,3), np.round(ACC, 3), np.round(rel_error_data,3)))

        return result_dict

    def predict(self,
                X_test, # no Y_test included
                X_test_aux=None,
                W=None,
                iter=10,
                pred_threshold=None,
                search_radius_const=10,
                SDL_option='filter',
                method='alt' # or 'exhaustive' or 'naive'
                ):
        '''
        Given input X = [data, ??] and loading dictionary W = [dict, beta], find missing label Y and code H
        by two-block coordinate descent
        '''

        W = self.loading
        X = self.X

        if pred_threshold is None:
            # Get threshold from training set
            if SDL_option == 'filter':
                X0_comp = W[0].T @ self.X[0]
                X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))
                if self.d3>0:
                    X0_ext = np.vstack((X0_ext, self.X_auxiliary))
                P_pred = np.matmul(W[1], X0_ext)
                P_pred = 1 / (np.exp(-P_pred) + 1)
                fpr, tpr, thresholds = metrics.roc_curve(X[1][0, :], P_pred[0,:], pos_label=None)
                pred_threshold = thresholds[np.argmax(tpr - fpr)]
                myauc_training = metrics.auc(fpr, tpr)


            elif SDL_option == 'feature':
                X0_comp = self.sparse_code(X[0], W[0], nonnegativity=False)
                X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))
                if self.d3>0:
                    X0_ext = np.vstack((X0_ext, self.X_auxiliary))
                P_pred = np.matmul(W[1], X0_ext)
                P_pred = 1 / (np.exp(-P_pred) + 1)
                # print('!!! error norm', np.linalg.norm(X[1][0, :]-P_pred[0,:])/X[1].shape[1])
                fpr, tpr, thresholds = metrics.roc_curve(self.X[1][0, :], P_pred[0,:], pos_label=None)
                pred_threshold = thresholds[np.argmax(tpr - fpr)]
                myauc_training = metrics.auc(fpr, tpr)

            self.result_dict.update({'Training_threshold':pred_threshold})
            self.result_dict.update({'Training_AUC':myauc_training})
            print('--- Training --- [threshold, AUC] = ', [np.round(pred_threshold,3), np.round(myauc_training,3)])

        # Now compute accuracy metrics on test set

        r = self.n_components
        n = X_test.shape[1]
        if W is None:
            W = self.loading
        # print("-- W[0][0,0]", np.linalg.norm(W[0][0,0]))

        if SDL_option == 'filter':
            H = W[0].T @ X_test
            if X_test_aux is not None:
                H = np.vstack((H, X_test_aux))
            H2 = np.vstack((np.ones(H.shape[1]), H))
            P_pred = np.matmul(W[1], H2)
            P_pred = 1 / (np.exp(-P_pred) + 1)  # predicted probability for Y_test
            # threshold predictive probabilities to get predictions
            Y_hat = P_pred.copy()
            Y_hat[Y_hat < pred_threshold] = 0
            Y_hat[Y_hat >= pred_threshold] = 1

            P_pred = P_pred[0,:]
            Y_hat = Y_hat[0,:]

        elif SDL_option == 'feature':
            if method == 'naive':
                H = self.sparse_code(X_test, W[0], nonnegativity=False)
                #print('---- H naive shape', H.shape)
                H_ext = np.vstack((np.ones(X_test.shape[1]), H))
                if X_test_aux is not None:
                    H_ext = np.vstack((H_ext, X_test_aux))
                P_pred = np.matmul(W[1], H_ext)
                P_pred = 1 / (np.exp(-P_pred) + 1)

                # threshold predictive probabilities to get predictions
                Y_hat = P_pred.copy()
                Y_hat[Y_hat < pred_threshold] = 0
                Y_hat[Y_hat >= pred_threshold] = 1

                P_pred = P_pred[0,:]
                Y_hat = Y_hat[0,:]

            elif method == 'alt':
                #print('alternating prection..')
                H = np.random.rand(r,n)
                Y_hat = np.random.rand(self.X[1].shape[0], X_test.shape[1])
                for step in trange(int(200)):
                    X = [X_test, Y_hat]

                    # Update code
                    radius = 10/(step+1)
                    H = self.update_code_joint_logistic(X, W, H, r=radius, sub_iter = 2, stopping_diff=0.0001)
                    # Update the missing label P_pred
                    H_ext = np.vstack((np.ones(X_test.shape[1]), H))

                    if X_test_aux is not None:
                        H_ext = np.vstack((H_ext, X_test_aux))

                    P_pred = np.matmul(W[1], H_ext)
                    P_pred = 1 / (np.exp(-P_pred) + 1)
                    Y_hat = P_pred

                    # threshold predictive probabilities to get predictions
                P_pred = P_pred[0,:]
                Y_hat[Y_hat < pred_threshold] = 0
                Y_hat[Y_hat >= pred_threshold] = 1
                Y_hat = Y_hat[0,:]

            elif method == 'exhaustive':
                print('exhaustive prection..')
                # Run over all possible y_hat values, do supervised sparse coding,
                # and find the one that gives minimum loss
                H = []
                Y_hat = []
                for i in trange(n):
                    loss_list = []
                    h_list = []
                    x_test = X_test[:,i][:,np.newaxis]

                    for j in np.arange(2):
                        y_guess = np.asarray([[j]])
                        x_guess = [x_test, y_guess]
                        h = self.update_code_joint_logistic(x_guess, W, xi=self.xi, sub_iter=40,
                                                            stopping_diff=0.001, H0=None, r=None)
                        h_ext = np.vstack((np.ones(1), h))
                        error_data = np.linalg.norm((x_test - W[0] @ h).reshape(-1, 1), ord=2) ** 2
                        error_label = np.sum(np.log(1+np.exp(W[1] @ h_ext))) - y_guess @ (W[1] @ h_ext).T
                        loss = (error_label + self.xi * error_data)[0,0]
                        # print('[j, loss] = ', [j, loss])
                        loss_list.append(loss)
                        h_list.append(h)

                    idx = np.argsort(loss_list)
                    #print('loss_list', loss_list)
                    # print('idx', idx)
                    y_hat = idx[0]
                    h_hat = h_list[idx[0]][:,0]

                    Y_hat.append(y_hat)
                    H.append(h_hat)

                Y_hat = np.asarray(Y_hat)
                #print('--- Y_hat', Y_hat)
                H = np.asarray(H).T
                H -= np.mean(H)
                H_ext = np.vstack((np.ones(X_test.shape[1]), H))
                P_pred = np.matmul(W[1], H_ext)
                P_pred = 1 / (np.exp(-P_pred) + 1)
                P_pred = P_pred[0,:]

        # Compute test data reconstruction loss
        H_test = self.sparse_code(X_test, W[0], nonnegativity=False)
        error_data = np.linalg.norm((X_test - W[0] @ H_test).reshape(-1, 1), ord=2)
        rel_error_data = error_data / np.linalg.norm(X_test.reshape(-1, 1), ord=2)

        # Save results
        self.result_dict.update({'Relative_reconstruction_loss (test)': rel_error_data})

        self.result_dict.update({'code_test': H_test})
        self.result_dict.update({'P_pred': P_pred})
        self.result_dict.update({'Y_hat': Y_hat})

        return P_pred, H, Y_hat

###### Helper functions

def sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)


def safe_vstack(Xs):
    if any(sp.issparse(X) for X in Xs):
        return sp.vstack(Xs)
    else:
        return np.vstack(Xs)


def update_code_within_radius(X, W, H0, r, a1=0, a2=0,
                              sub_iter=[2], stopping_grad_ratio=0.0001,
                              subsample_ratio=None, nonnegativity=True,
                              use_line_search=False):
    '''
    Find \hat{H} = argmin_H ( | X - WH| + alpha|H| ) within radius r from H0
    Use row-wise projected gradient descent
    Do NOT sparsecode the whole thing and then project -- instable
    12/5/2020 Lyu

    For NTF problems, X is usually tall and thin so it is better to subsample from rows
    12/25/2020 Lyu

    Apply single round of AdaGrad for rows, stop when gradient norm is small and do not make update
    12/27/2020 Lyu
    '''

    # print('!!!! X.shape', X.shape)
    # print('!!!! W.shape', W.shape)
    # print('!!!! H0.shape', H0.shape)

    if H0 is None:
        H0 = np.random.rand(W.shape[1], X.shape[1])
    H1 = H0.copy()
    i = 0
    dist = 1
    idx = np.arange(X.shape[0])
    H1_old = H1.copy()

    A = W.T @ W
    B = W.T @ X

    while (i < np.random.choice(sub_iter)):
        if_continue = np.ones(H0.shape[0])  # indexed by rows of H

        for k in [k for k in np.arange(H0.shape[0]) if if_continue[k]>0.5]:

            grad = np.dot(A[k, :], H1) - B[k, :]
            grad += a1 * np.sign(H1[k, :]) * np.ones(H0.shape[1]) + a2 * H1[k, :]
            grad_norm = np.linalg.norm(grad, 2)

            # Initial step size
            step_size = 1/(A[k,k]+1)
            # step_size = 1 / (np.trace(A)) # use the whole trace
            # step_size = 1
            if r is not None:  # usual sparse coding without radius restriction
                d = step_size * grad_norm
                step_size = (r / max(r, d)) * step_size

            H1_temp = H1.copy()
            # loss_old = np.linalg.norm(X - W @ H1)**2
            H1_temp[k, :] = H1[k, :] - step_size * grad
            if nonnegativity:
                H1_temp[k,:] = np.maximum(H1_temp[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
            #loss_new = np.linalg.norm(X - W @ H1_temp)**2
            #if loss_old > loss_new:

                # print('recons_loss:' , np.linalg.norm(X - W @ H1, ord=2) / np.linalg.norm(X, ord=2))

            """
            if use_line_search:
            # Armijo backtraking line search
                m = grad.T @ H1[k,:]
                H1_temp = H1.copy()
                loss_old = np.linalg.norm(X - W @ H1)**2
                loss_new = 0
                count = 0
                while (count==0) or (loss_old - loss_new < 0.1 * step_size * m):
                    step_size /= 2
                    H1_temp[k, :] = H1[k, :] - step_size * grad
                    if nonnegativity:
                        H1_temp[k,:] = np.maximum(H1_temp[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint
                    loss_new = np.linalg.norm(X - W @ H1_temp)**2
                    count += 1
            """
            H1 = H1_temp

        i = i + 1


    return H1

def code_update_sparse(X, W, H0=None, r=None, alpha=1, sub_iter=[5], stopping_grad_ratio=0.02, subsample_ratio=None, nonnegativity=True):
    '''
    Find \hat{H} = argmin_H ( || X - WH||^2 ) within radius r from H0
    With constraint hoyer_sparseness(rows of H) = sparsity
    s(x) = (\sqrt{n} - |x|_{1}/|x|_{2}) / (\sqrt{n} - 1)
    For dictionary update, one can input X.T and H.T to get W.T with sparse columns of W
    '''

    # print('!!!! H0.shape', H0.shape)

    if H0 is None:
        H0 = np.random.rand(W.shape[1], X.shape[1])

    H1 = H0.copy()

    dist = 1

    idx = np.arange(X.shape[0])
    # print('!!! X.shape', X.shape)


    if (subsample_ratio is not None) and (X.shape[0]>X.shape[1]):
        idx = np.random.randint(X.shape[0], size=X.shape[0]//subsample_ratio)
        A = W[idx,:].T @ W[idx,:]
        B = W[idx,:].T @ X[idx,:]

    else:
        A = W[:,:].T @ W[:,:]
        B = W[:,:].T @ X[:,:]

    for k in [k for k in np.arange(H0.shape[0])]:
        # block-optimize each row to induce row-wise sparsity
        i = 0
        while (i < np.random.choice(sub_iter)):
            H1_old = H1.copy()
            # row-wise gradient descent
            n = H0.shape[1]
            # grad_sparseness = (1/(np.sqrt(n)-1)) * (np.ones(H0.shape[1])-2*np.linalg.norm(H0[k,:],1)*(np.linalg.norm(H0[k,:],1)**(-2/3))*H0[k,:])/np.linalg.norm(H0[k,:],2)
            grad = (np.dot(A[k, :], H1) - B[k, :] + alpha * np.ones(H0.shape[1]))
            grad_norm = np.linalg.norm(grad, 2)
            step_size = (1 / (((i + 2) ** (1)) * (A[k, k] + 1)))
            if r is not None:  # usual sparse coding without radius restriction
                d = step_size * grad_norm
                step_size = (r / max(r, d)) * step_size

            if step_size * grad_norm / np.linalg.norm(H1_old, 2) > stopping_grad_ratio:
                H1[k, :] = H1[k, :] - step_size * grad

            # print('!!! H1.shape', H1.shap
            if nonnegativity:
                H1[k,:] = np.maximum(H1[k,:], np.zeros(shape=(H1.shape[1],)))  # nonnegativity constraint

            i = i + 1


    """
    for k in np.arange(H0.shape[0]):
        # Do Hoyer's projection to induce sparsity
        # (\sqrt{n} - L1 / |x|_{2}) / (\sqrt{n} - 1) = sparseness
        x = H1[k,:]
        n = H1.shape[1]
        L2 = np.linalg.norm(x)
        L1 = (np.sqrt(n) - sparsity * (np.sqrt(n)-1))*np.linalg.norm(x)
        H1[k,:] = hoyer_projection(x.copy(), L1, L2)
    """


    return H1



def fit_MLR_GD(Y, H, W0=None, sub_iter=100, stopping_diff=0.01):
        '''
        Convex optimization algorithm for Multiclass Logistic Regression using Gradient Descent
        Y = (n x k), H = (p x n) (\Phi in lecture note), W = (p x k)
        Multiclass Logistic Regression: Y ~ vector of discrete RVs with PMF = sigmoid(H.T @ W)
        MLE -->
        Find \hat{W} = argmin_W ( sum_j ( log(1+exp(H_j.T @ W) ) - Y.T @ H.T @ W ) )
        '''
        k = Y.shape[1] # number of classes
        if W0 is None:
            W0 = np.random.rand(H.shape[0],k) #If initial coefficients W0 is None, randomly initialize

        W1 = W0.copy()
        i = 0
        grad = np.ones(W0.shape)
        while (i < sub_iter) and (np.linalg.norm(grad) > stopping_diff):
            Q = 1/(1+np.exp(-H.T @ W1))  # probability matrix, same shape as Y
            # grad = H @ (Q - Y).T + alpha * np.ones(W0.shape[1])
            grad = H @ (Q - Y)
            W1 = W1 - (np.log(i+1) / (((i + 1) ** (0.5)))) * grad
            i = i + 1
            # print('iter %i, grad_norm %f' %(i, np.linalg.norm(grad)))
        return W1

def compute_accuracy_metrics(y_test, y_pred, pred_type=None):

    mcm = confusion_matrix(y_test, y_pred)
    tn = mcm[0, 0]
    tp = mcm[1, 1]
    fn = mcm[1, 0]
    fp = mcm[0, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    misclassification = 1 - accuracy
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fall_out = fp / (fp + tn)
    miss_rate = fn / (fn + tp)
    F_score = 2 * precision * recall / ( precision + recall )

    # Save results
    result_dict = {}
    #result_dict.update({'Relative_reconstruction_loss (test)': rel_error_data})
    result_dict.update({'Y_test': y_test})
    #result_dict.update({'P_pred': P_pred})
    result_dict.update({'Y_pred': y_pred})
    #result_dict.update({'AUC': myauc})
    #result_dict.update({'Training_threshold': mythre})
    #result_dict.update({'Opt_threshold': mythre_test})
    result_dict.update({'Accuracy': accuracy})
    result_dict.update({'Misclassification': misclassification})
    result_dict.update({'Precision': precision})
    result_dict.update({'Recall': recall})
    result_dict.update({'Sensitivity': sensitivity})
    result_dict.update({'Specificity': specificity})
    result_dict.update({'F_score': F_score})
    result_dict.update({'Fall_out': fall_out})
    result_dict.update({'Miss_rate': miss_rate})

    keys_list = list(result_dict.keys())

    for key in keys_list:
        if key not in ['Y_test', 'Y_pred']:
            key_new = key + " ({})".format(pred_type)
            result_dict[key_new] = result_dict.pop(key)

    return result_dict
