import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


#device = torch.device('cpu') # gpu or cpu

class smf(nn.Module):
    def __init__(self,
                 X_train, y_train,
                 hidden_size=4,
                 output_size=1,
                 device='cuda'):
        super(smf, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device =='cpu':
            self.device = torch.device('cpu')

        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device) # binary label. Need to revise for multi-class using one-hot encoding.
        #self.X_test = X_test
        #self.y_test = y_test
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.num_epochs = num_epochs
        #self.lr_classification = lr_classification
        #self.lr_matrix_factorization = lr_matrix_factorization
        #self.psi = psi
        self.model_Classification = self._initialize_classification_model().to(self.device)
        self.model_MF = self._initialize_matrix_factorization_model().to(self.device)
        #self.model_Classification_beta = self._initialize_classification_model_for_beta().to(self.device)

        self.result_dict = {}
        self.result_dict.update({'n_components' : hidden_size})
        self.result_dict.update({'Training_threshold': 0.5})

    def _initialize_classification_model(self):
        class Classification(nn.Module):
            def __init__(self, input_size, hidden_size, output_size=1):
                super(Classification, self).__init__()
                self.linear_W = nn.Linear(input_size, hidden_size, bias = False) # W.T @ X
                #nn.init.xavier_uniform_(self.linear_W.weight)  # Apply Xavier initialization
                #self.activation = nn.ReLU() # make W nonnegative
                self.linear_beta = nn.Linear(hidden_size, output_size) # activation beta.T @ (W.T @ X)
                #nn.init.xavier_uniform_(self.linear_beta.weight)  # Apply Xavier initialization
                self.Sigmoid = nn.Sigmoid()

            def forward(self, x):
                x1 = self.linear_W(x)
                #x2 = self.activation(x1)
                x3 = self.linear_beta(x1)
                x4 = self.Sigmoid(x3)
                return x4

        model = Classification(self.X_train.shape[1], self.hidden_size, self.output_size)

        return model.to(self.device)

    def _initialize_matrix_factorization_model(self):
        class MF(nn.Module):
            def __init__(self, X, hidden_size):
                super(MF, self).__init__()
                self.W = nn.Parameter(torch.rand(X.shape[0], hidden_size).clamp(min=1e-8))
                #nn.init.xavier_uniform_(self.W)  # Apply Xavier initialization
                self.H = nn.Parameter(torch.rand(hidden_size, X.shape[1]).clamp(min=1e-8))
                #nn.init.xavier_uniform_(self.H)  # Apply Xavier initialization

            def forward(self):
                return torch.mm(self.W, self.H)

        model = MF(self.X_train.T, self.hidden_size)
        return model




    def rank_r_projection(self, X, rank):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=rank, n_iter=7, random_state=42)
        X_reduced = svd.fit_transform(X.cpu())
        u = X_reduced.dot(np.linalg.inv(np.diag(svd.singular_values_)))
        s = svd.singular_values_
        vh = svd.components_
        r = rank
        u0 = u[:,:r]
        s0 = s[:r]
        v0 = vh[:r,:]
        recons = u0 @ np.diag(s0) @ v0
        return u0, s0, v0, recons

    def fit(self, num_epochs=1000,
            lr_classification=0.1,
            lr_matrix_factorization=0.1,
            xi=1,
            ini_loading = None,
            ini_code = None,
            initialize ='spectral',
            W_nonnegativity=True,
            H_nonnegativity=True,
            test_data=None, #or [X_test, y_test]
            record_recons_error=False):

        self.result_dict.update({'xi' : xi})
        self.result_dict.update({'nonnegativity' : [W_nonnegativity, H_nonnegativity]})
        self.result_dict.update({'iter': num_epochs})

        time_error = np.zeros(shape=[3, 0])
        elapsed_time = 0
        self.result_dict.update({"time_error": time_error})

        if ini_loading is not None:
            W0 = Variable(ini_loading[0]).to(self.device)
            Beta0 = Variable(ini_loading[1][:,1:]).to(self.device)
            Beta_bias = Variable(ini_loading[1][:,0]).to(self.device)

            self.model_MF.W = nn.Parameter(W0)
            self.model_Classification.linear_W.weight = nn.Parameter(W0.T)
            self.model_Classification.linear_beta.weight = nn.Parameter(Beta0)
            self.model_Classification.linear_beta.bias = nn.Parameter(Beta_bias)

        if ini_code is not None:
            H0 = Variable(ini_code).to(self.device)
            self.model_MF.H = nn.Parameter(H0)

        if  initialize == 'spectral':
            U0, S0, H0, recons = self.rank_r_projection(self.X_train.T, self.hidden_size)
            W0 = U0 #@ np.diag(S0)

            W0 = Variable(torch.from_numpy(W0)).float().to(self.device)
            H0 = Variable(torch.from_numpy(np.diag(S0) @ H0)).float().to(self.device)

            if ini_loading is None:
                self.model_MF.W = nn.Parameter(W0)
                self.model_Classification.linear_W.weight = nn.Parameter(W0.T)
            if ini_code is None:
                self.model_MF.H = nn.Parameter(H0)

        elif initialize == 'random':
            if ini_loading is None:
                self.model_MF.W = nn.Parameter(torch.rand(self.X_train.shape[1], self.hidden_size).to(self.device))
                self.model_Classification.linear_W.weight = nn.Parameter(W0.T)
            if ini_code is None:
                self.model_MF.H = nn.Parameter(torch.rand(self.hidden_size, self.X_train.shape[0]).to(self.device))

        criterion_Classification = nn.BCELoss()
        optimizer_Classification = optim.SGD(self.model_Classification.parameters(), lr=lr_classification)

        criterion_MF = nn.MSELoss()
        optimizer_MF = optim.SGD(self.model_MF.parameters(), lr=lr_matrix_factorization)

        #criterion_Classification_beta = nn.BCELoss()
        #optimizer_Classification_beta = optim.Adagrad(self.model_Classification_beta.parameters(), lr=lr_classification)

        for epoch in range(num_epochs):
            self.result_dict.update({'curren_epoch': epoch})
            start = time.time()


            y_hat = self.model_Classification(self.X_train)
            loss_Classification = criterion_Classification(y_hat.squeeze(), self.y_train.float())
            #optimizer_Classification.zero_grad()
            #loss_Classification.backward()
            #optimizer_Classification.step()

            X_hat = self.model_MF().to(self.device)
            loss_MF = criterion_MF(X_hat, self.X_train.T)
            #optimizer_MF.zero_grad()
            #loss_MF.backward()
            #optimizer_MF.step()


            X0 = np.asarray(self.X_train.T.detach().cpu().numpy())
            y_train_cpu = np.asarray(self.y_train.detach().cpu().numpy())
            y_train_cpu = y_train_cpu[np.newaxis,:]
            W0 = np.asarray(self.model_MF.W.data.detach().cpu().numpy())
            H0 = np.asarray(self.model_MF.H.data.detach().cpu().numpy())
            Beta_weight = self.model_Classification.linear_beta.weight
            Beta_weight = np.asarray(Beta_weight.detach().cpu().numpy())
            Beta_bias = self.model_Classification.linear_beta.bias
            Beta_bias = np.asarray(Beta_bias.detach().cpu().numpy())
            Beta_bias = Beta_bias[:,np.newaxis]
            Beta = np.hstack((Beta_bias, Beta_weight))
            W0 = [W0, Beta]

            W = update_dict_joint_logistic([X0, y_train_cpu], H0, W0, stopping_diff=0.0001,
                                             sub_iter = 5,
                                             r=None, nonnegativity=None,
                                             a1=0, a2=0,
                                             subsample_size = None)
            W = W/np.linalg.norm(W)


            # fitting logistic regression again with updated W
            #X0_comp = torch.mm(W.T, self.X_train.T).T
            X0_comp = W.T @ X0
            #X0_comp = np.asarray(X0_comp.detach().cpu().numpy())
            #y_train_cpu = np.asarray(self.y_train.detach().cpu().numpy())
            clf = LogisticRegression(random_state=0).fit(X0_comp.T, y_train_cpu[0])
            beta_weight = torch.from_numpy(clf.coef_).float().to(self.device)
            beta_bias = torch.from_numpy(clf.intercept_).float().to(self.device)


            # Code Update

            #X0 = np.asarray(self.X_train.T.detach().cpu().numpy())
            #W0 = np.asarray(common_W.detach().cpu().numpy())
            H0 = np.asarray(self.model_MF.H.data.detach().cpu().numpy())
            H = update_code_within_radius(X0, W, H0, r=None,
                                        a1=0, a2=0,
                                        nonnegativity=None)

            #print("np.linalg.norm(H0)=", np.linalg.norm(H0))
            #print("np.linalg.norm(H)=", np.linalg.norm(H))

            W = torch.from_numpy(W).float().to(self.device)
            H = torch.from_numpy(H).float().to(self.device)


            with torch.no_grad():
                self.model_Classification.linear_W.weight = nn.Parameter(W.T.clone())
                #self.model_Classification.linear_beta.weight = nn.Parameter(self.model_Classification_beta.linear_beta.weight.clone())
                #self.model_Classification.linear_beta.bias = nn.Parameter(self.model_Classification_beta.linear_beta.bias.clone())


                self.model_Classification.linear_beta.weight = nn.Parameter(beta_weight.clone())
                self.model_Classification.linear_beta.bias = nn.Parameter(beta_bias.clone())

            with torch.no_grad():
                self.model_MF.W = nn.Parameter(W.clone())
                self.model_MF.H = nn.Parameter(H.clone())


            end = time.time()
            elapsed_time += end - start
            self.result_dict.update({'elapsed_time': elapsed_time})

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],'
                      f'Loss_Classification: {loss_Classification.item():.4f}',
                      f'Loss_MF: {loss_MF.item():.4f}')

                if test_data is not None:
                    self.test(test_data[0], test_data[1])

                #print("Training accuracy=")
                #self.test(self.X_train, self.y_train)

                if record_recons_error:
                    loading = {}
                    W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
                    H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
                    Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
                    Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
                    Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))

                    self.result_dict.update({'loading': [W_dict, Beta_combined]})
                    self.result_dict.update({'code': H})

                    #print("common_W.norm()=", common_W.norm())
                    #print("np.linalg.norm(H0)=", np.linalg.norm(H0))
                    self.compute_recons_error()


        loading = {}
        W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
        H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
        Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
        Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
        Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))
        #print("Beta.shape=", Beta.shape)
        #print("Beta_bias.shape=", Beta_bias.shape)
        #print("Beta_combined.shape=", Beta_combined.shape)

        self.result_dict.update({'loading': [W_dict, Beta_combined]})
        self.result_dict.update({'code': H})
        return self.result_dict

    def compute_recons_error(self):
        # print the error every 50 iterations

        W = self.result_dict.get('loading')
        H = self.result_dict.get('code')
        X_train = np.asarray(self.X_train.cpu().numpy()).copy().T
        y_train = np.asarray(self.y_train.cpu().numpy()).copy()
        y_train = y_train.reshape(self.output_size, -1)
        X = [X_train, y_train]

        error_data = np.linalg.norm((X[0] - W[0] @ H).reshape(-1, 1), ord=2)**2
        rel_error_data = error_data / np.linalg.norm(X[0].reshape(-1, 1), ord=2)**2

        X0_comp = W[0].T @ X[0]
        X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))

        P_pred = np.matmul(W[1], X0_ext)
        P_pred = 1 / (np.exp(-P_pred) + 1)
        # print('!!! error norm', np.linalg.norm(X[1][0, :]-P_pred[0,:])/X[1].shape[1])
        #fpr, tpr, thresholds = metrics.roc_curve(X[1][0, :], P_pred[0,:], pos_label=None)

        P_pred = self.model_Classification.forward(self.X_train.to(self.device))
        P_pred = np.asarray(P_pred.detach().cpu().numpy()).T

        fpr, tpr, thresholds = metrics.roc_curve(X[1][0, :], P_pred[0,:], pos_label=None)

        mythre = thresholds[np.argmax(tpr - fpr)]
        myauc = metrics.auc(fpr, tpr)

        Y_hat = P_pred.copy()
        Y_hat[Y_hat < mythre] = 0
        Y_hat[Y_hat >= mythre] = 1
        P_pred = P_pred[0,:]
        Y_hat = Y_hat[0,:]

        self.result_dict.update({'Training_threshold':mythre})
        self.result_dict.update({'Training_AUC':myauc})
        print('--- Training --- [threshold, AUC] = ', [np.round(mythre,3), np.round(myauc,3)])

        mcm = confusion_matrix(y_train[0], Y_hat)
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

        self.result_dict.update({'Training_ACC':accuracy})
        #print("Training_ACC=", accuracy)

        #train_pred = self.model_Classification.forward(self.X_train.to(self.device))
        #print("train_pred=", train_pred[:10])
        #print("P_pred=", P_pred[:10])



        error_label = np.sum(np.log(1+np.exp(W[1] @ X0_ext))) - X[1] @ (W[1] @ X0_ext).T
        error_label = error_label[0][0]

        total_error_new = error_label + self.result_dict.get('xi') * error_data
        elapsed_time = self.result_dict.get("elapsed_time")
        time_error = self.result_dict.get("time_error")


        time_error = np.append(time_error, np.array([[elapsed_time, error_data, error_label]]).T, axis=1)
        print('--- Iteration %i: Training loss --- [Data, Label, Total] = [%f.3, %f.3, %f.3]' % (self.result_dict.get("curren_epoch"), error_data, error_label, total_error_new))

        self.result_dict.update({'Relative_reconstruction_loss (training)': rel_error_data})
        self.result_dict.update({'Classification_loss (training)': error_label})
        self.result_dict.update({'time_error': time_error})


    def test(self, X_test, y_test):
        with torch.no_grad():
            predictions = self.model_Classification(X_test.to(self.device))
            P_pred = np.asarray(predictions.detach().cpu().numpy()).T
            P_pred = P_pred[0,:]
            mythre = self.result_dict.get("Training_threshold")
            fpr, tpr, thresholds = metrics.roc_curve(y_test, P_pred, pos_label=None)
            mythre_test = thresholds[np.argmax(tpr - fpr)]
            myauc_test = metrics.auc(fpr, tpr)

            print("mythre=", mythre)
            print("mythre_test=", mythre_test)
            y_hat = (predictions.squeeze() > mythre).int()
            y_hat = np.asarray(y_hat.cpu().numpy())
            #print("Predictions:", predicted_labels.cpu().numpy())
            y_test = np.asarray(y_test.cpu().numpy()).copy()
            #accuracy = np.mean((y_test.cpu()==y_hat.cpu()).numpy())



            mcm = confusion_matrix(y_test, y_hat)
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


            self.result_dict.update({'Y_test': y_test})
            self.result_dict.update({'P_pred': P_pred})
            self.result_dict.update({'Y_pred': y_hat})
            self.result_dict.update({'AUC': myauc_test})
            self.result_dict.update({'Opt_threshold': mythre_test})
            self.result_dict.update({'Accuracy': accuracy})
            self.result_dict.update({'Misclassification': misclassification})
            self.result_dict.update({'Precision': precision})
            self.result_dict.update({'Recall': recall})
            self.result_dict.update({'Sensitivity': sensitivity})
            self.result_dict.update({'Specificity': specificity})
            self.result_dict.update({'F_score': F_score})
            self.result_dict.update({'Fall_out': fall_out})
            self.result_dict.update({'Miss_rate': miss_rate})

            print("Test accuracy = {}, Test AUC = {}".format(np.round(accuracy, 3), np.round(myauc_test, 3)) )


def update_dict_joint_logistic(X, H, W0, r, xi=0.1, a1=0, a2=0, sub_iter=2, stopping_diff=0.1, nonnegativity=True, subsample_size=None):
    '''
    X = [X0, X1]
    W = [W0, W1+W2]
    Find \hat{W} = argmin_W ( || X0 - W0 H||^2 + alpha|H| + Logistic_Loss(W[0].T @ X1, W[1])) within radius r from W0
    Compressed data = W[0].T @ X0 instead of H
    '''

    if W0 is None:
        W0 = np.random.rand(X[0].shape[0], H.shape[0])
        print('!!! W0.shape', W0.shape)

    #if not self.full_dim:
    A = H @ H.T

    W1 = W0[0].copy()
    i = 0
    dist = 1
    idx = np.arange(X[0].shape[0])
    while (i < sub_iter) and (dist > stopping_diff):
        W1_old = W1.copy()

        # Regression Parameters Update

        X0_comp = W1.T @ X[0]
        H1_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))

        D = W0[1] @ H1_ext
        P = 1 / (1 + np.exp(-D))

        grad_MF = (W1 @ H - X[0]) @ H.T
        grad_pred = X[0] @ (P-X[1]).T @ W0[1][:, 1: H.shape[0]+1] # exclude the first column of W[1] (intercept terms)
        grad = xi * grad_MF + grad_pred + a1 * np.sign(W1)*np.ones(shape=W1.shape) + a2 * W1
        # grad = grad_MF

        W1 -= (1 / (((i + 10) ** (0.5)) * (np.trace(A) + 1))) * grad

        if r is not None:  # usual sparse coding without radius restriction
            d = np.linalg.norm(W1 - W0[0], 2)
            W1 = W0[0] + (r / max(r, d)) * (W1 - W0[0])
        W0[0] = W1

        if nonnegativity:
            W1 = np.maximum(W1, np.zeros(shape=W1.shape))  # nonnegativity constraint

        dist = np.linalg.norm(W1 - W1_old, 2) / np.linalg.norm(W1_old, 2)
        dist = 1
        # print('!!! dist', dist)
        # H1_old = H1
        i = i + 1
        # print('!!!! i', i)  # mostly the loop finishes at i=1 except the first round


    return W1


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




def fit_LR_torch(input, output, device='cuda', num_epochs=10, lr=0.01):

    print("input.shape", input.shape)
    print("output.shape", output.shape)

    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device =='cpu':
        device0 = torch.device('cpu')

    class LogisticRegression(nn.Module):
        def __init__(self, input_size, output_size=1):
            super(LogisticRegression, self).__init__()
            self.linear_beta = nn.Linear(input_size, output_size)
            self.Sigmoid = nn.Sigmoid()

        def forward(self, a):
            x1 = self.linear_beta(a) # input a = W.T @ X
            x2 = self.Sigmoid(x1)
            return x2

        #model = __init__(input_size, output_size)

        #return model


    LR = LogisticRegression(input_size=input.shape[1],
                            output_size=1).to(device0)

    # Define the loss function and the optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(LR.parameters(), lr=lr)

    # Training loop

    for epoch in range(num_epochs):
        # Forward pass
        p_pred = LR.forward(input)
        # Calculate the loss
        loss = criterion(p_pred.squeeze().float(), output.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    beta_weight = LR.linear_beta.weight
    beta_bias = LR.linear_beta.bias
    return beta_weight, beta_bias
