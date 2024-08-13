import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
 
import time
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class smf(nn.Module):
    # Supervised Matrix Factorization by double 2-layer coupeld NN for GPU acceleration
    # Simultaneous dimension reduction and classification
    # Author: Joowon Lee and Hanbaek Lyu
    # REF = Joowon Lee, Hanbaek Lyu, and Weixin Yao,
    # “Supervised Matrix Factorization: Local Landscape Analysis and Applications ” ICML 2024
    # https://proceedings.mlr.press/v235/lee24p.html
    # (Nonnegative)MF + 2-layer NN classifier 
    # Model: Data (X) \approx Dictionary (W) @ Code (H)
    #        Label (Y) \approx logit(Beta.T @ W.T @ X) (for SMF-W)
    #        Label (Y) \approx logit(Beta.T @ H) (for SMF-H)


    def __init__(self,
                 X_train, y_train,
                 output_size,
                 hidden_size=4,
                 device='cuda'):
        super(smf, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device =='cpu':
            self.device = torch.device('cpu')

        if y_train.ndim == 1:
            self.multiclass = False
            self.output_size = 1
        else:
            self.multiclass = True
            self.output_size = y_train.shape[1]

        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device) 
        self.hidden_size = hidden_size

        if self.multiclass == True:
            self.model_Classification = self._initialize_multiclassification_model().to(self.device)
            self.model_Classification_beta = self._initialize_multiclassification_model_for_beta().to(self.device)
        else:
            self.model_Classification = self._initialize_classification_model().to(self.device)
            self.model_Classification_beta = self._initialize_classification_model_for_beta().to(self.device)
        self.model_MF = self._initialize_matrix_factorization_model().to(self.device)
        self.model_MF_H = self._initialize_matrix_factorization_model_for_H().to(self.device)

        self.result_dict = {}
        self.result_dict.update({'n_components' : hidden_size})
        self.result_dict.update({'Training_threshold': 0.5})

    def _initialize_classification_model(self):
        class Classification(nn.Module):
            def __init__(self, input_size, hidden_size, output_size=1):
                super(Classification, self).__init__()
                self.linear_W = nn.Linear(input_size, hidden_size, bias = False) # W.T @ X 
                self.linear_beta = nn.Linear(hidden_size, output_size) # activation beta.T @ (W.T @ X)
                # print(f"linear_beta's shape: {self.linear_beta.weight.shape}")
                # print(f"linear_W's shape: {self.linear_W.weight.shape}")

            def forward(self, x):
                x1 = self.linear_W(x)
                x2 = self.linear_beta(x1)
                print(f"x2's shape: {x2.shape}")
                x3 = torch.sigmoid(x2)
                return x3

        model = Classification(self.X_train.shape[1], self.hidden_size, self.output_size)
        return model.to(self.device)

    def _initialize_multiclassification_model(self):
        class MultiClassification(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MultiClassification, self).__init__()
                self.linear_W = nn.Linear(input_size, hidden_size, bias = False) # W.T @ X
                self.linear_beta = nn.Linear(hidden_size, output_size) # activation beta.T @ (W.T @ X)
                # print(f"!!! linear_beta: {self.linear_beta.weight.shape}")

            def forward(self, x):
                x1 = self.linear_W(x)
                x2 = self.linear_beta(x1)
                x3 = torch.zeros_like(x2)
                for i in range(x2.shape[0]):
                    max_i = torch.max(x2[i])
                    x3[i] = torch.exp(x2[i] - max_i) / (torch.exp(-max_i) + torch.sum(torch.exp(x2[i] - max_i)))
                # print(f"x3: {x3.shape}")
                return x3

        model = MultiClassification(self.X_train.shape[1], self.hidden_size, self.output_size)
        return model.to(self.device)
    
    def _initialize_matrix_factorization_model(self):
        class MF(nn.Module):
            def __init__(self, X, hidden_size):
                super(MF, self).__init__()
                self.W = nn.Parameter(torch.rand(X.shape[0], hidden_size).clamp(min=1e-8))
                self.H = nn.Parameter(torch.rand(hidden_size, X.shape[1]).clamp(min=1e-8))
                # print(f"W: {self.W.shape}")
                # print(f"H: {self.H.shape}")

            def forward(self):
                return torch.mm(self.W, self.H)

        model = MF(self.X_train.T, self.hidden_size)
        return model

    def _initialize_classification_model_for_beta(self):
        class Classification_beta(nn.Module):
            def __init__(self, hidden_size, output_size=1):
                super().__init__()
                self.linear_beta = nn.Linear(hidden_size, output_size)

            def forward(self, a):
                act = self.linear_beta(a) # input a = W.T @ X
                y_pred = torch.softmax(act)
                return y_pred

        model = Classification_beta(self.hidden_size, self.output_size)
        return model.to(self.device)
    
    def _initialize_multiclassification_model_for_beta(self):
        class MultiClassification_beta(nn.Module):
            def __init__(self, hidden_size, output_size=1):
                super().__init__()
                self.linear_beta = nn.Linear(hidden_size, output_size)
                # print(f"The beta: {self.linear_beta.weight.shape}")

            def forward(self, a):
                act = self.linear_beta(a) # input a = W.T @ X
                y_pred = torch.zeros_like(act)
                for i in range(act.shape[0]):
                    max_i = torch.max(act[i])
                    y_pred[i] = torch.exp(act[i] - max_i) / (torch.exp(-max_i) + torch.sum(torch.exp(act[i] - max_i)))
                return y_pred

        model = MultiClassification_beta(self.hidden_size, self.output_size)
        return model.to(self.device)

    def _initialize_matrix_factorization_model_for_H(self):
        class MF_H(nn.Module):
            def __init__(self, X, hidden_size):
                super(MF_H, self).__init__()
                self.H = nn.Parameter(torch.rand(hidden_size, X.shape[1]).clamp(min=1e-8))

            def forward(self, W):
                return torch.mm(W, self.H)

        model = MF_H(self.X_train.T, self.hidden_size)
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
            ini_loading=None,
            ini_code=None,
            initialize='spectral',
            W_nonnegativity=True,
            H_nonnegativity=True,
            test_data=None, #or [X_test, y_test]
            record_recons_error=False,
            threshold = 0.5):

        self.result_dict.update({'xi' : xi})
        self.result_dict.update({'nonnegativity' : [W_nonnegativity, H_nonnegativity]})
        self.result_dict.update({'iter': num_epochs})

        time_error = np.zeros(shape=[3, 0])
        elapsed_time = 0
        self.result_dict.update({"time_error": time_error})

        # ini_loading = [W, beta]
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
                W0 = torch.rand(self.X_train.shape[1], self.hidden_size).to(self.device)
                self.model_MF.W = nn.Parameter(W0)
                self.model_Classification.linear_W.weight = nn.Parameter(W0.T)
            if ini_code is None:
                self.model_MF.H = nn.Parameter(torch.rand(self.hidden_size, self.X_train.shape[0]).to(self.device))

        criterion_Classification = nn.CrossEntropyLoss()
        criterion_MF = nn.MSELoss()

        optimizer_Classification = optim.Adagrad(self.model_Classification.parameters(), lr=lr_classification, weight_decay=0.1)
        optimizer_MF = optim.Adagrad(self.model_MF.parameters(), lr=lr_matrix_factorization, weight_decay=0)

        if record_recons_error:
            if self.multiclass == False:
                self.result_dict.update({'curren_epoch': -1})
                self.result_dict.update({'elapsed_time': 0})

                W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
                H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
                Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
                Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
                Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))

                self.result_dict.update({'loading': [W_dict, Beta_combined]})
                self.result_dict.update({'code': H})
                self.compute_recons_error()
            else:
                self.result_dict.update({'curren_epoch': -1})
                self.result_dict.update({'elapsed_time': 0})

                W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
                H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
                Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
                Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
                Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))

                self.result_dict.update({'loading': [W_dict, Beta_combined]})
                self.result_dict.update({'code': H})
                self.compute_recons_error_multi()
                        

        for epoch in range(num_epochs):
            self.result_dict.update({'curren_epoch': epoch})
            start = time.time()

            # Update W
            optimizer_Classification.zero_grad()
            y_hat = self.model_Classification(self.X_train)
            # print(f"!!! y_hat.shape: {y_hat.squeeze().shape}")
            # print(f" !!! y_train.shape: {self.y_train.shape}")
            loss_Classification = criterion_Classification(y_hat.squeeze(), self.y_train.float())
            loss_Classification.backward()
            optimizer_Classification.step()

            optimizer_MF.zero_grad()
            X_hat = self.model_MF().to(self.device)
            loss_MF = criterion_MF(X_hat, self.X_train.T)
            loss_MF.backward()
            optimizer_MF.step()

            common_W = (xi/(1+xi)) * self.model_MF.W.data.to(self.device) + (1/(1+xi)) * self.model_Classification.linear_W.weight.T
            common_W = common_W/ common_W.norm()
            common_W = common_W.to(self.device)

            if W_nonnegativity:
                common_W = common_W.clamp(min=1e-8)

            with torch.no_grad():
                self.model_Classification.linear_W.weight = nn.Parameter(common_W.T.clone())

            with torch.no_grad():
                self.model_MF.W = nn.Parameter(common_W.clone())

            X0 = np.asarray(self.X_train.T.detach().cpu().numpy())
            y_train_cpu = np.asarray(self.y_train.detach().cpu().numpy())
            y_train_cpu = y_train_cpu[np.newaxis,:]
            y_train_cpu = y_train_cpu[0]
            W0 = np.asarray(self.model_MF.W.data.detach().cpu().numpy())
            X0_comp = W0.T @ X0
            # print(f"!!! y_train_cpu.shape: {y_train_cpu.shape}")

            # fitting logistic regression again with updated W
            ### Multinomial Case
            if self.multiclass == True:
                label_vec = np.copy(y_train_cpu.T)
                for i in range(1, label_vec.shape[0]):
                    label_vec[i, :][label_vec[i, :] == 1] = i+1
                label_vec = np.sum(label_vec, axis=0)
                clf = LogisticRegression(random_state=0, max_iter=300).fit(X0_comp.T, label_vec)
                coef = np.zeros((self.y_train.shape[1], W0.shape[1]))
                for row in range(self.y_train.shape[1]):
                    coef[row] = clf.coef_[row+1] - clf.coef_[0]
                intercepts = np.zeros(self.y_train.shape[1])
                for i in range(self.y_train.shape[1]):
                    intercepts[i] = clf.intercept_[i+1] - clf.intercept_[0]
                beta_weight = torch.from_numpy(coef).float().to(self.device)
                beta_bias = torch.from_numpy(intercepts).float().to(self.device)
                # W[1] = self.update_beta_joint_logistic(X, H, W, stopping_diff=0.0001,
                #                                  sub_iter = 5,
                #                                  r=search_radius, nonnegativity=self.nonnegativity[1],
                #                                  a1=self.L1_reg[1], a2=self.L2_reg[1],
                #                                  subsample_size = None)
            ### Binomial Case
            else:
                clf = LogisticRegression(random_state=0).fit(X0_comp.T, y_train_cpu)
                beta_weight = torch.from_numpy(clf.coef_).float().to(self.device)
                beta_bias = torch.from_numpy(clf.intercept_).float().to(self.device)

            """
            # torch version
            criterion_Classification_beta = nn.CrossEntropyLoss()
            optimizer_Classification_beta = optim.Adam(self.model_Classification_beta.parameters(), lr=0.1, weight_decay=0.1)
            common_W = torch.from_numpy(W).float().to(self.device)
            a1 = torch.mm(common_W.T, self.X_train.T).T
            for epoch1 in range(10):
                optimizer_Classification_beta.zero_grad()
                y_hat1 = self.model_Classification_beta(a1)
                loss_Classification_beta = criterion_Classification_beta(y_hat1.squeeze(), self.y_train.float())
                loss_Classification_beta.backward(retain_graph=True)
                optimizer_Classification_beta.step()
            """

            # find H with updated W
            criterion_MF_H = nn.MSELoss()
            optimizer_MF_H = optim.Adagrad(self.model_MF_H.parameters(), lr=1)
            for epoch1 in range(5):
                optimizer_MF_H.zero_grad()
                X_hat1 = self.model_MF_H(common_W)
                loss_MF_H = criterion_MF_H(X_hat1, self.X_train.T)
                loss_MF_H.backward(retain_graph=True)
                optimizer_MF_H.step()

                if H_nonnegativity:
                    self.model_MF.H.data = self.model_MF.H.data.clamp(min=1e-8)

            with torch.no_grad():
                self.model_Classification.linear_beta.weight = nn.Parameter(beta_weight.clone())
                self.model_Classification.linear_beta.bias = nn.Parameter(beta_bias.clone())

            with torch.no_grad():
                self.model_MF.H = nn.Parameter(self.model_MF_H.H.data.clone())

            end = time.time()
            elapsed_time += end - start
            self.result_dict.update({'elapsed_time': elapsed_time})

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}],'
                      f'Loss_Classification: {loss_Classification.item():.4f}',
                      f'Loss_MF: {loss_MF.item():.4f}')

                if test_data is not None:
                    if self.multiclass == False:
                        self.test(test_data[0], test_data[1])
                    else:
                        self.test_multi(test_data[0], test_data[1])

                if record_recons_error:
                    loading = {}
                    W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
                    H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
                    Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
                    Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
                    Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))

                    self.result_dict.update({'loading': [W_dict, Beta_combined]})
                    self.result_dict.update({'code': H})
                    if self.multiclass == False:
                        self.compute_recons_error()
                    else:
                        self.compute_recons_error_multi

        loading = {}
        W_dict = np.asarray(self.model_MF.W.data.cpu().numpy()).copy()
        H = np.asarray(self.model_MF.H.data.cpu().numpy()).copy()
        Beta = np.asarray(self.model_Classification.linear_beta.weight.detach().cpu().numpy()).copy()
        Beta_bias = np.asarray(self.model_Classification.linear_beta.bias.detach().cpu().numpy()).copy()
        Beta_combined = np.hstack((Beta_bias.reshape(self.output_size, -1),Beta))

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
    
    def compute_recons_error_multi(self):
        # print the error every 50 iterations
        W = self.result_dict.get('loading')
        H = self.result_dict.get('code')
        X_train = np.asarray(self.X_train.cpu().numpy()).copy().T
        y_train = np.asarray(self.y_train.cpu().numpy()).copy().T
        X = [X_train, y_train]

        error_data = np.linalg.norm((X[0] - W[0] @ H).reshape(-1, 1), ord=2)**2
        rel_error_data = error_data / np.linalg.norm(X[0].reshape(-1, 1), ord=2)**2

        X0_comp = W[0].T @ X[0]
        X0_ext = np.vstack((np.ones(X[1].shape[1]), X0_comp))
        # print(f"The X0_ext: {X0_ext.shape}")
        # print(f"The W[1]: {W[1].shape}")
        # print(f"X[1] : {X[1].shape}")
        error_label = np.sum(1 + np.sum(np.exp(W[1]@X0_ext), axis=0)) - np.trace(X[1].T @ W[1] @ X0_ext)
        total_error_new = error_label + self.result_dict.get('xi') * error_data

        elapsed_time = self.result_dict.get("elapsed_time")
        time_error = self.result_dict.get("time_error")
        time_error = np.append(time_error, np.array([[elapsed_time, error_data, error_label]]).T, axis=1)
        print('--- Iteration %i: Training loss --- [Data, Label, Total] = [%f.3, %f.3, %f.3]' % (self.result_dict.get("curren_epoch"), error_data, error_label, total_error_new))

        self.result_dict.update({'Relative_reconstruction_loss (training)': rel_error_data})
        self.result_dict.update({'Classification_loss (training)': error_label})
        self.result_dict.update({'time_error': time_error.T})

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
            y_test = np.asarray(y_test.cpu().numpy()).copy()

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

    def test_multi(self, X_test, y_test):
        with torch.no_grad():
            predictions = self.model_Classification(X_test.to(self.device))
            P_pred = np.asarray(predictions.detach().cpu().numpy())

            mythre = self.result_dict.get("Training_threshold")
            print("mythre=", mythre)

            y_hat = (predictions.squeeze() > mythre).int()
            y_hat = np.asarray(y_hat.cpu().numpy())
            y_test = np.asarray(y_test.cpu().numpy()).copy()

            y_test_result = []
            y_pred_result = []
            
            for i in np.arange(y_test.shape[0]):
                for j in np.arange(y_test.shape[1]):
                    if y_test[i,j] == 1:
                        y_test_result.append(1)
                    else:
                        y_test_result.append(0)
                    if P_pred[i,j] >= mythre:
                        y_pred_result.append(1)
                    else:
                        y_pred_result.append(0)

            confusion_mx = metrics.confusion_matrix(y_test_result, y_pred_result)
            accuracy = np.trace(confusion_mx)/np.sum(np.sum(confusion_mx))
            self.result_dict.update({'Accuracy': accuracy})
            
            print("Test accuracy = {}, Test confusion_mx = {}".format(accuracy, confusion_mx))
            


def display_dictionary(W, save_name=None, score=None, grid_shape=None):
    k = int(np.sqrt(W.shape[0]))
    rows = int(np.sqrt(W.shape[1]))
    cols = int(np.sqrt(W.shape[1]))
    if grid_shape is not None:
        rows = grid_shape[0]
        cols = grid_shape[1]

    figsize0=(6, 6)
    if (score is None) and (grid_shape is not None):
       figsize0=(cols, rows)
    if (score is not None) and (grid_shape is not None):
       figsize0=(cols, rows+0.2)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize0,
                            subplot_kw={'xticks': [], 'yticks': []})


    for ax, i in zip(axs.flat, range(100)):
        if score is not None:
            idx = np.argsort(score)
            idx = np.flip(idx)

            ax.imshow(W.T[idx[i]].reshape(k, k), cmap="viridis", interpolation='nearest')
            ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)
        else:
            ax.imshow(W.T[i].reshape(k, k), cmap="viridis", interpolation='nearest')
            if score is not None:
                ax.set_xlabel('%1.2f' % score[i], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)

    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    if save_name is not None:
        plt.savefig( save_name, bbox_inches='tight')
    plt.show()


def rank_r_projection(X, rank):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=rank, n_iter=7, random_state=42)
    X_reduced = svd.fit_transform(X)
    u = X_reduced.dot(np.linalg.inv(np.diag(svd.singular_values_)))
    s = svd.singular_values_
    vh = svd.components_
    r = rank
    u0 = u[:,:r]
    s0 = s[:r]
    v0 = vh[:r,:]
    recons = u0 @ np.diag(s0) @ v0
    return u0, s0, v0, recons

def find_initial(X, Y, covariate = None, r = 16, generate="random"):

    ## input
    # X : p x n matrix
    # Y : 1 x n matrix
    # r : number of components
    # covariate : p1 x n matrix (if any)

    ## output
    # W0 = [W0, beta_coef] : (p x r) initial loading, (r + p1) x 1 regression coefficient
    # [0] feature based W0
    # [1] filter based W0
    # [2] H0 : r x n initial code

    logistic_model = LogisticRegression(solver='liblinear', random_state=0)

    if generate == "spectral":
        U0, S0, H0, recons = rank_r_projection(X, r)
        W0 = U0 @ np.diag(S0)

        if covariate is not None:
            temp_X_H = np.hstack([H0.T, covariate.T]) # feature based
            logit_fit_H = logistic_model.fit(temp_X_H, Y.T)

            temp_X_W = np.hstack([X.T @ W0, covariate.T]) # filter based (Replace H0 with W0.T @ X)
            logit_fit_W = logistic_model.fit(temp_X_W, Y.T)
        else:
            logit_fit_H = logistic_model.fit(H0.T, Y.T)
            logit_fit_W = logistic_model.fit(X.T @ W0, Y.T)
    elif generate == "random":
        W0 = np.random.rand(X.shape[0], r)
        H0 = np.random.rand(r, X.shape[1])
        logit_fit_H = logistic_model.fit(H0.T, Y.T)
        logit_fit_W = logistic_model.fit(X.T @ W0, Y.T)

    reg_coef_H = np.asarray([np.append(logit_fit_H.intercept_[0], logit_fit_H.coef_[0])])
    reg_coef_W = np.asarray([np.append(logit_fit_W.intercept_[0], logit_fit_W.coef_[0])])

    #reg_coef_H = 1-2*np.random.rand(reg_coef_H.shape[0],reg_coef_H.shape[1])
    #reg_coef_W = 1-2*np.random.rand(reg_coef_H.shape[0],reg_coef_H.shape[1])

    return [W0,reg_coef_H], [W0,reg_coef_W], H0
