# -*- coding: utf-8 -*-
"""


"""

import sys

sys.path.insert(0, '../../Utilities/')

import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
import math
import tensorflow as tf
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import datetime
from pyDOE import lhs
from scipy.integrate import odeint

start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

# %%
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t_train, X_train, T_trian, H_trian, U0, t_f, lb, ub, layers, layers_alpha, sf):

        self.sf = sf

        # Data for training
        self.t_train = t_train
        self.X_train = X_train
        self.T_train = T_trian
        self.H_train = H_trian

        self.X0 = U0[0]
        self.t_f = t_f

        # Bounds
        self.lb = lb
        self.ub = ub

        self.Tlb = self.T_train.min(0)
        self.Tub = self.T_train.max(0)

        self.Hlb = self.H_train.min(0)
        self.Hub = self.H_train.max(0)

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.weights_alpha, self.biases_alpha = self.initialize_NN(layers_alpha)

        # Fixed parameters

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.saver = tf.train.Saver()

        # placeholders for inputs
        self.t_u = tf.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])
        self.X_u = tf.placeholder(tf.float64, shape=[None, self.X_train.shape[1]])
        self.X0_u = tf.placeholder(tf.float64, shape=[None, self.X0.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])
        self.T_u = tf.placeholder(tf.float64, shape=[None, self.T_train.shape[1]])
        self.H_u = tf.placeholder(tf.float64, shape=[None, self.H_train.shape[1]])
        # physics informed neural networks
        self.X_pred = self.net_u(self.t_u)

        ##
        self.alpha_pred = self.net_alpha(self.t_u, self.T_u, self.H_u)

        self.X0_pred = self.X_pred[0]

        self.X_f = self.net_f(self.t_u, self.T_u, self.H_u)

        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.X0_u - self.X0_pred))

        self.lossU = tf.reduce_mean(tf.square(self.X_u - self.X_pred))

        self.lossF = tf.reduce_mean(tf.square(self.X_f))

        self.loss = self.lossU0 + self.lossU + 100 * self.lossF

        # Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Initialize the nueral network

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # weights for the current layer
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64),
                            dtype=tf.float64)  # biases for the current layer
            weights.append(W)  # save the elements in W to weights (a row vector)
            biases.append(b)  # save the elements in b to biases (a 1Xsum(layers) row vector)
        return weights, biases

    # generating weights
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    # Architecture of the neural network
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1

        Z = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            Z = tf.tanh(tf.add(tf.matmul(Z, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(Z, W), b)
        return Y

    def neural_net_TH(self, t, T, H,weights, biases):
        num_layers = len(weights) + 1
        t = 2.0 * (t-  self.lb)  / (self.ub - self.lb)-1.0
        T = 2.0 * (T - self.Tlb) / (self.Tub - self.Tlb) - 1.0
        H = 2.0 * (H - self.Hlb) / (self.Hub - self.Hlb) - 1.0
        Z = tf.concat([T, H, t], axis=1)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            Z = tf.tanh(tf.add(tf.matmul(Z, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(Z, W), b)
        return Y


    def net_u(self, t):
        X = self.neural_net(t, self.weights, self.biases)
        return X

    def net_alpha(self, t,T, H):
        alpha = self.neural_net_TH(t,T, H,self.weights_alpha, self.biases_alpha)
        bound_alpha = [tf.constant(-0.4321, dtype=tf.float64), tf.constant(0.8007, dtype=tf.float64)]
        return bound_alpha[0] + (bound_alpha[1] - bound_alpha[0]) * tf.sigmoid(alpha)

    def net_f(self, t, T, H):

        alpha = self.net_alpha(t,T,H)
        X = self.net_u(t)
        X_t = tf.gradients(X, t, unconnected_gradients='zero')[0]

        f_X = X_t - (0.372 - alpha) * X + (0.0008)* (X ** 2) * self.sf
        return f_X

    def callback(self, loss, lossU0, lossU, lossF):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e'
              % (loss, lossU0, lossU, lossF))

    def train(self, nIter):

        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f,
                   self.X_u: self.X_train, self.X0_u: self.X0, self.T_u: self.T_train, self.H_u: self.H_train}

        start_time = time.time()
        for it in range(nIter + 1):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lossU0_value = self.sess.run(self.lossU0, tf_dict)
                lossU_value = self.sess.run(self.lossU, tf_dict)
                lossF_value = self.sess.run(self.lossF, tf_dict)
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value]))
                print('It: %d, Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, Time: %.2f' %
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, elapsed))
                start_time = time.time()

        if LBFGS:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,  # Inputs of the minimize operator
                                    fetches=[self.loss, self.lossU0, self.lossU, self.lossF],
                                    loss_callback=self.callback)  # Show the results of minimize operator

    def predict_data(self, t_star):

        tf_dict = {self.t_u: t_star}

        X = self.sess.run(self.X_pred, tf_dict)

        return X

    def predict_par(self, t_star, T_star, H_star):
        feed_dict = {self.T_u: T_star, self.H_u: H_star, self.t_u:t_star}
        alpha = self.sess.run(self.alpha_pred, feed_dict)
        return alpha


############################################################
if __name__ == "__main__":

    # Architecture of  the NN
    layers = [1] + 5 * [32] + [1]  # The inout is t while the outputs are x
    layers_alpha = [3] + 3 * [64] + [1]  # 输入3维输出一维  T H t共同决定alpha的值


    # Load data
    data_frame = pandas.read_csv('Data/Data_2020-2023.csv')
    data_frame24 = pandas.read_csv('Data/Data_2020-2024.csv')

    #轩 加的
    data_frame24only = pandas.read_csv('Data/Data_2024.csv')

    # data 24only
    X_star24only = data_frame24only['X']  # T x 1 array
    # 3 days average
    # X_star = X_star.rolling(window=3).mean()
    # X_star = X_star[2:]
    T_star24only = data_frame24only['T']
    H_star24only = data_frame24only['H']

    # data 2022
    X_star = data_frame['X']  # T x 1 array
    # 3 days average
    # X_star = X_star.rolling(window=3).mean()
    # X_star = X_star[2:]
    T_star = data_frame['T']
    H_star = data_frame['H']

    # data 2024
    X_star24 = data_frame24['X']  # T x 1 array
    # 3 days average
    # X_star = X_star.rolling(window=3).mean()
    # X_star = X_star[2:]
    T_star24 = data_frame24['T']
    H_star24 = data_frame24['H']


    # 2022  数据转换
    X_star = X_star.to_numpy(dtype=np.float64)
    X_star = X_star.reshape([len(X_star), 1])
    T_star = T_star.to_numpy(dtype=np.float64)
    T_star = T_star.reshape([len(T_star), 1])
    H_star = H_star.to_numpy(dtype=np.float64)
    H_star = H_star.reshape([len(H_star), 1])
    X0 = X_star[0]
    X0 = np.array(X0)

    # 2024  数据转换
    X_star24 = X_star24.to_numpy(dtype=np.float64)
    X_star24 = X_star24.reshape([len(X_star24), 1])
    T_star24 = T_star24.to_numpy(dtype=np.float64)
    T_star24 = T_star24.reshape([len(T_star24), 1])
    H_star24 = H_star24.to_numpy(dtype=np.float64)
    H_star24 = H_star24.reshape([len(H_star24), 1])
    X024 = X_star24[0]
    X024 = np.array(X024)

    # 2024only  数据转换
    X_star24only = X_star24only.to_numpy(dtype=np.float64)
    X_star24only = X_star24only.reshape([len(X_star24only), 1])
    T_star24only = T_star24only.to_numpy(dtype=np.float64)
    T_star24only = T_star24only.reshape([len(T_star24only), 1])
    H_star24only = H_star24only.to_numpy(dtype=np.float64)
    H_star24only = H_star24only.reshape([len(H_star24only), 1])
    X024only = X_star24only[0]
    X024only = np.array(X024only)


    #  时间
    t_star = np.arange(len(X_star))
    t_star = t_star.reshape([len(t_star), 1])
    #
    # lower and upper bounds
    lb = t_star.min(0)
    ub = t_star.max(0)

    # Scaling
    sf = 1
    X0 = X0 * sf
    X024 =  X024 * sf
    X_star = X_star * sf
    X_star0 = X_star[0:1, :]
    U0 = [X_star0]

    N_f = 90
    # t_f = lb + (ub - lb) * lhs(1, N_f)
    t_f = np.linspace(lb, ub, num=N_f, endpoint=True)

    ######################################################################
    ######################## Training and Predicting #####################
    ######################################################################
    t_train = t_star
    X_train = X_star
    T_trian = (T_star - 21) ** 2
    H_trian = (H_star - 84) ** 2
    from datetime import datetime

    now = datetime.now()
    # dt_string = now.strftime("%m-%d-%H-%M")
    dt_string = now.strftime("%m-%d")

    # save results
    current_directory = os.getcwd()
    for j in range(20):
        casenumber = 'set' + str(j + 1)
        #  存结果
        relative_path_results = '/Model1/Train-Results-' + dt_string + '-' + casenumber + '/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)
        # 存模型
        relative_path = '/Model1/Train-model-' + dt_string + '-' + casenumber + '/'
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            # break

        ####model
        total_records = []
        total_records_LBFGS = []
        model = PhysicsInformedNN(t_train, X_train, T_trian, H_trian, U0, t_f, lb, ub,layers, layers_alpha, sf)
        ####Training
        LBFGS = True
        # LBFGS = False
        model.train(10000)  # Training with n iterations

        ####save model
        model.saver.save(model.sess, save_models_to + "model.ckpt")

        ####Predicting
        X = model.predict_data(t_star)
        alpha = model.predict_par(t_star, T_trian, H_trian)
        import datetime

        end_time = time.time()
        print(datetime.timedelta(seconds=int(end_time - start_time)))

        ##################save data and plot  PINN

        ####save data
        np.savetxt(save_results_to + 'X.txt', X.reshape((-1, 1)))
        np.savetxt(save_results_to + 't_star.txt', t_star.reshape((-1, 1)))
        np.savetxt(save_results_to + 'alpha.txt', alpha.reshape((-1, 1)))


        ####records for Adam
        N_Iter = len(total_records)
        iteration = np.asarray(total_records)[:, 0]
        loss_his = np.asarray(total_records)[:, 1]
        loss_his_u0 = np.asarray(total_records)[:, 2]
        loss_his_u = np.asarray(total_records)[:, 3]
        loss_his_f = np.asarray(total_records)[:, 4]

        ####records for LBFGS
        if LBFGS:
            N_Iter_LBFGS = len(total_records_LBFGS)
            iteration_LBFGS = np.arange(N_Iter_LBFGS) + N_Iter * 100
            loss_his_LBFGS = np.asarray(total_records_LBFGS)[:, 0]
            loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:, 1]
            loss_his_u_LBFGS = np.asarray(total_records_LBFGS)[:, 2]
            loss_his_f_LBFGS = np.asarray(total_records_LBFGS)[:, 3]

        ####save records
        np.savetxt(save_results_to + 'iteration.txt', iteration.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his.txt', loss_his.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u0.txt', loss_his_u0.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u.txt', loss_his_u.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_f.txt', loss_his_f.reshape((-1, 1)))

        if LBFGS:
            np.savetxt(save_results_to + 'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1, 1)))

            ######################################################################
        # ############################# Plotting ###############################
        # ######################################################################
        SAVE_FIG = True

        # History of loss
        font = 24
        fig, ax = plt.subplots()
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize=font)
        plt.ylabel('$loss values$', fontsize=font)
        plt.yscale('log')
        plt.grid(True)
        plt.plot(iteration, loss_his, label='$loss$')
        plt.plot(iteration, loss_his_u0, label='$loss_{u0}$')
        plt.plot(iteration, loss_his_u, label='$loss_u$')
        plt.plot(iteration, loss_his_f, label='$loss_f$')
        if LBFGS:
            plt.plot(iteration_LBFGS, loss_his_LBFGS, label='$loss-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u0_LBFGS, label='$loss_{u0}-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u_LBFGS, label='$loss_u-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_f_LBFGS, label='$loss_f-LBFGS$')
        plt.legend(loc="upper right", fontsize=24, ncol=4)
        plt.legend()
        ax.tick_params(axis='both', labelsize=24)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'History_loss.png', dpi=300)
        #
        #
        # X
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, X / sf, 'r-', lw=4, color=(204 / 255, 190 / 255, 230 / 255), label='PCM-NN fitting')
        ax.plot(t_star, X_star / sf, 'r-', lw=4, color=(162 / 255, 176 / 255, 103 / 255), label='Data(20-23 average)')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize=font)
        ax.set_ylabel('($X$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'DD-fit.png', dpi=300)



        #  alpha
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, alpha, 'r-', lw=4, color=(74 / 255, 138 / 255, 93 / 255), label='alpha--PCM-NN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize=font)
        ax.set_ylabel('(alpha)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'alpha-fit.png', dpi=300)


        ######## ode 求解
        ### 定义函数
        def Par_TH1(t):
            tt = math.floor(t)
            if tt >= 29:
                T1 = T_star[29]
                H1 = H_star[29]
            else:
                T1 = T_star[tt]
                H1 = H_star[tt]
            return T1, H1


        def Par_fun(t):
            T1, H1 = Par_TH1(t)
            T = np.array((T1 - 21) ** 2)
            T = T.reshape([1, 1])
            H = np.array((H1 - 84) ** 2)
            H = H.reshape([1, 1])
            t=np.array(t)
            t=t.reshape([1,1])
            alpha = model.predict_par(t, T, H)
            return alpha


        #
        def ODEs_mean1(X, t):
            alpha = Par_fun(t)
            dXdt = (0.372 - alpha) * X - (0.0008) * (X ** 2) *sf
            return float(dXdt)


        def Par_TH24(t):
            tt = math.floor(t)
            if tt >= 29:
                T1 = T_star24[29]
                H1 = H_star24[29]
            else:
                T1 = T_star24[tt]
                H1 = H_star24[tt]
            return T1, H1


        def Par_fun1(t):
            T1, H1 = Par_TH24(t)
            T = np.array((T1 - 21) ** 2)
            T = T.reshape([1, 1])
            H = np.array((H1 - 84) ** 2)
            H = H.reshape([1, 1])
            t=np.array(t)
            t=t.reshape([1,1])
            alpha = model.predict_par(t, T, H)
            return alpha



        def ODEs_mean24(X, t):
            alpha = Par_fun1(t)
            dXdt = (0.372 - alpha) * X - (0.0008) * (X ** 2) * sf
            return float(dXdt)



        Sol1 = odeint(ODEs_mean1, X0, t_star.flatten())
        X1 = Sol1
        X1 = X1.reshape([len(X1), 1]) / sf

        Sol24 = odeint(ODEs_mean24, X024, t_star.flatten())
        X24 = Sol24
        X24 = X24.reshape([len(X24), 1]) / sf



        #
        np.savetxt(save_results_to + 'X1_ode.txt', X1.reshape((-1, 1)))

        np.savetxt(save_results_to + 'X24_ode.txt', X24.reshape((-1, 1)))


        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, X1, 'k--', marker='o', lw=2, color=(156 / 255, 156 / 255, 156 / 255), markersize=7, label='Odeslover')
        ax.plot(t_star, X_star / sf, 'r-', lw=2, color=(162 / 255, 176 / 255, 103 / 255), label='Data(20-23)')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='plain')
        ax.grid(True)
        ax.set_xlabel('Days', fontsize=font)
        ax.set_ylabel('X', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'X_ode.png', dpi=300)


        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, X24, 'k--', marker='o', lw=2, color=(156 / 255, 156 / 255, 156 / 255), markersize=7, label='Odeslover24')
        ax.plot(t_star, X_star24 / sf, 'r-', lw=2, color=(162 / 255, 176 / 255, 103 / 255), label='Data(20-24 average)')
    #新增的线
        ax.plot(t_star, X_star24only / sf, 'b-', lw=2, color=(74 / 255, 138 / 255, 93 / 255), label='Data(24)')

        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='plain')
        ax.grid(True)
        ax.set_xlabel('Days', fontsize=font)
        ax.set_ylabel('X', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'X24_ode.png', dpi=300)

