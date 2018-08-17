# coding: utf-8

# Importations

import numpy as np
import scipy
from scipy import signal
from sklearn.decomposition import PCA
import pandas as pd
import tables

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

np.random.seed(0)

nb_neurones = 100


# Analyse

def tension_to_index(tension):
    index = np.zeros((8))
    tension = np.round(tension)
    index[int((tension[0]+1)/2+(tension[1]+1)+2*(tension[2]+1))] = 1
    return index

def index_to_tension(index):
    tension = np.zeros((3))
    tension[0] = (index%2)*2-1
    tension[1] = ((index >> 1)%2)*2-1
    tension[2] = ((index >> 2)%2)*2-1
    return tension

def F(state):
    return np.tanh(np.matmul(state,w))

I = np.diag(np.ones(nb_neurones))
def q(state):
    return 1/2.0*np.linalg.norm(F(state)-state)**2

def centerStandDev(x):
    center = np.mean(x,axis=1)
    stand_dev = np.std(x,axis=1)
    return center,stand_dev

def calculateSelectedOutput(selected_neurons):
    s_win = win[:,selected_neurons]
    s_w = w[selected_neurons,:][:,selected_neurons]
    s_wout = wout[selected_neurons,:]
    
    selected_out = []
    prev_state = np.random.rand(len(selected_neurons))
    next_state = np.empty(len(selected_neurons))
    
    for x in x_test_array:
        next_state = np.tanh(np.matmul(x,s_win)+np.matmul(prev_state,s_w))
        selected_out.append(np.tanh(np.matmul(next_state,s_wout)))
        prev_state = next_state

    selected_out = np.array(selected_out)
    return selected_out

def calculateShortOutput(s_w,s_wout,nb_inputs):
    selected_out = []
    prev_state = np.random.rand(nb_neurones)
    next_state = np.empty(nb_neurones)
    
    for x in x_test_array[0:nb_inputs]:
        next_state = np.tanh(np.matmul(x,win)+np.matmul(prev_state,s_w))
        selected_out.append(np.tanh(np.matmul(next_state,s_wout)))
        prev_state = next_state

    selected_out = np.array(selected_out)
    return selected_out

def cov(tau,f,g):
    return np.array([np.average(f*g) if t==0 else np.average(f[:-t]*g[t:]) for t in tau])


class analDataPoint(tables.IsDescription):
    random_seed = tables.Int8Col()
    time_delay = tables.Int8Col()
    min_q = tables.Float64Col((54,nb_neurones))
    q_type = tables.Int64Col(54)
    dimension = tables.Float64Col()
    eigvals_real = tables.Float64Col(nb_neurones)
    eigvals_imag = tables.Float64Col(nb_neurones)
    converges = tables.BoolCol()
    train_output = tables.Float64Col((31,500,3))
    max_cov_val = tables.Float64Col((3,nb_neurones))
    max_cov_delay = tables.Int8Col((3,nb_neurones))
    

output_file = tables.open_file('/opt/DATA/analyse.h5', mode='w', title='Analyse des réseaux de neurones')
anal_table = output_file.create_table('/','data',analDataPoint,'data')
ptr = anal_table.row


# Importations des données

data_file = pd.HDFStore('/opt/DATA/train_test_arrays.h5').root.test_array.read()
x_test_array = data_file['x']
y_test_array = data_file['y']

for random_seed in range(10):
    for time_delay in range(10):
        
        print('seed: '+str(random_seed)+' delay: '+str(time_delay))
        
        ptr['random_seed'] = random_seed
        ptr['time_delay'] = time_delay

        data_file = pd.HDFStore('/opt/DATA/RNN.h5')
        group = data_file.get_node('RNN_seed_'+str(random_seed)+'_delay_'+str(time_delay))

        table = group.loss.read()
        loss = table['loss']
        table = group.training_steps.read()
        w_train_array = table['w']
        wout_train_array = table['wout']

        table = group.testing_results.read()
        states = table['states']
        output = table['output']

        table = group.final_weights.read()
        win = table['win'][0]
        w = table['w'][0]
        wout = table['wout'][0]

        data_file.close()
        
        ptr['converges'] = loss[-1]<0.02

        pca = PCA(n_components=3)
        pca.fit(states)
        x, y, z = pca.transform(states).T

        out_type = np.array([np.argmax(tension_to_index(out)) for out in output])
        center_array = []
        stand_dev_array = []
        for i in range(8):
            indexes = np.where(np.equal(out_type,i*np.ones(len(out_type))))[0]
            center, stand_dev = centerStandDev(np.stack((x[indexes],y[indexes],z[indexes])))
            center_array.append(center)
            stand_dev_array.append(stand_dev)
        center_array = np.array(center_array)
        stand_dev_array = np.array(stand_dev_array)
        
        max_state_val = max(np.sum(abs(states),axis=1))
        
        min_q = []
        for i in range(500):
            opt = scipy.optimize.minimize(q,(np.random.rand(nb_neurones)-0.5)*4*max_state_val,options={'maxiter':300})
            if opt['success'] == True:
                if len(min_q)==0 or not np.any(np.all((min_q-opt['x'])<0.01,axis=1)):
                    min_q.append(opt['x'])
        min_q = np.array(min_q)

        q_type = []
        for q_i in min_q:
            J = np.empty(w.shape)
            for i in range(nb_neurones):
                J[i,:] = np.dot(w[i,:],(1/np.cosh(q_i[i]))**2)-I[i,:]
            q_type.append(sum(np.linalg.eig(J)[0]>0))
        q_type = np.array(q_type,copy=True)
        
        min_q = np.copy(min_q)
        min_q.resize((54,nb_neurones))
        q_type.resize(54)
        ptr['min_q'] = min_q
        ptr['q_type'] = q_type

        pca_all = PCA()
        pca_all.fit(states)
        cov_eigvals = pca_all.explained_variance_

        ptr['dimension'] = np.sum(cov_eigvals)**2/np.sum(cov_eigvals**2)

        x_sp, y_sp, z_sp = pca.transform(min_q).T

        eigvals, eigvecs = np.linalg.eig(w)
        eigvals_train = []

        for i in range(w_train_array.shape[0]):
            eigvals_i, eigvecs_i = np.linalg.eig(w_train_array[i,:,:])
            eigvals_train.append(eigvals_i)
        eigvals_train = np.array(eigvals_train)

        eigvals, eigvecs = np.linalg.eig(w)
        
        ptr['eigvals_real'] = eigvals.real
        ptr['eigvals_imag'] = eigvals.imag
        
        train_output = np.zeros((31,500,3))
        i = 0
        for w_train,wout_train in zip(w_train_array,wout_train_array):
            train_output[i,:,:] = calculateShortOutput(w_train,wout_train,500)
            i += 1
        train_output[i,:,:] = calculateShortOutput(w,wout,500)
        
        ptr['train_output'] = train_output
        
        
        max_cov_val = np.empty((3,nb_neurones))
        max_cov_delay = np.empty((3,nb_neurones))
        
        for j in range(3):
            for i,s in enumerate(states.T):
                cov = signal.correlate(s,y_test_array[:,j],mode='full')/(s.std()*y_test_array[:,j].std()*s.size)
                max_cov_val[j,i] = cov[np.argmax(abs(cov))]
                max_cov_delay[j,i] = np.argmax(abs(cov))-(cov.size//2)

        
        ptr['max_cov_val'] = max_cov_val
        ptr['max_cov_delay'] = max_cov_delay
        
        
        ptr.append()

output_file.close()
