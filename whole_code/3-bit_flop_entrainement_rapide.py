# coding: utf-8

# # Importation des librairies et données d'entraînement

# Importations

import tables
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from sklearn.decomposition import PCA


# Défintion des variables d'entraînement

nb_neurones = 100
backprop_len = 10
nb_batches = 300
nb_epochs = 15000

class TrainDataPoint(tables.IsDescription):
    w = tables.Float64Col((nb_neurones,nb_neurones))
    wout = tables.Float64Col((nb_neurones,3))
    
class LossDataPoint(tables.IsDescription):
    loss = tables.Float64Col()

class TestDataPoint(tables.IsDescription):
    states = tables.Float64Col(nb_neurones)
    output = tables.Float64Col(3)

class FinalWeights(tables.IsDescription):
    win = tables.Float64Col((3,nb_neurones))
    w = tables.Float64Col((nb_neurones,nb_neurones))
    wout = tables.Float64Col((nb_neurones,3))


# Chargement des données pour entraînement et test

input_file = tables.open_file('/opt/DATA/train_test_arrays.h5', mode='r')

table = input_file.root.test_array
x_test_array = table.read()['x']
y_test_array = table.read()['y']

train_array_table = input_file.root.train_array

if (backprop_len*nb_batches*nb_epochs)>train_array_table.nrows:
    print('Warning: Number of data requested larger than available data points. Train array will loop on available data.')

indexes = [[(i*nb_epochs*backprop_len+j*backprop_len)%(train_array_table.nrows-backprop_len) for i in range(nb_batches)] for j in range(nb_epochs)]


# # Entraînement du réseau

for time_delay in range(10):
    for random_seed in range(10):
        
        print('/RNN_seed_'+str(random_seed)+'_delay_'+str(time_delay))
        
        # Ouverture du fichier pour sauvegarde des résultats
        
        output_file = tables.open_file('/opt/DATA/RNN.h5', mode='a')
        #output_file = tables.open_file('/opt/DATA/RNN.h5', mode='w', title='RNN training progression, test data and final weights')

        if '/RNN_seed_'+str(random_seed)+'_delay_'+str(time_delay) in output_file:
            output_file.remove_node('/RNN_seed_'+str(random_seed)+'_delay_'+str(time_delay),recursive=True)

        group = output_file.create_group('/', 'RNN_seed_'+str(random_seed)+'_delay_'+str(time_delay), 'Training and testing data for RNN with random seed='+str(random_seed)+' and delay='+str(time_delay))
        group._f_setattr('random_seed',random_seed)
        group._f_setattr('time_delay',time_delay)
        group._f_setattr('nb_neurones',nb_neurones)
        group._f_setattr('backprop_len',backprop_len)
        group._f_setattr('nb_batches',nb_batches)
        group._f_setattr('nb_epochs',nb_epochs)


        # Architecture du réseau

        np.random.seed(random_seed)
        win = tf.constant(np.random.uniform(-np.sqrt(1./3), np.sqrt(1./3), (3, nb_neurones)), dtype=tf.float64)
        w = tf.Variable(np.random.uniform(-np.sqrt(1./nb_neurones), np.sqrt(1./nb_neurones), (nb_neurones, nb_neurones)), dtype=tf.float64)
        wout = tf.Variable(np.random.uniform(-np.sqrt(1./nb_neurones), np.sqrt(1./nb_neurones), (nb_neurones, 3)), dtype=tf.float64)


        # Architecture pour entraînement

        x_train = tf.placeholder(tf.float64, [backprop_len, nb_batches, 3])
        y_train = tf.placeholder(tf.float64, [nb_batches, 3])
        state_train_init = tf.placeholder(tf.float64, [nb_batches, nb_neurones])
        momentum = tf.placeholder(tf.float64)

        x_series = tf.unstack(x_train, axis=0)
        current_state_train = state_train_init
        for current_input in x_series:
            current_input = tf.reshape(current_input, [nb_batches, 3])
            current_state_train = tf.reshape(current_state_train, [nb_batches, nb_neurones])
            next_state_train=tf.tanh(tf.matmul(current_state_train,w)+tf.matmul(current_input,win))
            current_state_train = next_state_train

        pred_y_train = tf.tanh(tf.matmul(next_state_train,wout))


        loss = tf.reduce_sum(tf.square(tf.subtract(y_train,pred_y_train)))/nb_batches/3
        train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)


        # Architecture pour test

        x_test = tf.placeholder(tf.float64, [3,])
        state_test_init = tf.placeholder(tf.float64, [1, nb_neurones])

        current_input = tf.reshape(x_test, [1, 3])
        current_state = tf.reshape(state_test_init, [1, nb_neurones])

        next_state = tf.tanh(tf.matmul(current_state,w)+tf.matmul(current_input,win))
        pred_y_test = tf.tanh(tf.matmul(next_state,wout))


        # Initialisation du réseau

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        # Entraînement du réseau

        train_table = output_file.create_table(group, 'training_steps', TrainDataPoint, 'progression of RNN weights during training')
        train_ptr = train_table.row

        loss_table = output_file.create_table(group, 'loss', LossDataPoint, 'progression of loss during training')
        loss_ptr = loss_table.row

        rnn_state = np.random.rand(nb_batches, nb_neurones)
        nb_small_loss = 0

        for epoch_index in range(nb_epochs):

            _train = np.array([train_array_table.read(start=i,stop=i+backprop_len) for i in indexes[epoch_index]]).swapaxes(0,1)
            _x_train = _train['x']
            _y_train = _train['y'][backprop_len-time_delay-1,:]

            _state, _loss, _train, _w, _wout = sess.run(
                [current_state_train,loss,train,w,wout],
                feed_dict = {
                    x_train : _x_train,
                    y_train : _y_train,
                    state_train_init : rnn_state
                })

            rnn_state = _state

            last_loss = _loss

            if _loss < 0.02:
                nb_small_loss += 1
            else:
                nb_small_loss = 0

            if epoch_index>5000 and nb_small_loss >= 200:
                break
            
            loss_ptr['loss'] = _loss
            loss_ptr.append()
            loss_table.flush()
            
            if (epoch_index)%500 == 0:
                train_ptr['w'] = _w
                train_ptr['wout'] = _wout
                train_ptr.append()
                train_table.flush()
                print(str(epoch_index))
            

        # Test du réseau

        test_table = output_file.create_table(group, 'testing_results', TestDataPoint, 'results of testing')

        test_ptr = test_table.row

        weights_table = output_file.create_table(group, 'final_weights', FinalWeights, 'final state of RNN')

        weights_ptr = weights_table.row

        rnn_state = np.random.rand(1,nb_neurones)

        for i in range(len(x_test_array)):
            _output, rnn_state, _w, _win, _wout = sess.run(
                    [pred_y_test, next_state, w, win, wout],
                    feed_dict = {
                        x_test : x_test_array[i,:],
                        state_test_init : rnn_state
                    })
            test_ptr['output'] = _output
            test_ptr['states'] = rnn_state
            test_ptr.append()

        weights_ptr['win'] = _win
        weights_ptr['w'] = _w
        weights_ptr['wout'] = _wout
        weights_ptr.append()

        test_table.flush()
        weights_table.flush()

        output_file.close()

# Fermeture du fichier des résultats

input_file.close()