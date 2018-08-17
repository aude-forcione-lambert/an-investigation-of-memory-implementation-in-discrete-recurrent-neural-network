import numpy as np
import scipy
from scipy import signal
from sklearn.decomposition import PCA
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

nb_neurones = 100
colors = ('indianred','darkorange','gold','lime','turquoise','royalblue','blueviolet','magenta')

# Fonctions d'affichage

def plotSphere(ax,c,r,color):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r[0]*np.cos(u)*np.sin(v)+c[0]
    y = r[1]*np.sin(u)*np.sin(v)+c[1]
    z = r[2]*np.cos(v)+c[2]
    ax.plot_wireframe(x, y, z, color=color, alpha=0.5)

def plotTrainResults(loss):
    plt.figure()
    plt.subplot(1,1,1)
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Erreur")
    plt.plot(np.arange(len(loss)),loss)

def plotTestResults(x,y,out,nb):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize = (10,5))

    ax1.plot(np.arange(nb),x[:nb,0],'r',linewidth=0.5)
    ax1.plot(np.arange(time_delay,nb),y[:nb-time_delay,0],'b--',linewidth=1)
    ax1.plot(np.arange(nb),out[:nb,0],'k',linewidth=1)
    ax1.xaxis.grid()
    ax1.set_ylabel("Channel 1")

    ax2.plot(np.arange(nb),x[:nb,1],'r',linewidth=0.5)
    ax2.plot(np.arange(time_delay,nb),y[:nb-time_delay,1],'b--',linewidth=1)
    ax2.plot(np.arange(nb),out[:nb,1],'k',linewidth=1)
    ax2.xaxis.grid()
    ax2.set_ylabel("Channel 2")

    ax3.plot(np.arange(nb),x[:nb,2],'r',linewidth=0.5)
    ax3.plot(np.arange(time_delay,nb),y[:nb-time_delay,2],'b--',linewidth=1)
    ax3.plot(np.arange(nb),out[:nb,2],'k',linewidth=1)
    ax3.xaxis.grid()
    ax3.set_ylabel("Channel 3")

    plt.xlabel("temps (cycles)")

def plotPrincipalComponents(dots,lines,spheres,slowpts,vecs,nb,pc_1,pc_2,pc_3,out_type,min_q=[],q_type=[],vecs_array=[],center_array=[],stand_dev_array=[]):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    if lines:
        ax.plot(pc_1[:nb], pc_2[:nb], pc_3[:nb], '-k', linewidth=0.5, alpha=0.3)
    
    if dots:
        for i in range(nb):
            ax.scatter(pc_1[i], pc_2[i], pc_3[i], c=colors[out_type[i]], alpha=0.3)
    
    if spheres:    
        for i in range(8):
            plotSphere(ax,center_array[i],stand_dev_array[i],'k')
    
    if slowpts:
        for i in range(min_q.shape[1]):
            if(q_type[i]<=3):
                ax.scatter(min_q[0,i], min_q[1,i], min_q[2,i], c=['k','b','g','y'][q_type[i]], marker='x', alpha=0.7)
            else:
                ax.scatter(min_q[0,i], min_q[1,i], min_q[2,i], c='r', marker='^', alpha=0.7)
                ax.text(min_q[0,i], min_q[1,i], min_q[2,i], str(q_type[i]))
    
    if vecs:
        for vec_i in vecs_array:
            ax.plot([0,vec_i[0]],[0,vec_i[1]],[0,vec_i[2]],'k')
            
    custom_legend = [
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[0], marker = '.', label = 'Out '+str(index_to_tension(0))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[1], marker = '.', label = 'Out '+str(index_to_tension(1))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[2], marker = '.', label = 'Out '+str(index_to_tension(2))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[3], marker = '.', label = 'Out '+str(index_to_tension(3))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[4], marker = '.', label = 'Out '+str(index_to_tension(4))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[5], marker = '.', label = 'Out '+str(index_to_tension(5))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[6], marker = '.', label = 'Out '+str(index_to_tension(6))),
        Line2D([0], [0], color = 'w', markersize = 15, markerfacecolor = colors[7], marker = '.', label = 'Out '+str(index_to_tension(7))),
        Line2D([0], [0], color = 'w', markersize = 10, markerfacecolor = 'k', marker = 'X', label = 'Point fixe'),
        Line2D([0], [0], color = 'w', markersize = 10, markerfacecolor = 'b', marker = 'X', label = 'Point de selle (1 dim)'),
        Line2D([0], [0], color = 'w', markersize = 10, markerfacecolor = 'g', marker = 'X', label = 'Point de selle (2 dims)'),
        Line2D([0], [0], color = 'w', markersize = 10, markerfacecolor = 'y', marker = 'X', label = 'Point de selle (3 dims)')
    ]
    ax.legend(handles = custom_legend, loc=9, ncol=4)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

def plotEigvalues(ax,w,w_train):
    
    eigvals, eigvecs = np.linalg.eig(w)
    eigvals_train = []
    
    for i in range(w_train.shape[0]):
        eigvals_i, eigvecs_i = np.linalg.eig(w_train[i,:,:])
        eigvals_train.append(eigvals_i)
    eigvals_train = np.array(eigvals_train)

    ax.set_prop_cycle( cycler( 'color', ((i,i,i) for i in np.linspace(0.9,0,w_train.shape[0])) ) )

    ax.set_xlabel('Valeur réelle')
    ax.set_ylabel('Valeur imaginaire')
    
    ax.plot(eigvals_train.T.real,eigvals_train.T.imag, '.', markersize=2)
    ax.plot(eigvals.real,eigvals.imag, 'rx', markersize=6)

def plotEigvalsMovie(w,w_train_array):
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8,4))
    ax1.set_xlabel('Valeur réelle')
    ax2.set_xlabel('Valeur réelle')
    ax1.set_ylabel('Valeur imaginaire')

    eigvals, eigvecs = np.linalg.eig(w)
    line, = ax1.plot(eigvals.real, eigvals.imag, '.b')
    plotEigvalues(ax2,w,w_train_array)

    def init():
        line.set_xdata([np.nan] * len(eigvals))
        line.set_ydata([np.nan] * len(eigvals))
        return line

    def animate(i):
        eigvals_i, eigvecs_i = np.linalg.eig(w_train_array[i,:,:])
        line.set_xdata(eigvals_i.real)
        line.set_ydata(eigvals_i.imag)
        return line

    ani = animation.FuncAnimation(
        fig, animate, frames = w_train_array.shape[0], interval=500, repeat_delay=5000, blit=False, repeat=True)

def plotNeuronsActivity(states,input_array,output_array,nb=0):
    if nb==0 : nb=states.shape[1]
    fig, ax_array = plt.subplots(states.shape[0]+6,1,sharex=True,figsize=(9,states.shape[0]*0.5))
    fig.subplots_adjust(hspace=0.1)

    dark_colors = ['r','g','b']

    plt.xlabel('temps')

    for i in range(states.shape[0]):
        ax_array[i].plot(np.arange(nb),states[i,:nb],'-k',linewidth=0.5)
        ax_array[i].plot(np.arange(nb),states[i,:nb],'.',markersize=2,color=dark_colors[i%3])
        ax_array[i].get_yaxis().set_ticks([0])
        ax_array[i].grid(True)
        ax_array[i].set_ylim(-np.max(abs(states))*1.1,np.max(abs(states))*1.1)
        ax_array[i].get_xaxis().set_ticks([])
        ax_array[i].set_frame_on(False)
        ax_array[i].set_ylabel(str(i),fontsize=5)

    for i in range(3):
        ax_array[2*i+states.shape[0]].plot(np.arange(nb),output_array[:nb,i],'-k')
        ax_array[2*i+states.shape[0]].get_yaxis().set_ticks([])
        ax_array[2*i+states.shape[0]].set_ylim(-1.1,1.1)
        ax_array[2*i+states.shape[0]].get_xaxis().set_ticks([])
        ax_array[2*i+states.shape[0]].set_frame_on(False)
        ax_array[2*i+states.shape[0]].set_ylabel('out '+str(i+1),fontsize=5)
        
        ax_array[2*i+states.shape[0]+1].plot(np.arange(nb),input_array[:nb,i],'-k')
        ax_array[2*i+states.shape[0]+1].get_yaxis().set_ticks([])
        ax_array[2*i+states.shape[0]+1].set_ylim(-1.1,1.1)
        ax_array[2*i+states.shape[0]+1].get_xaxis().set_ticks([])
        ax_array[2*i+states.shape[0]+1].set_frame_on(False)
        ax_array[2*i+states.shape[0]+1].set_ylabel('in '+str(i+1),fontsize=5)

# Fonctions d'analyse

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

def F(state,w):
    return np.tanh(np.matmul(state,w))

I = np.diag(np.ones(nb_neurones))
def q(state,w):
    return 1/2.0*np.linalg.norm(F(state,w)-state)**2

def centerStandDev(x):
    center = np.mean(x,axis=0)
    stand_dev = np.std(x,axis=0)
    return center,stand_dev

def calculateOutput(s_win,s_w,s_wout):
    selected_out = []
    prev_state = np.random.rand(s_w.shape[0])
    next_state = np.empty(s_w.shape[0])
    
    for x in x_test_array:
        next_state = np.tanh(np.matmul(x,s_win)+np.matmul(prev_state,s_w))
        selected_out.append(np.tanh(np.matmul(next_state,s_wout)))
        prev_state = next_state

    selected_out = np.array(selected_out)
    return selected_out

def calculateStates(s_win,s_w):
    selected_states = []
    prev_state = np.random.rand(s_w.shape[0])
    next_state = np.empty(s_w.shape[0])
    
    for x in x_test_array:
        next_state = np.tanh(np.matmul(x,s_win)+np.matmul(prev_state,s_w))
        selected_states.append(next_state)
        prev_state = next_state

    selected_states = np.array(selected_states)
    return selected_states

def matchPattern(base_array,pattern):
    indexes = []
    for i in range(len(base_array)-len(pattern)):
        if np.all(base_array[i:i+len(pattern)] == pattern):
            indexes.append(i)
    return np.array(indexes)