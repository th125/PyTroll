# SN IIb progenitor reader

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as si
import sklearn
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# Sandbox switch('False' for actual data processing)
sandbox_switch = 'False'

# Import data
cwd = os.getcwd()
initmass = []
data_eachmass = []
for i in range(10,50):
  if len(glob.glob(cwd+'/'+str(i))) == 1:
    initmass.append(i)
if os.path.exists("HRplot_ledoux.data"):
  os.remove("HRplot_ledoux.data")
w = open("HRplot_ledoux.data", 'w')
w.write("M_init H_env_init H_env_fin M_fin R_fin L_fin Teff_fin g sL_fin He_core C_core mH mHe Si_core Fe_core\n")
for i in initmass:
  f = open(cwd+'/'+str(i)+"/Minit_"+str(i)+"Msun_Ledoux.dat", 'r')
  w = open("HRplot_ledoux.data", 'a')
  fline = f.readlines()
  for idx, val in enumerate(fline):
    if idx > 0:
      data_eachmass.append(val.split())
      w.write(str(val.split())+'\n')
w.close()
r = open("HRplot_ledoux.data", 'r')
data = r.read().replace("[","").replace("]","").replace(",","").replace("'","")
w = open("HRplot_ledoux.data", 'w')
w.write(data)
w.close()
r = open("HRplot_ledoux.data", 'r')
data = []
for line in r.readlines():
  line = line.replace("[","").replace("]","").replace(",","").replace("'","")
  data.append(line.strip().split())
M_init_list = []
M_fin_list = []
M_fin_log_list = []
H_env_fin_list = []
H_env_fin_log_list = []
H_env_round_list = []
R_fin_list = []
L_fin_list = []
Teff_list = []
g_list = []
sL_fin_list = []
C_core_list = []
for i, idx in enumerate(data):
  if i != 0 and float(data[i][2]) != -500 and float(data[i][13]) > 0 and float(data[i][2]) > 0:# and float(data[i][2]) < 1. and np.log10(float(data[i][5])) < 5.7:
    M_init_list.append(float(data[i][0]))
    H_env_fin_list.append(float(data[i][2]))
    H_env_fin_log_list.append(np.log10(float(data[i][2])))
    H_env_round_list.append(round(float(data[i][2]), 0))
    M_fin_list.append(float(data[i][3]))
    M_fin_log_list.append(np.log10(float(data[i][3])))
    R_fin_list.append(np.log10(float(data[i][4])))
    L_fin_list.append(np.log10(float(data[i][5])))
    Teff_list.append(np.log10(float(data[i][6])))
    g_list.append(np.log10(float(data[i][7])))
    sL_fin_list.append(np.log10(float(data[i][8])))
    C_core_list.append(np.log10(float(data[i][10])))
#X, Y = np.meshgrid(Teff_list, L_fin_list)
#Z = (X**2+Y**2)*0

# Grid setting
XYZ = [[0 for x in range(3)] for y in range(len(M_init_list))]
for i, idx in enumerate(M_init_list):
  XYZ[i][0] = Teff_list[i]
  XYZ[i][1] = L_fin_list[i]
  XYZ[i][2] = M_init_list[i]
X1 = Teff_list
Y1 = L_fin_list
Z1 = H_env_round_list
W1 = M_init_list
X2 = Teff_list
Y2 = sL_fin_list
Z2 = H_env_round_list
W2 = M_init_list
XY1 = [[0 for x in range(2)] for y in range(len(H_env_fin_list))]
for i, idx in enumerate(H_env_fin_list):
  XY1[i][0] = Teff_list[i]
  XY1[i][1] = L_fin_list[i]
XY2 = [[0 for x in range(2)] for y in range(len(H_env_fin_list))]
for i, idx in enumerate(H_env_fin_list):
  XY2[i][0] = Teff_list[i]
  XY2[i][1] = sL_fin_list[i]
print(len(XY1))
print(len(Z1))
#ti1 = np.linspace(min(X1), max(X1), len(X1))
#ti2 = np.linspace(min(Y1), max(Y1), len(X1))
#rbf = si.Rbf(X1,Y1,Z1, epsilon = 0.3, function = 'gaussian')
#Zi1 = rbf(Xi1, Yi1)

# Select decision boundary plotting method: 1
if sandbox_switch != 'True':
  prc1 = Perceptron(tol = 1e-3, penalty = 'elasticnet')
  prc1.fit(XY1,Z1)
  tree1 = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy')
  tree1.fit(XY1,Z1)
  mlp1 = MLPClassifier(hidden_layer_sizes = [777, 777, 77], activation = 'relu', alpha = 0.005, solver = 'lbfgs', learning_rate = 'adaptive', max_iter = 1000)
  mlp1.fit(XY1,Z1)
  svc1 = SVC(gamma = 7, decision_function_shape = 'ovr')
  svc1.fit(XY1, Z1)
  knn1 = KNeighborsClassifier(n_neighbors = 4)
  knn1.fit(XY1, Z1)
  vot1 = VotingClassifier(estimators = [('prc1', prc1), ('tree1', tree1), ('mlp1', mlp1), ('svc1', svc1), ('knn1', knn1)], voting = 'hard', weights = [0, 2, 3, 2, 1])
  vot1.fit(XY1, Z1)
  XX1, YY1 = np.meshgrid(np.arange(min(X1)-0.3, max(X1)+0.3, 0.002), np.arange(min(Y1)-0.3, max(Y1)+0.3, 0.002))
  ZZ1 = mlp1.predict(np.c_[XX1.ravel(), YY1.ravel()])
  ZZ1 = ZZ1.reshape(XX1.shape)

# Select decision boundary plotting method: 2
  prc2 = Perceptron(tol = 1e-3, penalty = 'elasticnet')
  prc2.fit(XY2,Z2)
  tree2 = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy')
  tree2.fit(XY2,Z2)
  mlp2 = MLPClassifier(hidden_layer_sizes = [777, 777, 77], activation = 'relu', alpha = 0.005, solver = 'lbfgs', learning_rate = 'adaptive', max_iter = 1000)
  mlp2.fit(XY2,Z2)
  svc2 = SVC(gamma = 7, decision_function_shape = 'ovr')
  svc2.fit(XY2, Z2)
  knn2 = KNeighborsClassifier(n_neighbors = 4)
  knn2.fit(XY2, Z2)
  vot2 = VotingClassifier(estimators = [('prc2', prc2), ('tree2', tree2), ('mlp2', mlp2), ('svc2', svc2), ('knn2', knn2)], voting = 'hard', weights = [0, 2, 3, 2, 1])
  vot2.fit(XY2, Z2)
  XX2, YY2 = np.meshgrid(np.arange(min(X2)-0.3, max(X2)+0.3, 0.002), np.arange(min(Y2)-0.3, max(Y2)+0.3, 0.002))
  ZZ2 = mlp2.predict(np.c_[XX2.ravel(), YY2.ravel()])
  ZZ2 = ZZ2.reshape(XX2.shape)

# Select decision boundary - sandbox playground
if sandbox_switch == 'True':
  XY3 = [[0 for x in range(2)] for y in range(len(M_init_list))]
  for i, idx in enumerate(M_init_list):
    XY3[i][0] = H_env_fin_list[i]
    XY3[i][1] = C_core_list[i]
  X3 = H_env_fin_list
  Y3 = C_core_list
  Z3 = M_init_list
  prc3 = Perceptron(tol = 3e-1, penalty = 'elasticnet')
  prc3.fit(XY3,Z3)
  tree3 = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy')
  tree3.fit(XY3,Z3)
  mlp3 = MLPClassifier(hidden_layer_sizes = [77, 77, 7], activation = 'relu', alpha = 0.001, solver = 'lbfgs', learning_rate = 'adaptive', max_iter = 1000)
  mlp3.fit(XY3,Z3)
  svc3 = SVC(gamma = 7, decision_function_shape = 'ovr')
  svc3.fit(XY3, Z3)
  knn3 = KNeighborsClassifier(n_neighbors = 4)
  knn3.fit(XY3, Z3)
  vot3 = VotingClassifier(estimators = [('prc2', prc2), ('tree2', tree2), ('mlp2', mlp2), ('svc2', svc2), ('knn2', knn2)], voting = 'hard', weights = [0, 2, 3, 2, 1])
  vot3.fit(XY3, Z3)
  XX3, YY3 = np.meshgrid(np.arange(min(X3)-0.3, max(X3)+0.3, 0.002), np.arange(min(Y3)-0.3, max(Y3)+0.3, 0.002))
  ZZ3 = mlp3.predict(np.c_[XX3.ravel(), YY3.ravel()])
  ZZ3 = ZZ3.reshape(XX3.shape)

# Draw figures: HRD
if sandbox_switch != 'True':
  plt.figure(figsize=(18, 16))
  plt.suptitle("Z = 0.02 (solar)", fontsize = 16, y=0.02)
  plt.tight_layout()
  plt.subplot(221)
  plt.tricontourf(X1, Y1, Z1, np.unique(Z1), cmap = matplotlib.cm.gnuplot2)
  plt.tricontour(X1, Y1, Z1, np.unique(Z1), colors = 'k').clabel(np.unique(Z1), fontsize = 10, rightside_up = 'False')
  plt.scatter(X1, Y1, 120, Z1, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.gnuplot2)
  plt.xlim(max(X1)+0.1, min(X1)-0.1)
  plt.ylim(min(Y1)-0.1, max(Y1)+0.1)
  plt.xlabel('Effective temperature (log $T_{eff}$ $(K)$)', Fontsize = 15, labelpad = 7)
  plt.ylabel('Luminosity (log $L/L_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Envelope mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.subplot(222)
  plt.contourf(XX1, YY1, ZZ1, np.unique(Z1), cmap = matplotlib.cm.gnuplot2)
  plt.contour(XX1, YY1, ZZ1, np.unique(Z1), colors = 'k').clabel(np.unique(Z1), fontsize = 10, rightside_up = 'False')
  plt.scatter(X1, Y1, 120, Z1, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.gnuplot2)
  plt.xlim(max(X1)+0.1, min(X1)-0.1)
  plt.ylim(min(Y1)-0.1, max(Y1)+0.1)
  plt.xlabel('Effective temperature (log $T_{eff}$ $(K)$)', Fontsize = 15, labelpad = 7)
  plt.ylabel('Luminosity (log $L/L_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Envelope mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.subplot(223)
  plt.tricontourf(X2, Y2, Z2, np.unique(Z2), cmap = matplotlib.cm.gnuplot2)
  plt.tricontour(X2, Y2, Z2, np.unique(Z2), colors = 'k').clabel(np.unique(Z2), fontsize = 10, rightside_up = 'False')
  plt.scatter(X2, Y2, 120, Z2, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.gnuplot2)
  plt.xlim(max(X2)+0.1, min(X2)-0.1)
  plt.ylim(min(Y2)-0.1, max(Y2)+0.1)
  plt.xlabel('Effective temperature (log $T_{eff}$ $(K))$', Fontsize = 15, labelpad = 7)
  plt.ylabel('Spectroscopic HR luminosity (log $sL/sL_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Envelope mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.subplot(224)
  plt.contourf(XX2, YY2, ZZ2, np.unique(Z2), cmap = matplotlib.cm.gnuplot2)
  plt.contour(XX2, YY2, ZZ2, np.unique(Z2), colors = 'k').clabel(np.unique(Z2), fontsize = 10, rightside_up = 'False')
  plt.scatter(X2, Y2, 120, Z2, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.gnuplot2)
  plt.xlim(max(X2)+0.1, min(X2)-0.1)
  plt.ylim(min(Y2)-0.1, max(Y2)+0.1)
  plt.xlabel('Effective temperature (log $T_{eff}$ $(K))$', Fontsize = 15, labelpad = 7)
  plt.ylabel('Spectroscopic HR luminosity (log $sL/sL_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Envelope mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.tight_layout()
  plt.savefig("HRplot.png", facecolor = 'snow', edgecolor = 'k', format = 'png')
  plt.show()

# Draw figures: Teff / Rfin, Leff / Mfin, etc.
plt.figure(figsize = (25, 10))
plt.suptitle("Z = 0.02 (solar)", fontsize = 16, y=0.02)
plt.subplot(231)
plt.scatter(H_env_fin_log_list, Teff_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Effective temperature (log $T_{eff}$ $(K)$)', Fontsize = 15, labelpad = 7)
plt.subplot(234)
plt.scatter(H_env_fin_log_list, R_fin_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Final radius (log $R/R_\odot$)', Fontsize = 15, labelpad = 7)
plt.subplot(232)
plt.scatter(H_env_fin_log_list, L_fin_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Final luminosity (log $L/L_\odot$)', Fontsize = 15, labelpad = 7)
plt.subplot(235)
plt.scatter(H_env_fin_log_list, M_fin_log_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Final total mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.subplot(233)
plt.scatter(H_env_fin_log_list, g_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Surface gravity (log g $(cm/s^2)$)', Fontsize = 15, labelpad = 7)
plt.subplot(236)
plt.scatter(H_env_fin_log_list, C_core_list, 30, M_init_list, cmap = matplotlib.cm.jet_r)
plt.grid(color = 'lightgrey', linestyle = '--', linewidth = '0.5')
plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
plt.xlabel('Final H envelope mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.ylabel('Carbon core mass (log $M/M_\odot$)', Fontsize = 15, labelpad = 7)
plt.tight_layout()
plt.savefig("HRplot_LT.png", facecolor = 'snow', edgecolor = 'k', format = 'png')
plt.show()

# Draw figures: Sandbox
if sandbox_switch == 'True':
  plt.figure(figsize=(12, 7))
  plt.subplot(121)
  plt.tricontourf(X3, Y3, Z3, np.unique(Z1), cmap = matplotlib.cm.jet_r)
  plt.tricontour(X3, Y3, Z3, np.unique(Z1), colors = 'k').clabel(np.unique(Z1), fontsize = 10, rightside_up = 'False')
  plt.scatter(X3, Y3, 120, Z3, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.jet_r)
  plt.xlim(max(X3)+0.1, min(X3)-0.1)
  plt.ylim(min(Y3)-0.1, max(Y3)+0.1)
  plt.xlabel('M$_{init}$ ($M_\odot$)', Fontsize = 15, labelpad = 7)
  plt.ylabel('C core mass ($M_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.subplot(122)
  plt.contourf(XX3, YY3, ZZ3, np.unique(Z3), cmap = matplotlib.cm.jet_r)
  plt.contour(XX3, YY3, ZZ3, np.unique(Z3), colors = 'k').clabel(np.unique(Z1), fontsize = 10, rightside_up = 'False')
  plt.scatter(X3, Y3, 120, Z3, '*', linewidth = 1, edgecolors = 'k', cmap = matplotlib.cm.jet_r)
  plt.xlim(max(X3)+0.1, min(X3)-0.1)
  plt.ylim(min(Y3)-0.1, max(Y3)+0.1)
  plt.xlabel('M$_{init}$ ($M_\odot$)', Fontsize = 15, labelpad = 7)
  plt.ylabel('C core mass ($M_\odot$)', Fontsize = 15, labelpad = 7)
  plt.colorbar().set_label('Initial mass ($M_\odot$)', fontsize = 13, labelpad = 15, rotation = 90)
  plt.tight_layout()
  plt.show()

#Zi = si.griddata(, (Xi, Yi), method = 'linear')
#for i, idx in enumerate(M_init_list):
# for j, idx in enumerate(Z[i]):
#  if i == j:
#   Z[i][j] = M_init_list[i]
#   X_scat.append(X[i][i])
#   Y_scat.append(Y[i][i])
#print(Z)
#print(Z[0])
#print(Z[0][0])
#clf = MLPClassifier(hidden_later_sizes = [50, 50, 10], activation = 'relu')
#clf.fit(X, Y)
#fig = plt.figure()
#ax = fig.add_axes([0.07,0.07,0.85,0.85])
#cp = ax.contour(X, Y, Z, levels = np.linspace(Z.reshape(-1,1).min(), Z.reshape(-1,1).max(), 24))
#plt.scatter(X_scat, Y_scat, color='r')
#plt.xlim(5.5,3)
#plt.legend()
#plt.colorbar(cp)
#ax.set_xlabel("Teff")
#ax.set_ylabel("L")
#plt.show()
