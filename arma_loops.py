import pandas as pd
import numpy as np


#############################################
# Mean for log of data is very close to zero, the mean will not be subtracted
#######################################


#============================================================
#ARMA20
#===========================================================

def arma20(y_train, y_test, theta_new20, log):
    arma_20_p = [np.nan]
    if log == True:
        for i in range(1,len(y_train)):
            if i==1:
                arma_20_p.append( np.exp(-y_train['1st_dif_close'][i]*theta_new20[0][0] + np.log(y_train['close'][i-1])) )
            else:
                arma_20_p.append( np.exp(-y_train['1st_dif_close'][i]*theta_new20[0][0] - (theta_new20[1][0] * y_train['1st_dif_close'][i-1]) + np.log(y_train['close'][i-1])) )
    else:
        for i in range(1,len(y_train)):
            if i==1:
                arma_20_p.append( -y_train['1st_dif_close'][i]*theta_new20[0][0] + y_train['close'][i-1] )
            else:
                arma_20_p.append( -y_train['1st_dif_close'][i]*theta_new20[0][0] - (theta_new20[1][0] * y_train['1st_dif_close'][i-1]) +y_train['close'][i-1] )

    arma_20_p_t = pd.Series(arma_20_p[2:], index=y_train.index.to_list()[1:-1])

    arma_20_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arma_20_f.append( (-y_train['1st_dif_close'][-1]*theta_new20[0][0] - theta_new20[1][0]*y_train['1st_dif_close'][-2]) )
        elif h==1:
            arma_20_f.append(-arma_20_f[h-1]*theta_new20[0][0] - theta_new20[1][0]*y_train['1st_dif_close'][-1])
        else:
            arma_20_f.append(-arma_20_f[h-1]*theta_new20[0][0] -theta_new20[1][0]*arma_20_f[h-2] )

    arma_20_f_t = []
    if log == True:
        for j in range(0,len(arma_20_f)):
            if j == 0:
                arma_20_f_t.append( np.exp(arma_20_f[j] + log(y_train['close'][-1])) )
            else:
                arma_20_f_t.append( np.exp(arma_20_f[j]+sum(arma_20_f[0:j]) + log(y_train['close'][-1])) )
    else:
        for j in range(0,len(arma_20_f)):
            if j == 0:
                arma_20_f_t.append( arma_20_f[j] + y_train['close'][-1] )
            else:
                arma_20_f_t.append( (arma_20_f[j]+sum(arma_20_f[0:j])) + y_train['close'][-1] )

    arma_20_f_t = pd.Series(arma_20_f_t, index = y_test.index.to_list())
    return arma_20_p, arma_20_p_t, arma_20_f, arma_20_f_t

#============================================================
#ARMA21
#===========================================================

def arma21(y_train, y_test, theta_new21, log):
    arma_21_p = [np.nan]
    
    for i in range(1,len(y_train)):
        if i==1:
            arma_21_p.append( -y_train['1st_dif_close'][i]*theta_new21[0][0] + (theta_new21[2][0] * y_train['1st_dif_close'][i]))
        else:
            arma_21_p.append( (-y_train['1st_dif_close'][i]*theta_new21[0][0]) - (theta_new21[1][0] * y_train['1st_dif_close'][i-1]) + (theta_new21[2][0] * (y_train['1st_dif_close'][i] - arma_21_p[i-1])) )


    arma_21_p_t = [np.nan]
    if log == True:
        for i in range(1,len(arma_21_p)):
            arma_21_p_t.append( np.exp(arma_21_p[i] + np.log(y_train['close'][i-1])) )
    else:
        for i in range(1,len(arma_21_p)):
            arma_21_p_t.append(arma_21_p[i] + y_train['close'][i-1])
    
    arma_21_p_t = pd.Series(arma_21_p_t[2:], index=y_train.index.to_list()[1:-1])

    arma_21_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arma_21_f.append( (-y_train['1st_dif_close'][-1]*theta_new21[0][0] - theta_new21[1][0]*y_train['1st_dif_close'][-2]) + theta_new21[2][0]*(y_train['1st_dif_close'][-1] - arma_21_p[-2]))
        elif h==1:
            arma_21_f.append(-arma_21_f[h-1]*theta_new21[0][0] - theta_new21[1][0]*y_train['1st_dif_close'][-1])
        else:
            arma_21_f.append(-arma_21_f[h-1]*theta_new21[0][0] -theta_new21[1][0]*arma_21_f[h-2] )

    arma_21_f_t = []
    if log == True:
        for j in range(0,len(arma_21_f)):
            if j == 0:
                arma_21_f_t.append( np.exp(arma_21_f[j] + np.log(y_train['close'][-1])) )
            else:
                arma_21_f_t.append( np.exp(arma_21_f[j]+sum(arma_21_f[0:j]) + np.log(y_train['close'][-1])) )
    else:
        for j in range(0,len(arma_21_f)):
            if j == 0:
                arma_21_f_t.append( arma_21_f[j] + y_train['close'][-1] )
            else:
                arma_21_f_t.append( (arma_21_f[j]+sum(arma_21_f[0:j])) + y_train['close'][-1] )

    arma_21_f_t = pd.Series(arma_21_f_t, index = y_test.index.to_list())

    return arma_21_p, arma_21_p_t, arma_21_f, arma_21_f_t

#============================================================
#ARMA22
#===========================================================

def arma22(y_train, y_test, theta_new22, log):
    arma_22_p = [np.nan]
    for i in range(1,len(y_train)):
        if i==1:
            arma_22_p.append(-y_train['1st_dif_close'][i]*theta_new22[0][0] + theta_new22[2][0]* y_train['1st_dif_close'][i])
        elif i==2:
            arma_22_p.append(-y_train['1st_dif_close'][i]*theta_new22[0][0] - theta_new22[1][0]* y_train['1st_dif_close'][i-1] + theta_new22[2][0]*(y_train['1st_dif_close'][i] - arma_22_p[i - 1] ) + theta_new22[3][0]*(y_train['1st_dif_close'][i - 1]))
        else:
            arma_22_p.append( -y_train['1st_dif_close'][i]*theta_new22[0][0] - theta_new22[1][0]* y_train['1st_dif_close'][i-1] + theta_new22[2][0]*(y_train['1st_dif_close'][i] - arma_22_p[i - 1] ) + theta_new22[3][0]*(y_train['1st_dif_close'][i - 1] - arma_22_p[i-2]))


    arma_22_p_t = [np.nan]
    if log == True:
        for i in range(1,len(arma_22_p)):
            arma_22_p_t.append( np.exp(arma_22_p[i] + np.log(y_train['close'][i-1])) )
    else:
        for i in range(1,len(arma_22_p)):
            arma_22_p_t.append(arma_22_p[i] + y_train['close'][i-1])
    arma_22_p_t = pd.Series(arma_22_p_t[2:], index=y_train.index.to_list()[1:-1])

    arma_22_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arma_22_f.append(-y_train['1st_dif_close'][-1]*theta_new22[0] - theta_new22[1][0]*y_train['1st_dif_close'][-2] + theta_new22[2][0]*(y_train['1st_dif_close'][-1] - arma_22_p[-2]) + theta_new22[3][0]*(y_train['1st_dif_close'][-2]-arma_22_p[-3]))
        elif h==1:
            arma_22_f.append(-arma_22_f[h-1]*theta_new22[0][0] - theta_new22[1][0]*y_train['1st_dif_close'][-1] + theta_new22[3][0]*(y_train['1st_dif_close'][-1] - arma_22_p[-2]))
        else:
            arma_22_f.append(-arma_22_f[h-1]*theta_new22[0][0] -theta_new22[1][0]*arma_22_f[h-2] )

    arma_22_f_t = []
    if log == True:
        for j in range(0,len(arma_22_f)):
            if j == 0:
                arma_22_f_t.append( np.exp(arma_22_f[j] + np.log(y_train['close'][-1])) )
            else:
                arma_22_f_t.append( np.exp(arma_22_f[j]+sum(arma_22_f[0:j]) + np.log(y_train['close'][-1])) )
    else:
        for j in range(0,len(arma_22_f)):
            if j == 0:
                arma_22_f_t.append( arma_22_f[j] + y_train['close'][-1] )
            else:
                arma_22_f_t.append( (arma_22_f[j]+sum(arma_22_f[0:j])) + y_train['close'][-1] )

    arma_22_f_t = pd.Series(arma_22_f_t, index = y_test.index.to_list())

    return arma_22_p, arma_22_p_t, arma_22_f, arma_22_f_t

#============================================================
#ARMA12 0 12
#===========================================================

def arma12(y_train, y_test, theta_new12, log):
    arma_12_p = [np.nan]
    for i in range(1,len(y_train)):
        if i==1:
            arma_12_p.append(-y_train['1st_dif_close'][i]*theta_new12[0][0] 
                             + theta_new12[1][0]* y_train['1st_dif_close'][i])
        elif i==2:
            arma_12_p.append(-y_train['1st_dif_close'][i]*theta_new12[0][0] 
                             + theta_new12[1][0]*(y_train['1st_dif_close'][i] - arma_12_p[i - 1] ) 
                             + theta_new12[2][0]*(y_train['1st_dif_close'][i - 1]))
        elif i >= 3:
            arma_12_p.append( -y_train['1st_dif_close'][i]*theta_new12[0][0] 
                              + theta_new12[1][0]*( y_train['1st_dif_close'][i] - arma_12_p[i - 1] ) 
                              + theta_new12[2][0]*( y_train['1st_dif_close'][i - 1] - arma_12_p[i - 2]) )

    arma_12_p_t = [np.nan]
    if log == True:
        for i in range(1,len(arma_12_p)):
            arma_12_p_t.append( np.exp(arma_12_p[i] + np.log(y_train['close'][i-1])) )
    else:
        for i in range(1,len(arma_12_p)):
            arma_12_p_t.append(arma_12_p[i] + y_train['close'][i-1])
    arma_12_p_t = pd.Series(arma_12_p_t[2:], index=y_train.index.to_list()[1:-1])

    ##################  forecast ##################

    arma_12_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arma_12_f.append(-y_train['1st_dif_close'][-1]*theta_new12[0][0]
                             + theta_new12[1][0]*(y_train['1st_dif_close'][-1] - arma_12_p[-2]) 
                             + theta_new12[2][0]*(y_train['1st_dif_close'][-2]-arma_12_p[-3]))
        elif h==1:
            arma_12_f.append(-arma_12_f[h-1]*theta_new12[0][0]
                             + theta_new12[2][0]*(y_train['1st_dif_close'][-1] - arma_12_p[-2]))
        else:
            arma_12_f.append(-arma_12_f[h-1]*theta_new12[0][0] )
    
    arma_12_f_t = []
    if log == True:
        for j in range(0,len(arma_12_f)):
            if j == 0:
                arma_12_f_t.append( np.exp( arma_12_f[j] + np.log(y_train['close'][-1]) ) )
            else:
                arma_12_f_t.append( np.exp(arma_12_f[j]+sum(arma_12_f[0:j]) + np.log(y_train['close'][-1])) )
    else:
        for j in range(0,len(arma_12_f)):
            if j == 0:
                arma_12_f_t.append( arma_12_f[j] + y_train['close'][-1] )
            else:
                arma_12_f_t.append( (arma_12_f[j]+sum(arma_12_f[0:j])) + y_train['close'][-1] )

    arma_12_f_t = pd.Series(arma_12_f_t, index = y_test.index.to_list())

    return arma_12_p, arma_12_p_t, arma_12_f, arma_12_f_t
