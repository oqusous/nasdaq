import pandas as pd
import numpy as np

#################################################################
# ARIMA(3,1,4) 012 3456
#################################################################

def arima314(y_train, y_test,theta_new314,log):
    arIma_314_p = []
    for i in range(0,len(y_train)):
        if i == 0: 
            arIma_314_p.append(- y_train['close'][i]*(theta_new314[0][0] - 1) 
                            + theta_new314[3][0] * y_train['close'][i]
                            )

        elif i == 1:
            arIma_314_p.append(- y_train['close'][i]*(theta_new314[0][0] - 1) 
                            - (theta_new314[1][0] - theta_new314[0][0]) * y_train['close'][i - 1]
                            + theta_new314[3][0] *(y_train['close'][i] - arIma_314_p[i - 1] )
                            + theta_new314[4][0] *(y_train['close'][i - 1])
                            )

        elif i == 2: 
            arIma_314_p.append(- y_train['close'][i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_train['close'][i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_train['close'][i-2]
                            + theta_new314[3][0] * (y_train['close'][i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] * (y_train['close'][i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] * (y_train['close'][i - 2])
                            )
        elif i == 3 : 
            arIma_314_p.append(- y_train['close'][i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_train['close'][i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_train['close'][i-2]
                            + theta_new314[2][0] * y_train['close'][i-3]
                            + theta_new314[3][0] *(y_train['close'][i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] *(y_train['close'][i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] *(y_train['close'][i - 2]- arIma_314_p[i - 3])
                            + theta_new314[6][0] *(y_train['close'][i - 3])
                            )
        elif i >= 4 : 
            arIma_314_p.append(- y_train['close'][i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_train['close'][i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_train['close'][i-2]
                            + theta_new314[2][0] * y_train['close'][i-3]
                            + theta_new314[3][0] *(y_train['close'][i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] *(y_train['close'][i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] *(y_train['close'][i - 2]- arIma_314_p[i - 3])
                            + theta_new314[6][0] *(y_train['close'][i - 3] - arIma_314_p[i - 4])
                            )
                        
    arIma_314_p_t = []
    if log == True:
        for i in range(0,len(arIma_314_p)):
            arIma_314_p_t.append(np.exp(arIma_314_p[i]))
        arIma_314_p_t = pd.Series(arIma_314_p_t[:-1], index=y_train[1:].index.to_list())
    else:
        arIma_314_p_t = pd.Series(arIma_314_p[:-1], index=y_train[1:].index.to_list())

    arIma_314_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arIma_314_f.append(- y_train['close'][-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_train['close'][-2]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_train['close'][-3]
                            + theta_new314[2][0] * y_train['close'][-4]
                            + theta_new314[3][0]*(y_train['close'][-1] - arIma_314_p[-2])
                            + theta_new314[4][0]*(y_train['close'][-2] - arIma_314_p[-3])
                            + theta_new314[5][0]*(y_train['close'][-3] - arIma_314_p[-4])
                            + theta_new314[6][0]*(y_train['close'][-4] - arIma_314_p[-5])
                            )
        elif h==1:
            arIma_314_f.append(- arIma_314_f[h-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_train['close'][-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_train['close'][-2]
                            + theta_new314[2][0] * y_train['close'][-3]
                            + theta_new314[4][0]*(y_train['close'][-1] - arIma_314_p[-2])
                            + theta_new314[5][0]*(y_train['close'][-2] - arIma_314_p[-3])
                            + theta_new314[6][0]*(y_train['close'][-3] - arIma_314_p[-4])              
                            )
        elif h==2:
            arIma_314_f.append(- arIma_314_f[h-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * y_train['close'][-1]
                            + theta_new314[2][0] * y_train['close'][-2]
                            + theta_new314[1][0]*y_train['close'][-1]
                            + theta_new314[5][0]*(y_train['close'][-1] - arIma_314_p[-2])
                            + theta_new314[6][0]*(y_train['close'][-2] - arIma_314_p[-3])
                            )
        elif h==3:
            arIma_314_f.append(-arIma_314_f[h-1]*(-1+theta_new314[0][0])
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * arIma_314_f[h-3]
                            + theta_new314[2][0] * y_train['close'][-1]
                            + theta_new314[6][0]*(y_train['close'][-1] - arIma_314_p[-2])
                            )
        elif h>=4:
            arIma_314_f.append(-arIma_314_f[h-1]*(-1+theta_new314[0][0])
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * arIma_314_f[h-3]
                            + theta_new314[2][0] * arIma_314_f[h-4]
                            )
    arIma_314_f_t =[]
    if log == True:
        arIma_314_f_t = pd.Series( np.exp(arIma_314_f_t), index = y_test.index.to_list())
    else:
        arIma_314_f_t = pd.Series(arIma_314_f, index = y_test.index.to_list())

    return arIma_314_p, arIma_314_p_t, arIma_314_f, arIma_314_f_t



#################################################################
# ARIMA(3,1,4) Mean of train is subtracted and added again
#################################################################

def arima314x(y_train, y_test,theta_new314,log):
    y_mean = y_train['close'] - np.mean(y_train['close'])
    arIma_314_p = []
    for i in range(0,len(y_train)):
        if i == 0: 
            arIma_314_p.append(-y_mean[i]*(theta_new314[0][0] - 1) 
                            + theta_new314[3][0] * y_mean[i]
                            )

        elif i == 1:
            arIma_314_p.append(- y_mean[i]*(theta_new314[0][0] - 1) 
                            - (theta_new314[1][0] - theta_new314[0][0]) * y_mean[i - 1]
                            + theta_new314[3][0] *(y_mean[i] - arIma_314_p[i - 1] )
                            + theta_new314[4][0] *(y_mean[i - 1])
                            )

        elif i == 2: 
            arIma_314_p.append(- y_mean[i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_mean[i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_mean[i-2]
                            + theta_new314[3][0] * (y_mean[i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] * (y_mean[i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] * (y_mean[i - 2])
                            )
        elif i == 3 : 
            arIma_314_p.append(- y_mean[i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_mean[i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_mean[i-2]
                            + theta_new314[2][0] * y_mean[i-3]
                            + theta_new314[3][0] *(y_mean[i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] *(y_mean[i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] *(y_mean[i - 2]- arIma_314_p[i - 3])
                            + theta_new314[6][0] *(y_mean[i - 3])
                            )
        elif i >= 4 : 
            arIma_314_p.append(- y_mean[i]*(theta_new314[0][0] - 1)  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_mean[i-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_mean[i-2]
                            + theta_new314[2][0] * y_mean[i-3]
                            + theta_new314[3][0] *(y_mean[i] - arIma_314_p[i - 1] ) 
                            + theta_new314[4][0] *(y_mean[i - 1] - arIma_314_p[i - 2])
                            + theta_new314[5][0] *(y_mean[i - 2]- arIma_314_p[i - 3])
                            + theta_new314[6][0] *(y_mean[i - 3] - arIma_314_p[i - 4])
                            )
                        
    arIma_314_p_t = []
    if log == True:
        for i in range(0,len(arIma_314_p)):
            arIma_314_p_t.append(np.exp(arIma_314_p[i]))
        arIma_314_p_t = pd.Series(arIma_314_p_t[:-1], index=y_train[1:].index.to_list())
    else:
        arIma_314_p_t = pd.Series(arIma_314_p[:-1], index=y_train[1:].index.to_list())+np.mean(y_train['close'])

    arIma_314_f = []
    for h in range(0,len(y_test)):
        if h==0:
            arIma_314_f.append(- y_mean[-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_mean[-2]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_mean[-3]
                            + theta_new314[2][0] * y_mean[-4]
                            + theta_new314[3][0]*(y_mean[-1] - arIma_314_p[-2])
                            + theta_new314[4][0]*(y_mean[-2] - arIma_314_p[-3])
                            + theta_new314[5][0]*(y_mean[-3] - arIma_314_p[-4])
                            + theta_new314[6][0]*(y_mean[-4] - arIma_314_p[-5])
                            )
        elif h==1:
            arIma_314_f.append(- arIma_314_f[h-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] -theta_new314[0][0]) * y_mean[-1]
                            - (theta_new314[2][0] -theta_new314[1][0]) * y_mean[-2]
                            + theta_new314[2][0] * y_mean[-3]
                            + theta_new314[4][0]*(y_mean[-1] - arIma_314_p[-2])
                            + theta_new314[5][0]*(y_mean[-2] - arIma_314_p[-3])
                            + theta_new314[6][0]*(y_mean[-3] - arIma_314_p[-4])              
                            )
        elif h==2:
            arIma_314_f.append(- arIma_314_f[h-1]*(-1+theta_new314[0][0])  
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * y_mean[-1]
                            + theta_new314[2][0] * y_mean[-2]
                            + theta_new314[1][0]*y_mean[-1]
                            + theta_new314[5][0]*(y_mean[-1] - arIma_314_p[-2])
                            + theta_new314[6][0]*(y_mean[-2] - arIma_314_p[-3])
                            )
        elif h==3:
            arIma_314_f.append(-arIma_314_f[h-1]*(-1+theta_new314[0][0])
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * arIma_314_f[h-3]
                            + theta_new314[2][0] * y_mean[-1]
                            + theta_new314[6][0]*(y_mean[-1] - arIma_314_p[-2])
                            )
        elif h>=4:
            arIma_314_f.append(-arIma_314_f[h-1]*(-1+theta_new314[0][0])
                            - (theta_new314[1][0] - theta_new314[0][0]) * arIma_314_f[h-2]
                            - (theta_new314[2][0] - theta_new314[1][0]) * arIma_314_f[h-3]
                            + theta_new314[2][0] * arIma_314_f[h-4]
                            )
    arIma_314_f_t =[]
    if log == True:
        arIma_314_f_t = pd.Series( np.exp(arIma_314_f_t), index = y_test.index.to_list())
    else:
        arIma_314_f_t = pd.Series(arIma_314_f, index = y_test.index.to_list())+np.mean(y_train['close'])

    return arIma_314_p, arIma_314_p_t, arIma_314_f, arIma_314_f_t