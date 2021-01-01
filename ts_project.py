import pandas as pd
import numpy as np
import traceback
import itertools
import pickle
from statsmodels.api import tsa
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets 
from matplotlib import gridspec
import warnings
import copy
import seaborn as sns
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from scipy.stats import chi2
import statsmodels.api as sm
from scipy.signal import dlsim
import sys
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

##############################################################
# PLOT AND OTHER BASIC DECLUTTERING FUNCTIONS
##############################################################
def one_plot(y, label, title, xlabel, ylabel, save):
    plt.figure(figsize=(20,10))
    plt.plot(y, label=label)
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=20)
    plt.legend(loc="best", frameon=False, fontsize=15)
    if save ==True:
        plt.savefig('Images/{}.png'.format(title), dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()

def two_plots(y1,y2, label1,label2, title, xlabel, ylabel, save):
    plt.figure(figsize=(20,10))
    plt.plot( y1, label=label1)
    plt.plot( y2, label = label2)
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=25)
    plt.legend(loc="best", frameon=False, fontsize=15)
    if save ==True:
        plt.savefig('Images/{}.png'.format(title).format(title), dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()

def hisogram_plot(y, title, ylabel, xlabel,bins,save):
    plt.figure(figsize=(15,8))
    y.hist(bins=bins)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel( xlabel)
    if save ==True:
        plt.savefig('Images/{}.png'.format(title), dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()

def stl_plot(results, save):
    plt.rc('figure',figsize=(16,12))
    plt.rc('font',size=13)
    results.plot()
    if save == True:
        plt.savefig('Images/tdecomp.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()

####################################################################################
# BASIC STATS
####################################################################################
def f_Tt(Rt, Tt):
    return max( 0, round(1-(np.var(Rt)/(np.var(Rt+Tt))), 4) )
def f_St(Rt, St):
    return max( 0, round( 1-(np.var(Rt)/(np.var(Rt+St))) , 4) )

def adfuller_fun(series):
    result = adfuller(series)
    print('ADF Statistic: {}'.format(round(result[0],4)))
    print('p-value: {}'.format(result[1]))
    for key, value in result[4].items():
        print('Critical Values:')
        print('{}, {}'.format(key, round(value,4)))

def correlation_coefficent_cal(xarray,yarray):
    # print(yarray)
    xbar = np.mean(xarray)
    # print(xbar)
    ybar = np.mean(yarray)
    # print(ybar)
    xt_xbar = xarray - xbar
    # print(xt_xbar)
    yt_ybar = yarray - ybar
    # print(yt_ybar)
    dem = np.sum(xt_xbar*yt_ybar)
    # print(dem)
    num = np.sqrt(np.sum(xt_xbar**2))*(np.sqrt(np.sum(yt_ybar**2)))
    # print(num)
    r = round(dem/num, 5)
    return r


def calc_ACF(y):
    T = len(y)
    tu_hats = []
    yarray = np.array(y)
    ybar = np.mean(y)
    for t in range(T):
        den = (yarray[t:] - ybar) * (yarray[:T-t] - ybar)
        num = (yarray-ybar)**2
        tu_hat = np.sum(den)/np.sum(num)
        tu_hats.append(tu_hat)
    return tu_hats

def box_pierce_q(residual, lags, T):
    acfq = acf(residual, alpha=0.05)[0]
    # print(acf[0:20])
    z = acfq[1:lags+1]
    # print(z)
    # print(lags+1)
    # print(T)
    Q = T*sum(np.power(z,2))
    return Q

####################################################################################
# GPAC
####################################################################################

def gpac(Ry,k,j):
    gpac_table = pd.DataFrame(np.full((j,j), np.nan), index=range(0,j), columns=range(1,j+1))
    # print(gpac_table)
    # first denominator for each k (fdfk)
    fdfk = []
    # full list of denominators for each k
    floNfek = {}
    # full list of numerators for each k
    flodfek = {}
    for ki in range(1,k+1):
        # first column in gpac
        if ki == 1:
            for ji in range(0,j):
                # print(Ry[ki+ji],'/',Ry[ji])
                gpac_table.loc[ji, ki]  = Ry[ki+ji]/Ry[ji]

        # all other columns tbd by determinate/Cramer Rule
        else:
            # Symmetric Ry index construction
            ry_sym_index = []
            for ji in range(0,j):
                for kz in range(0, ki):
                    indicies = list(range(ji - kz, ki - kz))
                    # Absolute is used for Ry symmetric property
                    indicies = np.array( [abs(z) for z in indicies] )
                    if len(indicies) == ki:
                        ry_sym_index.append(indicies)
                    else:
                        break
            fdfk.append(ry_sym_index)
            # print(ry_sym_index)

        # Creating all denominators by increasing first column by an increment of 1 as j increases
        # then subtracting the adjacent columns -1 consecutively
        for i in range(len(fdfk)):
            fdfk[i] = np.array(fdfk[i]).reshape(-1, len(fdfk[i]))
        # print(fdfk)
        k_shape_d = 2
        for ide in range(len(fdfk)):
            flodfek[k_shape_d] = []
            for ji in range(0,j):
                # print(fdfk[ide])
                stack_np = np.zeros(shape=(k_shape_d,k_shape_d), dtype='int32')
                for col in range(k_shape_d):
                    if col == 0:
                        coli = fdfk[ide][:,0]+ji
                        stack_np[:,0] = coli
                    else:
                        coli = stack_np[:,col-1] - np.ones(shape=(1, k_shape_d),dtype='int32' )
                        stack_np[:,col] = coli
                flodfek[k_shape_d].append( abs(stack_np) )
            k_shape_d+=1

        # Creating all numerators by altering the last column in the previously created denominators
        k_shape_n = 2
        floNfek = copy.deepcopy(flodfek)

        for key in floNfek.keys():
            for inu in range(len(floNfek[key])):
                floNfek[key][inu][:, -1] = floNfek[key][inu][:, 0]+1
            k_shape_n+=1
    

    # Now we have matricies with the index of Ry we need to calculated the gpac
    # the loops below will accress these indcies and replace them with Ry[index] instead of just index
    Ry_num = copy.deepcopy(floNfek)
    Ry_den = copy.deepcopy(flodfek)
    Ry_array = np.array(Ry, dtype='float64')

    for keys1 in floNfek.keys():
        for vi in range(len(floNfek[keys1])):
            vi_shape = floNfek[keys1][vi].shape
            Ry_num[keys1][vi] = (Ry_num[keys1][vi]).astype('float64')
            for row_i in range(vi_shape[0]):
                for col_i in range(vi_shape[1]):
                    Ry_value = Ry_array[floNfek[keys1][vi][row_i, col_i]]
                    Ry_num[keys1][vi][row_i, col_i] = Ry_value

    for keys1 in flodfek.keys():
        for vi in range(len(flodfek[keys1])):
            vi_shape = flodfek[keys1][vi].shape
            Ry_den[keys1][vi] = (Ry_den[keys1][vi]).astype('float64')
            for row_i in range(vi_shape[0]):
                for col_i in range(vi_shape[1]):
                    Ry_value = Ry_array[flodfek[keys1][vi][row_i, col_i]]
                    Ry_den[keys1][vi][row_i, col_i] = Ry_value

    # filling the remainder of the table
    for ki in range(2,k+1):
        for ji in range(0,j):
            # print('(',ki,ji,')\n',floNfek[ki][ji],'\n--------\n', flodfek[ki][ji], end='\n\n')
            if np.linalg.det( Ry_den[ki][ji] ) != 0:
                gpac_table.loc[ji, ki]  = np.linalg.det( Ry_num[ki][ji] )/np.linalg.det( Ry_den[ki][ji] )
            else:
                gpac_table.loc[ji, ki] = np.nan


    return gpac_table

def heatmap_gpac(gpac_table,k,j, Bool):
    f, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(gpac_table, cmap="YlGnBu", annot=True, linewidths=.5,ax=ax, fmt='.3f')
    ax.set_ylim(9,-0.5)
    ax.set_title('GPAC ARMA', fontsize = 16)
    ax.set_yticklabels(range(0,j+1), va='center', rotation = 90, position=(0,0.28))
    ax.set_xticklabels(range(1,k+1), ha='center', rotation = 0, position=(0.5,0))
    ax.set_xlabel('k', fontsize=20)
    # ax.set_xticks(range(1,k))
    ax.set_ylabel('j', fontsize=20, rotation=0)
    # ax.set_yticks(range(0,j))
    plt.subplots_adjust(wspace = 1)
    plt.tick_params(axis='both',labelsize=15)
    if Bool == True:
        plt.savefig('Images/gpac.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    plt.show()

####################################################################################
# LM ALGO
####################################################################################

def e_dlsim(y, ar, ma):
    np.random.seed(42)
    num = ar # switched from ma because we are seeking e term not the y term
    den = ma
    system = (num.astype('float64'),den.astype('float64'),1)
    #print('system in dlsim func: ',system)
    _, e_dlsim = dlsim(system, y)
    return e_dlsim.astype('float64')

def a_b_constructor(na, nb, n, theta_int):

    if ((na != 0) and (na == nb or na > nb)):
        a = np.ndarray.flatten(theta_int[0:na].astype('float64')).astype('float64')
    elif (na != 0 and na < nb):
        a =np.concatenate((np.ndarray.flatten(theta_int[0:na].astype('float64')), np.zeros(nb-na).astype('float64')), axis=None).astype('float64')
    elif na == 0:
        a = np.zeros(shape=n).astype('float64')
    
    if ((nb != 0) and (na == nb or nb > na)):
        b = np.ndarray.flatten(theta_int[na:].astype('float64')).astype('float64')

    elif (nb != 0 and nb < na):
        b = np.concatenate((np.ndarray.flatten(theta_int[na:].astype('float64')), np.zeros(na-nb).astype('float64')), axis=None).astype('float64')

    elif nb == 0:
        b = np.zeros(shape=n).astype('float64')

    return a, b 

def LeMaAl(y, na, nb, eps=0.001):
    sse =[]
    y = y.astype('float64')
    mu_max = 10e20
    ite = 0
    # combination coefficent
    max_iters = 100
    delta = 10e-6
    N = len(y)
    n = na+nb

    ## error with zero theta.
    with np.errstate(divide='ignore'):
     np.float64(1.0) / 0.0

    theta_int = np.zeros((na+nb, 1)).astype('float64')
    #print('theta_int\n', theta_int, end='\n\n')
    a0, b0 = a_b_constructor(na, nb, n, theta_int)
    #print('a0=',a0,'b0=',b0, sep='\n', end='\n\n')
    e = e_dlsim(y, np.r_[1, np.array(a0,dtype='float64')], np.r_[1, np.array(b0,dtype='float64')])
    #print('e0\n', e, end='\n\n')

    sse.append(np.dot(e.T.astype('float64'), e.astype('float64')).astype('float64')[0][0])
    while ite < max_iters:
        
        sse_old = np.dot(e.T.astype('float64'), e.astype('float64')).astype('float64')
        #print('sse_old', sse_old, end='\n\n')
        mu = 0.01
        ## H, g, J and SSE must
        # Jacobian
        J = np.zeros(shape=(N,n)).astype('float64')
        i_used_a = 0
        i_used_b = 0
        for i in range(n):
            ai, bi = a_b_constructor(na, nb, n, theta_int)
            #print(ite, i, '\nei system')
            ei = e_dlsim(y, np.r_[1, np.array(ai,dtype='float64')], np.r_[1, np.array(bi,dtype='float64')]).astype('float64')
        
            if i < na: # go through the a1 to ana values 
                aid = ai
                aid[i_used_a] = ai[i_used_a]+delta
                #print(ite, i, '\nei+d system')
                eid = e_dlsim(y.astype('float64'), np.r_[1, np.array(aid,dtype='float64')], np.r_[1, np.array(bi,dtype='float64')]).astype('float64')
                #print('\n')
                ji = ( ei - eid )/delta
                i_used_a+=1

            else: # go through the b1 to bnb values 
                bid = bi
                bid[i_used_b] = bi[i_used_b]+delta
                #print(ite, i, '\nei+d system')
                eid = e_dlsim(y.astype('float64'), np.r_[1, np.array(ai,dtype='float64')], np.r_[1, np.array(bid,dtype='float64')]).astype('float64')
                ji = ( ei - eid )/delta
                #print('\n')
                i_used_b+=1
            
            J[:,[i]] =  ji.astype('float64')
            
            
        #print('J\n', J, end='\n\n')

        # Hessian
        H = np.dot(J.T.astype('float64'), J.astype('float64')).astype('float64')
        #print('H\n', H, end='\n\n')

        # gradient vector
        g= np.dot(J.T.astype('float64'), e.astype('float64')).astype('float64')
        #print('g\n', g)

        I  = np.identity(n).astype('float64')
        HmuI = H.astype('float64')+(mu*I.astype('float64')).astype('float64')
        #print('mu', mu)
        #print('HmuI\n', HmuI, end='\n\n')

        #delta theta
        delta_theta = np.dot(np.linalg.inv(HmuI.astype('float64')), g.astype('float64')).astype('float64')
        #print('delta_theta\n', delta_theta, end='\n\n')
        theta_new = theta_int + delta_theta
        #print('theta_new\n', theta_new, end='\n\n')

        # # New Error using theta_new = theta_int + (HmuI^-1)*g [delta_theta]
        # construct new den and num for dlsim
        a, b = a_b_constructor(na, nb, n, theta_new)
        #print('a=',a,'b=',b, sep='\n', end='\n\n')
        # new error
        e = e_dlsim(y, np.r_[1, np.array(a,dtype='float64')], np.r_[1, np.array(b,dtype='float64')])
        e_mean = np.nanmean(e)
        inds = np.where(np.isnan(e))
        e[inds] = np.take(e_mean, inds[1])
        # print(e, np.r_[1, list(a)], np.r_[1, list(b)])
        sse_new =  np.dot(e.T.astype('float64'), e.astype('float64')).astype('float64')
        #print('sse_new:', sse_new, end='\n\n')
        # Iteration process
        if np.isinf(sse_new):
            sse_new = [[1e10]]
            #print('sse_new: changed to = ', '1e10', end='\n\n')
        elif np.isnan(sse_new):
            sse_new = [[1e10]]
            #print('sse_new: changed to = ', '1e10', end='\n\n')
        # elif sse_new >= 1e100:
        #     sse_new = sse_old -100
        #     print('sse_new: changed to = ', '1e10', end='\n\n')

        

        if sse_new < sse_old:
            #print('\niteration:', ite, '\nsse_new < sse_old\n\n')
            #print(np.linalg.norm(delta_theta,2), eps)
            if np.linalg.norm(delta_theta,2) < eps:
                var_e = np.var(e).astype('float64')
                cov_theta_hat = var_e * np.linalg.inv(H.astype('float64'))
                # print('\nprocess converged, results:', '\ntheta_hat\n', theta_new, '\nvariance of e\n',var_e, '\ncov_theta_hat\n', cov_theta_hat, end='\n\n')
                return theta_new, var_e, cov_theta_hat, sse

            else:
                #print('\niteration:', ite, '\nsse_new < sse_old, but delta_theta > eps\n\n')
                theta_int = theta_new.astype('float64')
                mu = mu/10
                sse.append(sse_new[0][0])
                ite+=1
            
                if ite >= max_iters:
                    #print('\niteration:', ite, '\nMax iterations reached\n\n')
                    sys.exit()
                else:
                    continue

        elif sse_new >= sse_old:
            #print('\niteration:', ite, '\nsse_new >= sse_old\n\n')
            theta_int_fixed = np.array(theta_int, copy=True)
            while sse_new >= sse_old:
                mu = mu*10
                if mu > mu_max:
                    #print('\niteration:', ite, '\nmu value reached max\n\n')
                    sys.exit()
                else:
                    HmuI = H.astype('float64')+(mu*I).astype('float64')
                    #print('mu', mu)
                    #print('HmuI\n', HmuI, end='\n\n')

                    #delta theta
                    delta_theta = np.dot(np.linalg.inv(HmuI.astype('float64')), g.astype('float64')).astype('float64')
                    #print('delta_theta\n', delta_theta, end='\n\n')
                    #print('theta_int\n', theta_int_fixed, end='\n\n')
                    theta_new = theta_int_fixed + delta_theta
                    #print('theta_new\n', theta_new, end='\n\n')

                    # # New Error using theta_new = theta_int + (HmuI^-1)*g [delta_theta]
                    # construct new den and num for dlsim
                    a, b = a_b_constructor(na, nb, n, theta_new)
                    #print('a=',a,'b=',b, sep='\n', end='\n\n')
                    # new error
                    e = e_dlsim(y.astype('float64'), np.r_[1, np.array(a, dtype='float64')], np.r_[1, np.array(b, dtype='float64')])
                    # print(e, np.r_[1, list(a)], np.r_[1, list(b)])
                    e_mean = np.nanmean(e)
                    inds = np.where(np.isnan(e))
                    e[inds] = np.take(e_mean, inds[1])
                    sse_new =  np.dot(e.T.astype('float64'), e.astype('float64')).astype('float64')
                    #print('sse_new:', sse_new, end='\n\n')
                    if np.isinf(sse_new):
                        sse_new = [[1e10]]
                        #print('sse_new: changed to = ', '1e10', end='\n\n')
                    elif np.isnan(sse_new):
                        sse_new = [[1e10]]
                        #print('sse_new: changed to = ', '1e10', end='\n\n')
                    sse.append(sse_new[0][0])

                    ite+=1              
                theta_int = theta_new

def LM_loop(y_train, gs_orders, save, submean, eps):
    t_var_cov_LM = {'Order':[], 'theta_new':[], 'var_e':[], 'cov_theta_hat':[]}
    for order in gs_orders['Order'].to_list():
        ab = [int(x.replace('(', '').replace(')','')) for x in order.split(', ')]
        try:
            if submean== False:
                theta_new, var_e, cov_theta_hat, _ = LeMaAl(y_train['1st_dif_close'].dropna(), ab[0], ab[1],eps)
                t_var_cov_LM['Order'].append(order)
                t_var_cov_LM['theta_new'].append(theta_new)
                t_var_cov_LM['var_e'].append(var_e)
                t_var_cov_LM['cov_theta_hat'].append(cov_theta_hat)
                print(order, 'converged')
            else:
                theta_new, var_e, cov_theta_hat, _ = LeMaAl(y_train['1st_dif_close'].dropna() - np.mean(y_train['1st_dif_close'].dropna()), ab[0], ab[1],eps)
                t_var_cov_LM['Order'].append(order)
                t_var_cov_LM['theta_new'].append(theta_new)
                t_var_cov_LM['var_e'].append(var_e)
                t_var_cov_LM['cov_theta_hat'].append(cov_theta_hat)
                print(order, 'converged')
        except:
            t_var_cov_LM['Order'].append(order)
            t_var_cov_LM['theta_new'].append('LM did not converge')
            t_var_cov_LM['var_e'].append('LM did not converge')
            t_var_cov_LM['cov_theta_hat'].append('LM did not converge')
            print(order, 'did not converge')
    if save == True:
        pickle.dump( t_var_cov_LM, open( "pickle/LM_results.p", "wb" ) )

    return t_var_cov_LM

####################################################################################
# ARIMA ARMA
####################################################################################
def arma_comp_roots_table(gs_arma_table_idxs, gs_arma_table):
    root_comp_table = gs_arma_table.loc[gs_arma_table_idxs, ['Order','ParamRoots']]
    root_comp_table['Ar Roots'] = np.full(fill_value='s',shape=(len(gs_arma_table_idxs)))
    root_comp_table['Ma Roots'] = np.full(fill_value='s',shape=(len(gs_arma_table_idxs)))
    root_comp_table['Ratios'] = np.zeros(shape=(len(gs_arma_table_idxs)))
    for idx, row in root_comp_table.iterrows():
        root_comp_table['Ar Roots'][idx] =  root_comp_table['ParamRoots'][idx][0:int(root_comp_table['Order'][idx][1])]
        root_comp_table['Ma Roots'][idx] =  root_comp_table['ParamRoots'][idx][int(root_comp_table['Order'][idx][1]):]
        ratios = []
        if len(root_comp_table['Ar Roots'][idx]) >= len(root_comp_table['Ma Roots'][idx]):
            for i in range(len(root_comp_table['Ma Roots'][idx])):
                # print(i)
                ratios.append( round(min( abs(np.complex(root_comp_table['Ar Roots'][idx][i])), abs(np.complex(root_comp_table['Ma Roots'][idx][i])) )/
                            max( abs(np.complex(root_comp_table['Ar Roots'][idx][i])), abs(np.complex(root_comp_table['Ma Roots'][idx][i])) ),4) )
        else:
            # print(i)
            for i in range(len(root_comp_table['Ar Roots'][idx])):
                ratios.append( round( min( abs(np.complex(root_comp_table['Ar Roots'][idx][i])), abs(np.complex(root_comp_table['Ma Roots'][idx][i])) )/ 
                            max( abs(np.complex(root_comp_table['Ar Roots'][idx][i])), abs(np.complex(root_comp_table['Ma Roots'][idx][i])) ),4) )
        root_comp_table['Ratios'][idx] = str(ratios)
    root_comp_table.drop(['ParamRoots'],axis=1)
    return root_comp_table

def arma_table(dictOfModels, y_test, y_train, lags=20, alpha=0.05):

    emptydf = pd.DataFrame()
    
    for name, modelpf in dictOfModels.items():
        tempdf = pd.DataFrame()
        arma_p = modelpf[0]
        arma_p_t = modelpf[2]
        arma_res = y_train['1st_dif_close'][1:-1] - arma_p[2:]
        arma_res_t = y_train['close'][1:-1] - arma_p_t[2:]
        Qab = acorr_ljungbox(arma_res.dropna(), lags, True)[2][-1]
        Qab_p = acorr_ljungbox(arma_res.dropna(), lags, True)[3][-1]
        dofab = (lags - int(name[0]) - int(name[1]))
        chi_crit = chi2.ppf(1-alpha, dofab)
        p_ = np.var(arma_res_t)
        # Forecast
        arma_f_t = modelpf[1]
        forecast_e_arma = y_test['close'] - arma_f_t
        f_ = np.var(forecast_e_arma)

        tempdf= pd.DataFrame({
        'Method': ['ARMA('+name[0]+','+name[1]+')'],
        'Q-Value': [Qab],
        'Q P-value':[Qab_p],
        # 'Qcrit': [chi_crit],
        'Var Fore Er/Var Res':[f_/p_],
        'MSE Residuals': [np.mean(np.power(arma_res, 2))],
        'Mean of Residuals':[np.mean(arma_res)],
        'MSE Forecast Errors': [np.mean(np.power(forecast_e_arma, 2))]
        })


        emptydf = pd.concat([emptydf, tempdf],axis=0)
        
    return emptydf

def arima_table(dictOfModels, y_test, y_train, lags=20, alpha=0.05):

    emptydf = pd.DataFrame()
    
    for name, modelpf in dictOfModels.items():
        tempdf = pd.DataFrame()
        arma_p = modelpf[0]
        arma_p_t = modelpf[2]
        arma_res_t = y_train['close'][1:] - arma_p_t
        Qab = acorr_ljungbox(arma_res_t.dropna(), lags, True)[2][-1]
        Qab_p = acorr_ljungbox(arma_res_t.dropna(), lags, True)[3][-1]
        dofab = (lags - int(name[0]) - int(name[1]))
        chi_crit = chi2.ppf(1-alpha, dofab)
        p_ = np.var(arma_res_t)
        # Forecast
        arma_f_t = modelpf[1]
        forecast_e_arma = y_test['close'] - arma_f_t
        f_ = np.var(forecast_e_arma)

        tempdf= pd.DataFrame({
        'Method': ['ARIMA('+name[0]+','+name[1]+','+name[2]+')'],
        'Q-Value': [Qab],
        'Q P-value':[Qab_p],
        # 'Qcrit': [chi_crit],
        'Var Fore Er/Var Res':[f_/p_],
        'MSE Residuals': [np.mean(np.power(arma_res_t, 2))],
        'Mean of Residuals':[np.mean(arma_res_t)],
        'MSE Forecast Errors': [np.mean(np.power(forecast_e_arma, 2))]
        })


        emptydf = pd.concat([emptydf, tempdf],axis=0)
        
    return emptydf

# Forecast plot:

def plot_ar_ma_forecast(y_test, arima_f_t, orderName, Bool=False):

    fig = plt.figure(figsize=(20, 10)) 
    grids = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    ax1 = plt.subplot(grids[0])
    ax1.plot(y_test['close'][0:10],  label='Observed', color='teal')
    ax1.plot(arima_f_t[0:10], label='Forecast {}'.format(orderName), color='r')
    ax1.set_title('{} Forecast - Forecast for 1-10 h step'.format(orderName))
    ax1.set_xticklabels(rotation=45, labels=y_test.index.strftime('%Y-%m-%d').to_list()[0:10])
        
    ax2 = plt.subplot(grids[1])
    ax2.plot(y_test['close'],  label='Observed', color='teal')
    ax2.plot(arima_f_t, label='Forecast {}'.format(orderName), color='r')
    ax2.set_title('{} Forecast - 908 h steps'.format(orderName))

    filename = orderName.replace(',', '').replace(')', '').replace('(', '_')

    if Bool == True:
        plt.savefig('Images/{}.png'.format(filename), dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

    for ax in [ax1,ax2]:
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.legend()
    plt.tight_layout()
    plt.show()

def param_sig(cov_theta_hat, theta_new):
    i=0
    pos = []
    neg = []
    for theta in theta_new:
        conf = 2*(np.sqrt(cov_theta_hat[i][i]))
        pos.append(theta+conf)
        neg.append(theta-conf)
    return neg, pos

def round_flatten(array):
    return [np.round(x,4) for x in np.ndarray.flatten(array)]

def roots_check(theta_new, na, nb):
    flat = np.ndarray.flatten(theta_new)
    if na > 0 and nb >0:
        return np.roots(np.r_[1, flat[0:na]]), np.roots(np.r_[1, flat[na:]])
    elif na > 0 and nb == 0:
        return np.roots(np.r_[1, flat]), 0
    elif na==0 and nb >0:
        return 0, np.roots(np.r_[1, flat])

def aRMA_GS(y_train, colname, j, k, save, submean):
    a=range(1,k)
    b=(range(0,j))
    ab = list(itertools.product(a,b))
    ab_arma = {'Order':[], 'aibi':[], 'AIC':[], 'ArMa Params':[],'ParamsPvalues':[], 'ConfInt_n':[], 'ConfInt_p':[],'ParamRoots':[], 'ResidQ_':[], 'Q_Pvalue':[]}
    ab_idx = []
    arma_params ={}
    for x in ab:
        try:
            if submean == True:
                model_arma = tsa.ARMA(y_train[colname].dropna() - np.mean(y_train[colname].dropna() ), order=x).fit(trend='nc',disp=0)
            else:
                model_arma = tsa.ARMA(y_train[colname].dropna(), order=x).fit(trend='nc',disp=0)
            ai = 0
            bi = 0
            Q = round(acorr_ljungbox(model_arma.resid, lags=20, boxpierce=True)[2][-1], 0)
            Qp = round(acorr_ljungbox(model_arma.resid, lags=20, boxpierce=True)[3][-1], 4)
            if len(model_arma.arparams) > 0:
                for a in model_arma.arparams:
                    ab_idx.append(str(x).strip()+'_a_'+str(ai))
                    ab_arma['Order'].append(str(x))
                    ab_arma['aibi'].append(ai)
                    ab_arma['AIC'].append( int(round(model_arma.aic,0)) )
                    ab_arma['ArMa Params'].append(round(a,4))
                    ab_arma['ParamsPvalues'].append( round(list(model_arma.pvalues)[ai], 4) )
                    ab_arma['ConfInt_n'].append( round(model_arma.conf_int().iloc[ai,0], 4))
                    ab_arma['ConfInt_p'].append( round(model_arma.conf_int().iloc[ai,1], 4))
                    ab_arma['ParamRoots'].append( str(round(model_arma.arroots[ai], 4)) if np.iscomplex(model_arma.arroots[ai]) else round(model_arma.arroots[ai], 4) )
                    ab_arma['ResidQ_'].append( Q )
                    ab_arma['Q_Pvalue'].append( Qp )
                    ai+=1
            if len(model_arma.maparams) > 0:
                for b in model_arma.maparams:
                    ab_idx.append(str(x).strip()+'_b_'+str(bi))
                    ab_arma['Order'].append(str(x))
                    ab_arma['aibi'].append(bi)
                    ab_arma['AIC'].append( int(round(model_arma.aic,0)) )
                    ab_arma['ArMa Params'].append(round(b,4))
                    ab_arma['ParamsPvalues'].append( round(list(model_arma.pvalues)[bi], 4) )
                    ab_arma['ConfInt_n'].append( round(model_arma.conf_int().iloc[bi,0], 4))
                    ab_arma['ConfInt_p'].append( round(model_arma.conf_int().iloc[bi,1], 4))
                    ab_arma['ParamRoots'].append( str(round(model_arma.maroots[bi], 4)) if np.iscomplex(model_arma.maroots[bi]) else round(model_arma.maroots[bi], 4) )
                    ab_arma['ResidQ_'].append( Q )
                    ab_arma['Q_Pvalue'].append( Qp )
                    bi+=1
            arma_params[str(x)] = (np.r_[model_arma.arparams*-1, model_arma.maparams]).reshape(-1,1)
        except Exception:
            print('error for na and nb order {}'.format(str(x)))
            traceback.print_exc()
            pass

    if save == True:
        pickle.dump( ab_arma, open( "pickle/arma_gs_table.p", "wb" ) )
        pickle.dump( arma_params, open( "pickle/arma_gs_params.p", "wb" ) )

    return ab_arma, ab_idx, arma_params

def aRiMA_GS(y_train,column,j, k, d, save):
    a=range(1,k)
    b=(range(0,j))
    ab = list(itertools.product(a,b))
    ab_arma = {'Order':[], 'aibi':[], 'AIC':[],'ParamsPvalues':[], 'ConfInt_n':[], 'ConfInt_p':[],'ParamRoots':[], 'ResidQ_':[], 'Q_Pvalue':[]}
    ab_idx = []
    arma_params ={}
    for x in ab:
        try:
            model_arma = tsa.statespace.SARIMAX(y_train[column].dropna(), trend='c', order=(x[0],d,x[1])).fit(disp=False)
            ai = 0
            bi = 0
            Q = round(acorr_ljungbox(model_arma.resid[1:], lags=20, boxpierce=True)[2][-1], 0)
            Qp = round(acorr_ljungbox(model_arma.resid[1:], lags=20, boxpierce=True)[3][-1], 4)
            if len(model_arma.arparams) > 0:
                for a in model_arma.arparams:
                    ab_idx.append(str(x).strip()+'_a_'+str(ai))
                    ab_arma['Order'].append(str(x))
                    ab_arma['aibi'].append(ai)
                    ab_arma['AIC'].append( int(round(model_arma.aic,0)) )
                    ab_arma['ParamsPvalues'].append( round(list(model_arma.pvalues)[ai], 4) )
                    ab_arma['ConfInt_n'].append( round(model_arma.conf_int().iloc[ai,0], 4))
                    ab_arma['ConfInt_p'].append( round(model_arma.conf_int().iloc[ai,1], 4))
                    ab_arma['ParamRoots'].append( str(round(model_arma.arroots[ai], 4)) if np.iscomplex(model_arma.arroots[ai]) else round(model_arma.arroots[ai], 4) )
                    ab_arma['ResidQ_'].append( Q )
                    ab_arma['Q_Pvalue'].append( Qp )
                    ai+=1
            if x[1] > 0:
                for b in model_arma.maparams:
                    ab_idx.append(str(x).strip()+'_b_'+str(bi))
                    ab_arma['Order'].append(str(x))
                    ab_arma['aibi'].append(bi)
                    ab_arma['AIC'].append( int(round(model_arma.aic,0)) )
                    ab_arma['ParamsPvalues'].append( round(list(model_arma.pvalues)[bi], 4) )
                    ab_arma['ConfInt_n'].append( round(model_arma.conf_int().iloc[bi,0], 4))
                    ab_arma['ConfInt_p'].append( round(model_arma.conf_int().iloc[bi,1], 4))
                    ab_arma['ParamRoots'].append( str(round(model_arma.maroots[bi], 4)) if np.iscomplex(model_arma.maroots[bi]) else round(model_arma.maroots[bi], 4) )
                    ab_arma['ResidQ_'].append( Q )
                    ab_arma['Q_Pvalue'].append( Qp )
                    bi+=1
            arma_params[str(x)] = (np.r_[model_arma.arparams*-1, model_arma.maparams if x[1]>0 else np.array([]) ]).reshape(-1,1)
        except Exception:
            print('error for na and nb order {}'.format(str(x)))
            traceback.print_exc()
            pass

    if save == True:
        pickle.dump( ab_arma, open( "pickle/arima_gs_table.p", "wb" ) )
        pickle.dump( arma_params, open( "pickle/arima_gs_params.p", "wb" ) )

    return ab_arma, ab_idx, arma_params

####################################################################################
# BASELINE FUNCTION PLOTS AND CALC
####################################################################################
def bl_plots_v2(y_train, y_test, bl1, bl2, columnName, methodName, Bool):

    plt.figure(figsize=(20,10))
    grids = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])

    ax1 = plt.subplot(grids[0])
    ax1.plot(y_train[columnName][1:7], label='Training set', lw=6,linestyle='dashed')
    ax1.plot(bl1[0:6], label='One Step Prediction {}'.format(methodName))
    ax1.set_title('First 5 forecast values')
    ax1.set_xticks(y_train['close'][1:7].index.strftime('%Y-%m-%d').to_list())
    ax1.set_xticklabels(rotation=45, labels=y_train['close'][1:7].index.strftime('%Y-%m-%d').to_list())
    ax1.set_xlabel('Time', fontsize=15)
    ax1.grid(axis='both', color='silver')

    ax2 = plt.subplot(grids[1])
    ax2.plot(y_train[columnName][1:], label='Training set', lw=6,linestyle='dashed')
    ax2.plot(bl1, label='One Step Prediction {}'.format(methodName))
    ax2.plot(y_test[columnName], label='Test set', linestyle='dashed', color='green')
    ax2.plot(bl2, label='One Step Forecast {}'.format(methodName), color='red')
    ax2.set_title('One step prediction and forecast for NASDAQ Index Stock Price COB')
    ax2.legend(frameon=False, fontsize=15)
    ax2.set_xlabel('Time', fontsize=15)
    ax2.set_ylabel('NASDAQ Stock Price', fontsize=15)

    ax3 = plt.subplot(grids[2])
    ax3.plot(y_train[columnName][-3:], label='Training set', lw=6,linestyle='dashed')
    ax3.plot(bl1[-3:], label='One Step Prediction {}'.format(methodName))
    ax3.plot(y_test[columnName][0:6], label='Test set', linestyle='dashed', color='green')
    ax3.plot(bl2[0:6], label='One Step Forecast {}'.format(methodName), color='red')
    ax3.set_title('First 5 forecast values')
    ax3.set_xticks(y_train[-3:].index.strftime('%Y-%m-%d').to_list()+y_test[0:6].index.strftime('%Y-%m-%d').to_list())
    ax3.set_xticklabels(rotation=45, labels= y_train[-3:].index.strftime('%Y-%m-%d').to_list()+ y_test[0:6].index.strftime('%Y-%m-%d').to_list())
    ax3.set_xlabel('Time', fontsize=15)
    ax3.grid(axis='both', color='silver')

    if Bool == True:
        plt.savefig('Images/{}_v2.png'.format(methodName), dpi=None, facecolor='w', edgecolor='w',
            orientation='landscape', papertype=None, format='png',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

plt.show()




def bl_plots(y_train, y_test, bl1, bl2, columnName, methodName, Bool):
    plt.figure(figsize=(20,10))
    plt.plot(y_train[columnName], label='Training set', lw=6,linestyle='dashed')
    plt.plot(bl1, label='One Step Prediction {}'.format(methodName))
    plt.plot(y_test[columnName], label='Test set', linestyle='dashed')
    plt.plot(bl2, label='One Step Forecast {}'.format(methodName))
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('yt', fontsize=15)
    plt.title('{} Prediction and Forecasting NASDAQ'.format(methodName), fontsize=25)
    plt.legend(loc="best", frameon=False, fontsize=15)

    if Bool == True:
        plt.savefig('Images/{}.png'.format(methodName), dpi=None, facecolor='w', edgecolor='w',
            orientation='landscape', papertype=None, format='png',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

    plt.show()

def test_plots(y_test, bl2, columnName, methodName, Bool):
    plt.figure(figsize=(20,10))
    plt.plot(y_test[columnName], label='Test', linestyle='dashed')
    plt.plot(bl2, label='Forecaste {}'.format(methodName))
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('yt', fontsize=15)
    plt.title('{} Prediction and Forecasting NASDAQ'.format(methodName), fontsize=25)
    plt.legend(loc="best", frameon=False, fontsize=15)

    if Bool==True:
        plt.savefig('Images/{}.png'.format(methodName), dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

    plt.show()

def ave_forecast(df_tr, df_te):
    ytt = []
    for i in range(1, len(df_tr)):
        ytt.append(df_tr.iloc[0:i].mean())
        
    test = np.ones(len(df_te)) * df_tr.mean()

    return pd.Series(ytt, index=df_tr.index.to_list()[1:]), pd.Series(test, index=df_te.index.to_list())


def naive_forecast(df_tr, df_te):

    ytt = df_tr.shift(1)[1:]
    test = np.ones(len(df_te)) * df_tr[-1]

    return pd.Series(ytt, index=df_tr.index.to_list()[1:]), pd.Series(test, index=df_te.index.to_list())


def drift_forecast(df_tr, df_te):
    ytt = []
    m_te = (df_tr[-1] - df_tr[0]) / (len(df_tr)-1)
    b_te = df_tr[-1]
    ys_te = []

    for i in (range(len(df_tr))):
        if i == 1:
            ytt.append(df_tr[i-1])
        if i >= 2:
            drift= df_tr[i-1] + (df_tr[i-1] - df_tr[0])/(i)
            ytt.append(drift)
    h=1
    for i in (range(len(df_te))):
        ys_te.append((m_te*h)+b_te)
        h+=1
    return  pd.Series(ytt, index=df_tr.index.to_list()[1:]), pd.Series(ys_te, index=df_te.index.to_list())



def acf_bl_multi_plot(data_training, df_name, methodName, Bool):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))
    fig.suptitle('{} ACF plots for Baseline and Holt-Winter Forecasting Methods'.format(df_name), fontsize=20, y=1.01)

    for k,v in {'Average':['e_ave', ax1], 'Naive':['e_naive', ax2], 'Drift':['e_drf', ax3], 
    'SES':['e_ses', ax4]}.items():
        try:

            n_o_lags = int(len(data_training)/20)
            acf_pos = acf(data_training[v[0]].dropna(), alpha=0.05)[0][1:][:20]
            acf_neg = acf(data_training[v[0]].dropna(), alpha=0.05)[0][::-1][-20-1:]
            v[1].stem(range( -len(acf_neg), len(acf_pos) ), np.r_[acf_neg,acf_pos])
            v[1].set_title('{} Forecast method for 20 lags'.format(k), fontsize=15)
        except:
            continue

    if Bool==True:
        plt.savefig('Images/{}.png'.format(methodName), dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    plt.show()

def acf_multi_plot(data_training, df_name, methodName, Bool):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))
    fig.suptitle('{} ACF plots for residuals'.format(df_name), fontsize=20, y=1.01)

    for k,v in {'Average':['res_ave', ax1], 'Naive':['res_nai', ax2], 'Drift':['res_dft', ax3], 
    'SES':['res_ses', ax4], 'Holt':['res_hlm', ax5], 'Holt Winter':['res_hw', ax6]}.items():
        try:

            n_o_lags = 20
            acf_pos = acf(data_training[v[0]].dropna(), alpha=0.05)[0][1:][:20]
            acf_neg = acf(data_training[v[0]].dropna(), alpha=0.05)[0][::-1][-20-1:]
            v[1].stem(range( -len(acf_neg), len(acf_pos) ), np.r_[acf_neg,acf_pos])
            v[1].set_title('{} Forecast method for {} lags'.format(k, n_o_lags), fontsize=15)
        except:
            continue
    if Bool==True:
        plt.savefig('Images/{}.png'.format(methodName), dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    plt.show()

def acf_single_plot(y,lags, alpha,title, xlabel,ylabel,save):
    plt.figure(figsize=(20,10))
    acf1 = acf(y, alpha=alpha)[0]
    plt.stem(range(-lags,lags+1), np.r_[acf1[1:lags+1][::-1], acf1[0:lags+1]] )
    plt.xticks(range(-lags,lags+1,2))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save == True:
        plt.savefig('Images/{}.png'.format(title), dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    plt.show()

def bl_comparison_table(data_training, data_testing, y_test, lags, T):

    cc_ave = correlation_coefficent_cal(data_testing['e_ave'].values, y_test['close'].values)
    cc_naive = correlation_coefficent_cal(data_testing['e_nai'].values, y_test['close'].values)
    cc_drf = correlation_coefficent_cal(data_testing['e_dft'].values, y_test['close'].values)
    cc_ses = correlation_coefficent_cal(data_testing['e_ses'].values, y_test['close'].values)
    cc_hlm = correlation_coefficent_cal(data_testing['e_hlm'].values, y_test['close'].values)
    cc_hwsm = correlation_coefficent_cal(data_testing['e_hw'].values, y_test['close'].values)

    q_ave = acorr_ljungbox(data_training['res_ave'], lags, True)[2][-1]
    q_naive = acorr_ljungbox(data_training['res_nai'], lags, True)[2][-1]
    q_ses = acorr_ljungbox(data_training['res_ses'], lags, True)[2][-1]
    q_drf = acorr_ljungbox(data_training['res_dft'], lags, True)[2][-1]
    q_hlm = acorr_ljungbox(data_training['res_hlm'], lags, True)[2][-1]
    q_hwsm = acorr_ljungbox(data_training['res_hw'], lags, True)[2][-1]

    pq_ave = acorr_ljungbox(data_training['res_ave'], lags, True)[3][-1]
    pq_naive = acorr_ljungbox(data_training['res_nai'], lags, True)[3][-1]
    pq_ses = acorr_ljungbox(data_training['res_ses'], lags, True)[3][-1]
    pq_drf = acorr_ljungbox(data_training['res_dft'], lags, True)[3][-1]
    pq_hlm = acorr_ljungbox(data_training['res_hlm'], lags, True)[3][-1]
    pq_hwsm = acorr_ljungbox(data_training['res_hw'], lags, True)[3][-1]

    p_var_ave = np.var(data_training['res_ave'].values)
    p_var_naive = np.var(data_training['res_nai'].values)
    p_var_ses = np.var(data_training['res_ses'])
    p_var_drf = np.var(data_training['res_dft'].values)
    p_var_hlm = np.var(data_training['res_hlm'])
    p_var_hwsm = np.var(data_training['res_hw'])

    f_var_ave = np.var(data_testing['e_ave'].values)
    f_var_naive = np.var(data_testing['e_nai'].values)
    f_var_ses = np.var(data_testing['e_ses'].values)
    f_var_drf = np.var(data_testing['e_dft'].values)
    f_var_hlm = np.var(data_testing['e_hlm'].values)
    f_var_hwsm = np.var(data_testing['e_hw'].values)


    table= pd.DataFrame({
        'Method': ['Average', 'Naive', 'SES', 'Drift', 'Holt Linear', 'Holt Winter'],

        'Q-Values': [q_ave, q_naive, q_ses, q_drf, q_hlm, q_hwsm],
        
        'P-value of Q': [pq_ave, pq_naive, pq_ses, pq_drf, pq_hlm, pq_hwsm],

        'Var Fore Er/Var Res':[f_var_ave/p_var_ave, f_var_naive/p_var_naive, f_var_ses/p_var_ses, f_var_drf/p_var_drf, f_var_hlm/p_var_hlm, f_var_hwsm/p_var_hwsm],

        'MSE of Residuals': [np.mean(np.power(data_training['res_ave'], 2)) , np.mean(np.power(data_training['res_nai'], 2)), 
        np.mean(np.power(data_training['res_ses'], 2)), np.mean(np.power(data_training['res_dft'], 2)), np.mean(np.power(data_training['res_hlm'], 2)),
        np.mean(np.power(data_training['res_hw'], 2))],

        'Mean of Residuals': [np.mean(data_training['res_ave']) , np.mean(data_training['res_nai']), 
        np.mean(data_training['res_ses']), np.mean(data_training['res_dft']), np.mean(data_training['res_hlm']),np.mean(data_training['res_hw'])],

        'MSE of Forecast Error': [np.mean(np.power(data_testing['e_ave'], 2)) , np.mean(np.power(data_testing['e_nai'], 2)), 
        np.mean(np.power(data_testing['e_ses'], 2)), np.mean(np.power(data_testing['e_dft'], 2)), np.mean(np.power(data_testing['e_hlm'], 2)),
        np.mean(np.power(data_testing['e_hw'], 2))],

        'Corr. Coef. Error v Actual': [cc_ave, cc_naive, cc_ses, cc_drf, cc_hlm, cc_hwsm]
        })
    
    return table

def holtWs_plot(y_test, holtwsm_f, y_train, holtwsm_fittedvalues, title, xlabel, ylabel, save):
    plt.figure(figsize=(20,10))
    plt.plot(y_test, label='NASDAQ Test set', linestyle='dashed')
    plt.plot(y_test.index.to_list(), holtwsm_f, label = 'Holt Winter Seasonal Method one step Forecast')
    plt.plot(y_train[1:], label='NASDAQ Train set', lw=6, linestyle='dashed')
    plt.plot(y_train.index.to_list()[1:], holtwsm_fittedvalues[1:], label = 'Holt Winter Seasonal Method one step Prediction')
    plt.legend(loc='best')
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=25)
    plt.legend(loc="best", frameon=False, fontsize=15)

    if save == True:
        plt.savefig('Images/hws.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

    plt.show()

def holt_LM(y_test, y_train, holtt_f, holtt_fittedvalues,save):
    plt.figure(figsize=(20,10))
    plt.plot(y_test, label='Test set', linestyle='dashed')
    plt.plot(y_test.index.to_list(), holtt_f, label = 'Holt Linear Method one step Forecast')
    plt.plot(y_train[1:], label='Train set', lw=6, linestyle='dashed')
    plt.plot(y_train.index.to_list()[1:], holtt_fittedvalues[1:], label = 'Holt Linear Method one step Prediction')
    plt.legend(loc='best')
    plt.xticks(rotation='vertical',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Stock Price', fontsize=15)
    plt.title('Holts Linear Forecasting NASDAQ Stock', fontsize=25)
    plt.legend(loc="best", frameon=False, fontsize=15)
    if save == True:

        plt.savefig('Images/hml.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='landscape', papertype=None, format='png',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

    plt.show()


####################################################################################
# MULTI-VARIATE LINEAR REGRESSION
####################################################################################

def forward_stepwise_reg(X, y, limit):
    iteration_props = []
    kept_feats = []
    cycles=1
    for cycle in range(0, len(X.columns)):

        remaining_feats = [x for x in list(X.columns) if x not in kept_feats]
        step_min_list = []
        for col in remaining_feats:
            
            added_feats = []
            added_feats.extend(kept_feats)
            added_feats.append(col)
            X_ = X[added_feats]
            lm = OLS(y, X_).fit()

            if lm.pvalues[-1] <= limit:
                step_min_list.append(col)
            
        if len(step_min_list) == 0:
            # print('There are no more features with p-value(s) <= '+str(limit) , end='\n\n')
            break
        else:
            minimum_pvalue = OLS(y, X[step_min_list]).fit()
            index_no_of_min = list(minimum_pvalue.pvalues).index(min(minimum_pvalue.pvalues))
            feat_name = list(minimum_pvalue.pvalues.keys())[index_no_of_min]
            kept_feats.append(feat_name)
            # print(kept_feats)
            cycle_lm = OLS(y, X[kept_feats]).fit()
            p_check = []
            for i in range(len(cycle_lm.pvalues)):
                if cycle_lm.pvalues[i] <= limit:
                    p_check.append(cycle_lm.pvalues.index[i])
            cycle_lm = OLS(y, X[p_check]).fit()


            # print('kept feats for cycle '+str(cycles)+' model: ', kept_feats, 'p-values of coef:', cycle_lm.pvalues, 'AIC of model: ', cycle_lm.aic, 'BIC of model: ', cycle_lm.bic, 'Adjusted R2:', cycle_lm.rsquared_adj, '-/-/-/-/-/-/-/-/-/-/-/', sep='\n', end='\n\n')
            iteration_props.append({'iter':cycles, 'params': cycle_lm.params, 'p-v': cycle_lm.pvalues, 'AIC': cycle_lm.aic, 'BIC':cycle_lm.bic, 'AdR2':cycle_lm.rsquared_adj})
            cycles+=1


    # print('End', kept_feats, sep='\n\n')
    return iteration_props


def stock_selector(X, y, limit, kept_feats):
    iteration_props =[]
    remaining_feats = [x for x in list(X.columns) if x not in kept_feats]
    model_num = 0
    for col in remaining_feats:
        added_feats = []
        added_feats.extend(kept_feats)
        added_feats.append(col)
        X_ = X[added_feats]
        lm = OLS(y, X_).fit()
        #print('Model Num',model_num ,'feats',added_feats, 'p-values of coef:', lm.pvalues, 'AIC of model: ', lm.aic, 'BIC of model: ', lm.bic, 'Adjusted R2:', lm.rsquared_adj, '-/-/-/-/-/-/-/-/-/-/-/', sep='\n', end='\n\n')
        iteration_props.append({'Model_Num':model_num, 'feats':added_feats, 'params': lm.params, 'p-v': lm.pvalues, 'AIC': lm.aic, 'BIC':lm.bic, 'AdR2':lm.rsquared_adj})
        model_num+=1

    #print('End', kept_feats, sep='\n\n')
    return iteration_props


def lm_comparison_table(data_training, data_testing, y_test, lags, T, Bool):
    cc_ = correlation_coefficent_cal(data_testing, y_test)
    q_ = acorr_ljungbox(data_training.dropna(), lags, True)[2][-1]
    pq_ = acorr_ljungbox(data_training.dropna(), lags, True)[3][-1]
    p_ = np.var(data_training)
    f_ = np.var(data_testing)
    table= pd.DataFrame({
        'Method': ['Linear Regression'],
        'Q-Value': [q_],
        'P-value of Q': [pq_],
        'Var Fore Er/Var Res':[f_/p_],
        'MSE Residuals': [np.mean(np.power(data_training.iloc[1:], 2))],
        'Mean of Residuals':[np.mean(data_training.iloc[1:])],
        'MSE Forecast Errors': [np.mean(np.power(data_testing, 2))],
        'Corr. Coef. Error v Test Set': [cc_]
        })
    
    if Bool == True:
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        pd.plotting.table(ax, table)  # where df is your data frame
        plt.savefig('lm_comparison_table_.png')

    return( table )

def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)