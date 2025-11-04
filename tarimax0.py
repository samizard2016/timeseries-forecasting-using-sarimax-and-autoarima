"""
paul, samir
Pune, Maharashtra
27 september 2021
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
from numpy import cov
import sklearn
from sklearn.metrics import mean_squared_error
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import warnings
import itertools
import gc
from statistics import *
import math

class TARIMAX:
    def __init__(self,*kwargs);
        _d = {}
        _d.update(kwargs)
        self.data = _d['data_file_name']

    def parse_quarter(self,x):
    """
    parse_quarter allows parsing string into quarterly dates from string including "Q1 2020",
    "Q22020","2020 Q3", "2021Q1","2020-Q2","Q2-2021" to "Q2-2021"
    """
        x = x.upper()
        if x[0].upper()=='Q' and len(x)==7:
             return "-".join(x.split())
        elif '-' in x:
            dts = x.split('-')
            if dts[0][0]=='Q':
                return x
            else:
                return "-".join(dts[::-1])
        else:
            if len(x)==7:
                return "-".join(x.split()[::-1])
            elif len(x)==6 and x[0]=='Q':
                 return "-".join([x[:2],x[3:]])
            else:
                return "-".join([x[4:],x[:4]])


    # def parse_date(self,date_col):
    #     # date_col = df.iloc[:,0]
    #     # Date = []
    #     # for i in date_col :
    #     #     year = i[:4]
    #     #     Qtr = i[5:]
    #     #     qt = (int(Qtr)-1)*3+1
    #     #     qtr = '%02d' % qt
    #     #     date = year + "-" + qtr + "-" + "01"
    #     #     Date.append(date)
    #     # data_ = df.copy()
    #     # data_.iloc[:,0] = Date
    #     # data_.rename(columns={data_.columns[0]:'Date'}, inplace=True)
    #     # will rewrite this part of code
    #     pass




        ## DATE FORMAT - this would come from GUI

        # set_date_format = True
        # while set_date_format :
        #     qt = input("Please give date format , eg Q, M, A, DMY")
        #     date_format = qt.upper()
        #     if date_format in ['Q', 'M', 'A', 'DMY'] :
        #         if date_format == 'Q' :
        #             data = parse_date(data)
        #             data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")
        #             data.index = data['Date']
        #             data.index.freq = "QS"
        #         if date_format == 'M' or date_format == 'A':
        #             break
        #         if date_format == 'DMY' :
        #             data['Date'] = pd.to_datetime(data['Date'], format = "%d-%m-%Y")
        #             data.index = data['Date']
        #         set_date_format = False



        # DECIDING ON THE VARIABLE TO FORECAST - to be set from the GUI
        #
        # data['Date'].head()
        #
        #
        # # ### user input filter
        #
        # # In[7]:
        #
        #
        # df = data.copy()
        # print("variable list :", [i for i in df.columns ],'\n')
        #
        # # setting dependent variable
        # set_dep = True
        # while set_dep :
        #     dep = input("please give dependent variable(Y) name except :  Date ")
        #     if dep in [i for i in df.columns if i != "Date" ] :
        #         set_dep = False
        #
        # print("Y : ", dep)

        # SELECTION OF X VARIABLES - in GUI and set to be passed in the dict
        # # setting fixed independent variables
        # set_fix = True
        # while set_fix :
        #     q = input("Do you want to fix independent variables [Y/N]: ")
        #     if q.upper() in ["Y","N"] :
        #         if q.upper() == "N" :
        #             set_fixed = False
        #         elif q.upper() == "Y" :
        #             set_fixed = True
        #         set_fix = False
        #
        #
        # fixed = []
        # while set_fixed :
        #     already_fixed = ["Date"]+ [dep]+ fixed
        #     fixed_vars = ''
        #     for element in already_fixed:
        #         fixed_vars = fixed_vars + " " + str(element)
        #
        #     fixed_indep = input("please give fixed independent variables(X) names except : {} ".format(fixed_vars))
        #     if fixed_indep in [i for i in df.columns if i not in already_fixed ] :
        #         fixed.append(fixed_indep)
        #
        #         fix_next =  True
        #         while fix_next :
        #             q = input("Do you want to fix more independent variables [Y/N]: ")
        #             if q.upper() in ["Y","N"] :
        #                 if q.upper() == "N" :
        #                     set_fixed = False
        #                 fix_next = False
        #
        # fixed_total = ["Date"] + [dep] + fixed
        # variables = [item for item in df.columns if item not in fixed_total]

#setting filter on coefficients of independent variables
indep = []
sign = []
set_coeff = True
while set_coeff :
    q = input("Do you wish to add filters on coefficient sign on independent variables [Y/N]: ")
    if q.upper() in ['Y','N']:
        if q.upper() == "Y":
            set_filter = True
        elif q.upper() == "N" :
            set_filter = False
        set_coeff = False

if set_filter :
    more_filter = True
    while more_filter:
        already_filtered =  ['Date'] + [dep] + indep
        filtered_vars = ''
        for element in already_filtered:
            filtered_vars = filtered_vars + " " + str(element)

        set_ele = True
        while set_ele :
            ele = input("Please give variable name except : {} ".format(filtered_vars))
            if ele in [i for i in df.columns if i not in already_filtered] :
                set_ele = False

        set_sign = True
        while set_sign :
            sig = input("Please give direction of coefficients : [P/N] (+ve/-ve) ")
            if sig.upper() in ['P','N']:
                set_sign = False

        indep.append(ele)
        sign.append(sig.upper())
        si = "+ve" if sig.upper() == "P" else "-ve"
        print(ele, si)

        set_next_filter = True
        while set_next_filter :
            q = input("Do you want to add more filters [Y/N]: ")
            if q.upper() in ["Y", "N"] :
                if q.upper() == "N" :
                    more_filter = False
                set_next_filter = False

coeff_filter = {}
for i in indep :
    for j in sign:
        if indep.index(i) == sign.index(j) :
            coeff_filter[i] = j

set_r2 = True
while set_r2 :
    r2_cut = int(input("Please give the cut off for lowest r2 in filter, 0-100 :"))
    if r2_cut >0 and r2_cut <100 :
        r2_cutoff = r2_cut/100
        set_r2 = False
# return(dep,fixed,variables,coeff_filter)

MAPE = int(input("please give acceptable level of MAPE for model eg: 5,10 "))
APE = int(input("please give acceptable level of maximum absolute percentage error for model eg: 5,10 "))
POS = int(input("please give alpha for percentage positive error ratio , pos = (positivee errors/ total errors)*100. accepatble range will be set to (alpha, 100-alpha) " ))


# In[8]:


def var_combination(fixed,variables) :
# Making list of all possible combinations of variables
    for i in range(len(variables)+1) :
        if i :
            c = c + list(itertools.combinations(variables,i))
        else :
            c = list(itertools.combinations(variables,i))
    for i in range(len(c)) :
        c[i] = tuple([dep]) + tuple(fixed) + c[i]
    return(c)
c = var_combination(fixed,variables)


# In[9]:




# In[10]:





# ### Grid search for ARIMA

# In[11]:


def ARIMA_results(results,data, endog, exog)  :
    #Prediction on data given endog and exog
    prediction = pd.DataFrame(results.predict(start = endog.index[0], end = endog.index[-1], exog = exog[exog.columns], dynamic = False))
    pcname = 'predicted value- y({})-x{}'.format(endog.columns[0],list(exog.columns))
    prediction.columns = [pcname]
    # predict = pd.concat([predict_test,predict_train])

    #Collecting Results from ARIMA fit
    df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
    df_ = pd.read_html(results.summary().tables[0].as_html())[0]
    df.columns = ['Coefficient','Std Err','z','p>z','CI_up','CI_down']
    df_.columns = ['measure','value','measure','value']
    df_ = df_.iloc[:,:2].append(df_.iloc[:,2:])
    df_.index = list(range(len(df_)))
    df_.index = df_.pop('measure')

    # **computing MAPE, r2, pval , tval, standardized coefficients**
    compare_df = data[dep].to_frame().merge(prediction, left_index = True, right_index = True)
    per = (compare_df[dep] - compare_df[pcname])/compare_df[dep]*100
    ape = abs(per)
    max_ape = max(list(ape))
    mape = mean(ape)
    covar = ((cov(compare_df[dep],compare_df[pcname])**2)/(cov(compare_df[dep],compare_df[dep])*cov(compare_df[pcname], compare_df[pcname])))[0,1]

    r2 = math.sqrt(covar)

    neg_err = list(per+ape).count(0)
    pos_err = len(per) - neg_err
    pos_err_ratio = pos_err/len(per)*100
    neg_err_ratio = neg_err/len(per)*100
    if neg_err_ratio != 0 :
        err_ratio = pos_err_ratio/neg_err_ratio
    else :
        err_ratio = 0

    cof = df['Coefficient'].to_frame()
    df_ = pd.concat([df_.iloc[:2],df_.iloc[7:13]])
    tes = df_.copy()
    calc_stats = [max_ape,mape,pos_err_ratio,neg_err_ratio,r2]
    calc = pd.DataFrame(calc_stats)
    calc.index = ['max APE','MAPE','Pos/Total error','Neg/Total error', 'R-Squared']
    tes.columns = cof.columns
    calc.columns = cof.columns
    concat = pd.concat([cof,tes,calc])
    mname = 'y({})-x{}'.format(endog.columns[0],list(exog.columns))
    concat.columns = [mname]
#     print(concat)

    return concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2, pcname,calc


# In[ ]:





# In[12]:


def ARIMA_coeff(results,endog,exog,calc_test) :
    #Collecting Results from ARIMA fit
    df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
    df_ = pd.read_html(results.summary().tables[0].as_html())[0]
    df.columns = ['Coefficient','Std Err','z','p>z','CI_up','CI_down']
    df_.columns = ['measure','value','measure','value']
    df_ = df_.iloc[:,:2].append(df_.iloc[:,2:])
    df_.index = list(range(len(df_)))
    df_.index = df_.pop('measure')
    df_ = pd.concat([df_.iloc[:2],df_.iloc[7:13]])
    cof = df['Coefficient'].to_frame()
    tes = df_.copy()
    tes.columns = cof.columns
    calc_test.columns = cof.columns
    concat_ = pd.concat([cof,tes,calc_test])
    mname = 'y({})-x{}'.format(endog.columns[0],list(exog.columns))
    concat_.columns = [mname]
    return cof, df, df_, concat_


# In[ ]:





# In[10]:


# **Setting filter**
def filter_function(concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2) :
    reason = []
    accept = True
    for i in coeff_filter :
        if coeff_filter[i] == "P" :
            if (df.loc[i,'Coefficient']<0) :
                accept = False
                reason.append("Coefficient sign - {} ".format(i))

        if coeff_filter[i] == "N" :
            if (df.loc[i,'Coefficient']>0) :
                accept = False
                reason.append("Coefficient sign - {} ".format(i))
    if pos_err_ratio < POS or pos_err_ratio > 100-POS :
        accept = False
        reason.append("Systematic Error")
    if mape > MAPE :
        accept = False
        reason.append("MAPE>{}".format(str(MAPE)))
    if max(list(ape)) > APE :
        accept = False
        reason.append("max APE > {}".format(str(APE)))
    if r2 < r2_cutoff :
        accept = False
        reason.append("R-squared < {}".format(str(r2_cut)))
    if accept :
        reason.append("All parameters satisfied")

#     for i in coeff_filter :
#         print(i ," : ",df.loc[i,'Coefficient'])

#     print("+ve error ratio : ", pos_err_ratio)
#     print("MAPE test : ", mape)
#     print("max APE test : ", max(list(ape)))
    if accept :
        print("Model accepted", '\n', "reason : ")
        for i in reason :
            print(i)
#     else :
#         print("Model rejected", '\n',"reason :")
#         for i in reason :
#             print(i)
    return accept


# In[13]:


def filter_function2(recoff_, redf)  :
    rereason = []
    reaccept = True
    for i in coeff_filter :
        if coeff_filter[i] == "P" :
            if (redf.loc[i,'Coefficient']<0) :
                reaccept = False
                rereason.append("Coefficient sign - {} ".format(i))

        if coeff_filter[i] == "N" :
            if (redf.loc[i,'Coefficient']>0) :
                reaccept = False
                rereason.append("Coefficient sign - {} ".format(i))
    return(reaccept)


# In[ ]:





# In[14]:


def ARIMA_forecast(data, exog_, endog_, result) :
    # Creating dataframe to store forecasted values
    forecast_range = []
    x = data.index[-1]
    for i in range(1,n_steps+1) :
        next_period = (x.replace(day=1) + datetime.timedelta(days=93*i)).replace(day=1)
        forecast_range.append(next_period)
    forecast_df = pd.DataFrame(columns = ['forecasted value'], index = forecast_range)
    # Creating exogenous variable for simulation forecasting
    forecast_exog = pd.DataFrame(columns = exog_.columns, index = forecast_range)
    forecast_exog1 = pd.DataFrame(columns = exog_.columns, index = forecast_range)
    forecast_exog2 = pd.DataFrame(columns = exog_.columns, index = forecast_range)
    forecast_exog3 = pd.DataFrame(columns = exog_.columns, index = forecast_range)
    forecast_exog4 = pd.DataFrame(columns = exog_.columns, index = forecast_range)
    theta1 = 5; theta2 = 10; theta3 = -5 ; theta4 = -10
    for i in forecast_exog.columns :
        forecast_exog[i] = data[i].iloc[-1]
    for i in forecast_exog1.columns :
        forecast_exog1[i] = data[i].iloc[-1]*(1+theta1/100)
    for i in forecast_exog2.columns :
        forecast_exog2[i] = data[i].iloc[-1]*(1+theta2/100)
    for i in forecast_exog3.columns :
        forecast_exog3[i] = data[i].iloc[-1]*(1+theta3/100)
    for i in forecast_exog4.columns :
        forecast_exog4[i] = data[i].iloc[-1]*(1+theta4/100)

#     for j in [forecast_exog,forecast_exog1,forecast_exog2,forecast_exog3,forecast_exog4] :
#         for i in j.columns :
#             j[i] = data[i].iloc[-1]

    # Forecasting and storing outputs
    forecast_df = pd.DataFrame(result.predict(start = forecast_exog.index[0], end = forecast_exog.index[-1], exog = forecast_exog, dynamic = True))
    forecast_df1 = pd.DataFrame(result.predict(start = forecast_exog1.index[0], end = forecast_exog1.index[-1], exog = forecast_exog1, dynamic = True))
    forecast_df2 = pd.DataFrame(result.predict(start = forecast_exog2.index[0], end = forecast_exog2.index[-1], exog = forecast_exog2, dynamic = True))
    forecast_df3 = pd.DataFrame(result.predict(start = forecast_exog3.index[0], end = forecast_exog3.index[-1], exog = forecast_exog3, dynamic = True))
    forecast_df4 = pd.DataFrame(result.predict(start = forecast_exog4.index[0], end = forecast_exog4.index[-1], exog = forecast_exog4, dynamic = True))

#     pcname = 'predicted value- pdq{}-y({})-x{}'.format(param,endog_.columns[0],list(exog_.columns))
    pcname = 'predicted value- y({})-x{}'.format(endog_.columns[0],list(exog_.columns))
    forecast_df.columns = [pcname]
    forecast_df1.columns = [pcname]
    forecast_df2.columns = [pcname]
    forecast_df3.columns = [pcname]
    forecast_df4.columns = [pcname]
    return forecast_df,forecast_df1,forecast_df2,forecast_df3,forecast_df4


# In[ ]:





# In[15]:


def gridsearch_arima() :
    count = 1
    counts = 0
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    bestAIC = np.inf
    bestParam = None
    bestSParam = None
    tempAIC = np.inf
    tempParam = None
    tempAIC = None
    total_cof = []
    op_df = pd.DataFrame()
    output_df = pd.DataFrame()
    comp_df = pd.DataFrame()
    forecast_ = pd.DataFrame()
    fcast_df = pd.DataFrame()
    for i in range(5) :
        for param in pdq :
            # **Setting endogenous and exogenous variables**
            temp_train = train.loc[:,list(c[i])]
            endog_train = temp_train[dep].to_frame()
            exog_train = temp_train.drop([dep], axis = 1)

            temp_test = test.loc[:,list(c[i])]
            endog_test = temp_test[dep].to_frame()
            exog_test = temp_test.drop([dep], axis = 1)

            temp = data.loc[:,list(c[i])]
            exog_ = temp.drop([dep], axis = 1)
            endog_ = temp[dep].to_frame()

            #Fitting Model
            for t in ['nc','c'] :

                try :
                    mod =  statsmodels.tsa.arima_model.ARIMA(endog_train, order= param, exog = exog_train)
                    results_train = mod.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
                    # print('\n','ARIMA{} - AIC:{}'.format(param, results_train.aic))
                    # print(" Y : ", endog_train.columns[0],'\n', "X : ", list(exog_train.columns))
                    concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2,pcname,calc_test = ARIMA_results(results_train,test,endog_test,exog_test)
                    accept_once = filter_function(concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2)
                    print("accept once: ",  accept_once)
                except :
                    continue


                    #checking for coefficient sign on full samples
                if accept_once == True :
                    print("checking again for criteria")
                    mod_re =  statsmodels.tsa.arima_model.ARIMA(endog_, order= param, exog = exog_)
                    results_re = mod_re.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
                    recoff, redf, redf_, reconcat = ARIMA_coeff(results_re,endog_,exog_, calc_test)
                    accept_again = filter_function2(recoff, redf)
                    print("accept again : ", accept_again)
                    forecast_,forecast1_,forecast2_,forecast3_,forecast4_ = ARIMA_forecast(data, exog_, endog_, results_re)

                # **merging coefficients of accepted models together**
                if accept_once == True and accept_again == True :
                    print('All filters satisfied')
                    total_cof += list(recoff.index)

                    if count ==1 :
                        op_df = reconcat
                        fcast_df = forecast_
                        fcast1_df = forecast1_
                        fcast2_df = forecast2_
                        fcast3_df = forecast3_
                        fcast4_df = forecast4_
                        print('1st Model')
                        count = 0
                    elif count == 0 :
                        op_df = op_df.merge(reconcat, how = 'outer', left_index = True, right_index = True)
                        fcast_df = fcast_df.merge(forecast_, left_index = True, right_index = True)
                        fcast1_df = fcast1_df.merge(forecast1_, left_index = True, right_index = True)
                        fcast2_df = fcast2_df.merge(forecast2_, left_index = True, right_index = True)
                        fcast3_df = fcast3_df.merge(forecast3_, left_index = True, right_index = True)
                        fcast4_df = fcast4_df.merge(forecast4_, left_index = True, right_index = True)
                        print("model added")

                counts += 1
                print(counts)

                # Best Model based on AIC
                if results_train.aic<bestAIC :
                    bestAIC = results_train.aic
                    bestParam = param

    total_cof = list(set(total_cof))
    return op_df,fcast_df,fcast1_df,fcast2_df,fcast3_df,fcast4_df, total_cof


# In[ ]:





# In[16]:


output,forecasts,forecasts1,forecasts2,forecasts3,forecasts4, m_cof = gridsearch_arima()


# In[ ]:





# In[81]:


output


# In[ ]:


output,forecasts,forecasts1,forecasts2,forecasts3,forecasts4, m_cof = gridsearch_arima()
# Transformations of outputs and rejection of models with AR coefficients >1
def output_transform(output,forecasts,forecasts1,forecasts2,forecasts3,forecasts4,m_cof) :
    if output.empty == False :
        reindex = []
        for i in output.index.to_list() :
            if i[:3] == 'ma.' or i[:3] == 'ar.' :
                reindex.append(i[:2]+i[4:5])
            else :
                reindex.append(i)
        output.index = reindex
        arma_cols = [i for i in m_cof if i[:3] == 'ma.' or i[:3] == 'ar.']
        var_cols = [i for i in m_cof if i[:3] != 'ma.' and i[:3] != 'ar.']
        arma_rename = [i[:2]+i[4:5] for i in arma_cols]
        arma_rename.sort()
        m_cof = var_cols + arma_rename
        other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
        rearr = ['Dep. Variable:','Model:'] + m_cof + other_rows
        output = output.T[rearr]
        trans_ = output.copy()

        condition = (abs(trans_['ar1']) <1) & (abs(trans_['ar2']) <1)
        choice = [i for i in condition]

        output = output[choice]
        output = output.T
        output.columns = output.iloc[1,:]
        output = output[output.index != 'Model:']
        output = output.T.sort_values(['MAPE','R-Squared' ], ascending = (True,False)).T

        forecasts = forecasts.T
        forecasts = forecasts[choice]
        forecasts = forecasts.T

        forecasts1 = forecasts1.T
        forecasts1 = forecasts1[choice]
        forecasts1 = forecasts1.T

        forecasts2 = forecasts2.T
        forecasts2 = forecasts2[choice]
        forecasts2 = forecasts2.T

        forecasts3 = forecasts3.T
        forecasts3 = forecasts3[choice]
        forecasts3 = forecasts3.T

        forecasts4 = forecasts4.T
        forecasts4 = forecasts4[choice]
        forecasts4 = forecasts4.T

        forecasts.columns =output.columns.to_list()
        forecasts1.columns =output.columns.to_list()
        forecasts2.columns =output.columns.to_list()
        forecasts3.columns =output.columns.to_list()
        forecasts4.columns =output.columns.to_list()

        return  output,forecasts,forecasts1,forecasts2,forecasts3,forecasts4
op,fore,fore1,fore2,fore3,fore4 = output_transform(output,forecasts,forecasts1,forecasts2,forecasts3,forecasts4,m_cof)
op.to_csv('ARIMAX output {}'.format(file_name))
fore.to_csv('ARIMAX forecasts {}'.format(file_name))
fore1.to_csv('ARIMAX forecasts1 {}'.format(file_name))
fore2.to_csv('ARIMAX forecasts2 {}'.format(file_name))
fore3.to_csv('ARIMAX forecasts3 {}'.format(file_name))
fore4.to_csv('ARIMAX forecasts4 {}'.format(file_name))


# In[ ]:





# In[ ]:





# ### elasticity inputs

# In[ ]:


def simulation_inputs() :
    n_steps = int(input("please give the no of period for forecasting : "))


# In[21]:


def cp_inputs():
    # Fixing cetrus perribus simulations
    cp = []
    fixed_cp = []
    set_cp = True

    vari = [i for i in variables+fixed ]
    print("variable list :", vari,'\n')

    while set_cp :
        cp_vars = ""
        for element in cp:
            cp_vars = cp_vars + " " + str(element)

        fixed_cp = input("please give variable name for simulation from variable list :  except : {} ".format(cp_vars))
        if fixed_cp in [i for i in variables+fixed] :
            cp.append(fixed_cp)

            fix_next =  True
            while fix_next :
                q = input("Do you want to give more variables for simulation [Y/N]: ")
                if q.upper() in ["Y","N"] :
                    if q.upper() == "N" :
                        set_cp = False
                    fix_next = False
    cp_list = list(set(data.columns).intersection(set(cp)))
    n_steps = int(input("please give the no of period for forecasting : "))
    return cp_list


# In[17]:


cp_list
# set(forecast_exog.columns)
# list(exog_set.intersection(set(cp_list)))


# In[22]:


cp_list = cp_inputs()


# In[23]:


def ARIMA_forecast(exogen, endogen, result, name) :
    # Creating dataframe to store forecasted values
    forecast_range = []
    x = exogen.index[-1]
    for i in range(1,n_steps+1) :
        next_period = (x.replace(day=1) + datetime.timedelta(days=93*i)).replace(day=1)
        forecast_range.append(next_period)
    forecast_df = pd.DataFrame(columns = ['forecasted value'], index = forecast_range)
    # Creating exogenous variable for simulation forecasting
    forecast_exog = pd.DataFrame(columns = exogen.columns, index = forecast_range)
    thetas = [-10,-5,0,5,10]
    for i in forecast_exog.columns :
        forecast_exog[i] = exogen[i].iloc[-1]
    exog_set = set(forecast_exog.columns)
    cp_ = list(exog_set.intersection(set(cp_list)))
    for i in cp_ :
        for theta in thetas :
            temp_df = forecast_exog.copy()
            temp_df[i] = temp_df[i]*(1+theta/100)
            temp_forecast = pd.DataFrame(results_re.predict(start = forecast_exog.index[0], end = forecast_exog.index[-1], exog = temp_df, dynamic = True))
            pcname = 'predicted value-{}-sim({})-theta({})-y({})-x{}'.format(name,i,theta,endog_.columns[0],list(exog_.columns))
            temp_forecast.columns = [pcname]
            if cp_list.index(i) == 0 and thetas.index(theta) == 0 :
                simulation_df = temp_forecast
            else :
                simulation_df = simulation_df.merge(temp_forecast, left_index = True, right_index = True)
    return simulation_df


# In[ ]:





# In[24]:


def AR_filter(output) :
    if output.empty == False :
        reindex = []
        for i in output.index.to_list() :
            if i[:3] == 'ma.' or i[:3] == 'ar.' :
                reindex.append(i[:2]+i[4:5])
            else :
                reindex.append(i)
        output.index = reindex
        arma_cols = [i for i in output.index if i[:3] == 'ma.' or i[:3] == 'ar.']
        var_cols = [i for i in output.index if i[:3] != 'ma.' and i[:3] != 'ar.']
        arma_rename = [i[:2]+i[4:5] for i in arma_cols]
        arma_rename.sort()
        m_cof = var_cols + arma_rename
#         other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
#         rearr = ['Dep. Variable:','Model:'] + m_cof + other_rows
#         other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
        rearr =  m_cof

        output = output.T[rearr]
        trans_ = output.copy()
        condition = ((abs(trans_['ar1']) <1) & (abs(trans_['ar2']) <1)) | ( math.isnan(trans_['ar1']))
        print(condition.values[0])
#         choice = [i for i in condition]
#         output = output[choice]
#         output = output.T
#     return output

    return condition.values[0]



# In[ ]:





# In[25]:


# temp_test0
# type(AR_filter(temp_test0))
# temp_test2
AR_filter(reconcat).values[0]


# In[ ]:





# In[26]:


def gridsearch_arima() :
    count = 1
    counts = 0
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    bestAIC = np.inf
    bestParam = None
    bestSParam = None
    tempAIC = np.inf
    tempParam = None
    tempAIC = None
    total_cof = []
    op_df = pd.DataFrame()
    output_df = pd.DataFrame()
    comp_df = pd.DataFrame()
    forecast_ = pd.DataFrame()
    fcast_df = pd.DataFrame()
    for i in range(5) :
        for param in pdq :
            # **Setting endogenous and exogenous variables**
            temp_train = train.loc[:,list(c[i])]
            endog_train = temp_train[dep].to_frame()
            exog_train = temp_train.drop([dep], axis = 1)

            temp_test = test.loc[:,list(c[i])]
            endog_test = temp_test[dep].to_frame()
            exog_test = temp_test.drop([dep], axis = 1)

            temp = data.loc[:,list(c[i])]
            exog_ = temp.drop([dep], axis = 1)
            endog_ = temp[dep].to_frame()

            #Fitting Model
            for t in ['nc','c'] :

                try :
                    mod =  statsmodels.tsa.arima_model.ARIMA(endog_train, order= param, exog = exog_train)
                    results_train = mod.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
                    print('\n','ARIMA{} - AIC:{}'.format(param, results_train.aic))
                    print(" Y : ", endog_train.columns[0],'\n', "X : ", list(exog_train.columns))
                    concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2,pcname,calc_test = ARIMA_results(results_train,test,endog_test,exog_test)
                    accept_once = filter_function(concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2)
                    print("accept once: ",  accept_once)

                    #checking for coefficient sign on full samples
                    if accept_once == True :
                        print("checking again for criteria")
                        mod_re =  statsmodels.tsa.arima_model.ARIMA(endog_, order= param, exog = exog_)
                        results_re = mod_re.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
                        recoff, redf, redf_, reconcat = ARIMA_coeff(results_re,endog_,exog_, calc_test)
                        accept_again = filter_function2(recoff, redf)
                        print("accept again : ", accept_again)
                        rem_name = reconcat.loc['Model:'][0]
                        forecast_ = ARIMA_forecast(exog_, endog_, results_re, rem_name)

                    if accept_once == True and accept_again == True :
                        print("applying AR coeff filters")
                        temp_test0 = reconcat
                        ar_filter = AR_filter(reconcat)
                        print(ar_filter)
                        if ar_filter == True :
                            print('All filters satisfied')
                            total_cof += list(recoff.index)
                            if count ==1 :
                                op_df = reconcat
                                fcast_df = forecast_
                                print('1st Model')
                                count = 0
                                temp_test1 = reconcat
                            elif count == 0 :
                                op_df = op_df.merge(reconcat, how = 'outer', left_index = True, right_index = True)
                                fcast_df = fcast_df.merge(forecast_, left_index = True, right_index = True)
                                print("model added")
                                temp_test2 = reconcat

                    counts += 1
                    print(counts)

                    # Best Model based on AIC
                    if results_train.aic<bestAIC :
                        bestAIC = results_train.aic
                        bestParam = param
                except :
                    continue
    total_cof = list(set(total_cof))
    return op_df,fcast_df,total_cof


# In[ ]:





# In[27]:


output
forecasts
# type(reconcat.loc['Model:'][0])
# reconcat.loc['Model:'][0]


# In[ ]:





# In[356]:


output,forecasts,m_cof = gridsearch_arima()


# In[333]:


# output,forecasts, m_cof = gridsearch_arima()
# Transformations of outputs and rejection of models with AR coefficients >1
def output_transform(output,forecasts,m_cof) :
    if output.empty == False :
        reindex = []
        for i in output.index.to_list() :
            if i[:3] == 'ma.' or i[:3] == 'ar.' :
                reindex.append(i[:2]+i[4:5])
            else :
                reindex.append(i)
        output.index = reindex
        arma_cols = [i for i in m_cof if i[:3] == 'ma.' or i[:3] == 'ar.']
        var_cols = [i for i in m_cof if i[:3] != 'ma.' and i[:3] != 'ar.']
        arma_rename = [i[:2]+i[4:5] for i in arma_cols]
        arma_rename.sort()
        m_cof = var_cols + arma_rename
        other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
        rearr = ['Model:','Dep. Variable:'] + m_cof + other_rows
        output = output.T[rearr]
        trans_ = output.copy()

        condition = (abs(trans_['ar1']) <1) & (abs(trans_['ar2']) <1)
        choice = [i for i in condition]

#         output = output[choice]
        output = output.T
#         output.columns = output.iloc[0,:]
#         output = output[output.index != 'Model:']
        output = output.T.sort_values(['MAPE','R-Squared' ], ascending = (True,False)).T

#         forecasts = forecasts.T
#         forecasts = forecasts[choice]
#         forecasts = forecasts.T

#         forecasts.columns =output.columns.to_list()

        return  output,forecasts
op,fore = output_transform(output,forecasts,m_cof)
# op.to_csv('ARIMAX output {}'.format(file_name))
# fore.to_csv('ARIMAX forecasts {}'.format(file_name))


# In[137]:


forecast_ = ARIMA_forecast(exog_, endog_, results_re)


# In[347]:


op
# fore


# In[ ]:


################################################################################################################################


# #### for testing ARIMAX loops

# In[ ]:


['Price_LPP_Large (105+ gm)', 'Hdi', 'Nom_gdp_pc']


# In[349]:


count = 1
counts = 0
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
bestAIC = np.inf
bestParam = None
bestSParam = None
tempAIC = np.inf
tempParam = None
tempAIC = None
total_cof = []
op_df = pd.DataFrame()
output_df = pd.DataFrame()
comp_df = pd.DataFrame()
forecast_ = pd.DataFrame()
t = 'nc'
# testing individual model of autoarima
list1 = ['Vol_LPP_Large (105+ gm)','Price_LPP_Large (105+ gm)', 'Nom_gdp_pc', 'Hdi']
param = (0,0,0)
temp_train = train.loc[:,list1]
endog_train = temp_train[dep].to_frame()
exog_train = temp_train.drop([dep], axis = 1)

temp_test = test.loc[:,list1]
endog_test = temp_test[dep].to_frame()
exog_test = temp_test.drop([dep], axis = 1)

temp = data.loc[:,list1]
exog_ = temp.drop([dep], axis = 1)
endog_ = temp[dep].to_frame()


mod =  statsmodels.tsa.arima_model.ARIMA(endog_train, order= param, exog = exog_train)
results_train = mod.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
# print('\n','ARIMA{} - AIC:{}'.format(param, results_train.aic))
# print(" Y : ", endog_train.columns[0],'\n', "X : ", list(exog_train.columns))
concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2,pcname,calc_test = ARIMA_results(results_train,test,endog_test,exog_test)
accept_once = filter_function(concat,compare_df,cof,df,pos_err_ratio,mape,ape,r2)
print("accept once: ",  accept_once)


    #checking for coefficient sign on full samples
if accept_once == True :
    print("checking again for criteria")
    mod_re =  statsmodels.tsa.arima_model.ARIMA(endog_, order= param, exog = exog_)
    results_re = mod_re.fit(method = 'css', trend = t, solver = 'lbfgs', maxiter = 500, tol = 0.0001)
    recoff, redf, redf_, reconcat = ARIMA_coeff(results_re,endog_,exog_, calc_test)
    accept_again = filter_function2(recoff, redf)
    print("accept again : ", accept_again)
    forecast_ = ARIMA_forecast(exog_, endog_, results_re)

# **merging coefficients of accepted models together**
if accept_once == True and accept_again == True :
    print('All filters satisfied')
    total_cof += list(recoff.index)

    if count ==1 :
        op_df = reconcat
        fcast_df = forecast_
        print('1st Model')
        count = 0
    elif count == 0 :
        op_df = op_df.merge(reconcat, how = 'outer', left_index = True, right_index = True)
        fcast_df = fcast_df.merge(forecast_, left_index = True, right_index = True)
        print("model added")
counts += 1
print(counts)


# In[353]:


results_re.aic
results_re.model


# In[95]:


reindex = []
for i in output.index.to_list() :
    if i[:3] == 'ma.' or i[:3] == 'ar.' :
        reindex.append(i[:2]+i[4:5])
    else :
        reindex.append(i)
output.index = reindex
arma_cols = [i for i in m_cof if i[:3] == 'ma.' or i[:3] == 'ar.']
var_cols = [i for i in m_cof if i[:3] != 'ma.' and i[:3] != 'ar.']
arma_rename = [i[:2]+i[4:5] for i in arma_cols]
arma_rename.sort()
m_cof = var_cols + arma_rename
other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
rearr = ['Dep. Variable:','Model:'] + m_cof + other_rows
output = output.T[rearr]
trans_ = output.copy()

# condition = (abs(trans_['ar1']) <1) & (abs(trans_['ar2']) <1)
# choice = [i for i in condition]

# output = output[choice]
# output = output.T
# output.columns = output.iloc[1,:]
# output = output[output.index != 'Model:']
# output = output.T.sort_values(['MAPE','R-Squared' ], ascending = (True,False)).T


# In[140]:


# output,test_prediction,forecasts, m_cof = gridsearch_arima()
def output_transform(output, test_prediction,forecasts, m_cof) :
    if output.empty == False :
        reindex = []
        for i in output.index.to_list() :
            if i[:3] == 'ma.' or i[:3] == 'ar.' :
                reindex.append(i[:2]+i[4:5])
            else :
                reindex.append(i)
        output.index = reindex
        arma_cols = [i for i in m_cof if i[:3] == 'ma.' or i[:3] == 'ar.']
        var_cols = [i for i in m_cof if i[:3] != 'ma.' and i[:3] != 'ar.']
        arma_rename = [i[:2]+i[4:5] for i in arma_cols]
        arma_rename.sort()
        m_cof = var_cols + arma_rename
#         other_rows = ['No. Observations:','Pos/Total error','Neg/Total error','MAPE','max APE','R-Squared','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
        other_rows = ['No. Observations:','AIC','BIC','HQIC','Log Likelihood','S.D. of innovations']
        rearr = ['Dep. Variable:','Model:'] + m_cof + other_rows
        output = output.T[rearr].T
        trans = output.T
    #     trans = trans[(abs(trans['ar1']) <1) & (abs(trans['ar2']) <1)]
        output = trans.T
        output.columns = output.iloc[1,:]
        output = output[output.index != 'Model:']
#         output = output.T.sort_values(['MAPE','R-Squared' ], ascending = (True,False)).T
        test_prediction.columns = [dep]+ output.columns.to_list()
        forecasts.columns =output.columns.to_list()
        return  output, test_prediction, forecasts
op,pre,fore = output_transform(output,test_prediction,forecasts, m_cof)
op.to_csv('ARIMAX output {}'.format(file_name))
pre.to_csv('ARIMAX test predictions {}'.format(file_name))
fore.to_csv('ARIMAX forecasts {}'.format(file_name))


# In[ ]:


##############################################################################################################################


# In[ ]:


################################################################################################################################


# ### Further tests

# In[24]:


type(results.summary().tables[1])
results.summary().tables[1]
results_as_html = results.summary().tables[1].as_html()
res = pd.read_html(results_as_html, header=0, index_col=0)[0]


# In[74]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[ ]:





# In[27]:


# Alias Description for data index frequency
# B	business day frequency
# C	custom business day frequency (experimental)
# D	calendar day frequency
# W	weekly frequency
# M	month end frequency
# BM	business month end frequency
# MS	month start frequency
# BMS	business month start frequency
# Q	quarter end frequency
# BQ	business quarter endfrequency
# QS	quarter start frequency
# BQS	business quarter start frequency
# A	year end frequency
# BA	business year end frequency
# AS	year start frequency
# BAS	business year start frequency
# H	hourly frequency
# T	minutely frequency
# S	secondly frequency
# L	milliseonds
# U	microseconds
