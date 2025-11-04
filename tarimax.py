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
from statsmodels.stats.stattools import durbin_watson
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
# from batch_sarimax import BSARIMAX
import pmdarima as pm
import warnings
import itertools
import gc
from statistics import *
import math
import logging
import pprint as pp
import pmdarima.arima._doc

class TARIMAX:
    def __init__(self,**kwargs):
        self.logger = logging.getLogger("anuman")
        _d = {}
        _d.update(kwargs)
        try:
            self.data_file = _d['data_file_name']
            self.date_var = _d['date_var']
            self.period = _d['period']
            self.dependent_var = _d['dependent_var']
            self.vars = _d['vars']
            self.seasonal = False
            if 'train_test_ratio' in _d:
                self.train_test_ratio = _d['train_test_ratio']/100
            # self.vars.remove(_d['date_var'])
            self.data = pd.read_excel(self.data_file)
            if 'intercept' in _d:
                self.constant=_d['intercept']
        except Exception as err:
            self.logger.error(f"tarimax.__init__: {err}")
        # self.data.to_csv('t_data.csv',index=False)
        self.logger.info(f"date_var {self.date_var} dv {self.dependent_var} period {self.period}")
    def set_index(self):
        try:
            self.data['period'] = self.data[self.date_var].apply(self.parse_quarter)
            self.data['period'] = pd.to_datetime(['-'.join(x.split('-')[::-1]) for x in self.data['period']])
            self.data.set_index('period',inplace=True)
            self.data = self.data[self.vars]
            self.data.to_csv('selected_vars.csv',index=False)
            # remove old date var from the vars list
            if self.date_var in self.vars:
                self.vars.remove(self.date_var)
            self.split_data_tt()
        except Exception as err:
            self.logger.error(f"error encountered in tarimax.set_index: {err}")
    @staticmethod
    def get_exog(dep,exog_fixed,exog_others):
        for i in range(len(exog_others)+1):
            if i==0:
                comb = list(itertools.combinations(exog_others,i))
            else:
                comb = comb + list(itertools.combinations(exog_others,i))
        for i in range(len(comb)):
            comb[i] = tuple([dep]) + tuple(exog_fixed) + comb[i]
        return comb
    def get_dependent_var(self):
        return self.data[[self.dependent_var]]
    def significant_order_from_pacf(self,data):
        pacf_summary = pacf(data,nlags=20,method=('ols'))
        cc = abs(pacf_summary[1:10])
        occurrences_more_than_p3 = cc > .3
        count = occurrences_more_than_p3.sum()
        return count

    def adf_test(self):
        try:
            dftest = adfuller(self.data[[self.dependent_var]], autolag = 'AIC')
            # self.data[[self.dependent_var]].to_csv("adf_data.csv",index=False)
            adf =  {
                        "ADF":dftest[0],
                        "P-Value": dftest[1],
                        "Number Of Lags": dftest[2],
                        "Number of Observations used for\n ADF Regression and Critical Values Calculation": dftest[3]
                    }
            for key, val in dftest[4].items():
                adf[f"Critical Values - {key}"] = val
        # df = pd.DataFrame(adf,index=[0])
        # df.to_csv('adft.csv')
            return adf
        except Exception as err:
            self.logger.error(f"Couldn't run ADFT: {err}")


    def SARIMAX(self,**kwargs):
        _d = {}
        _d['data'] = self.train_data[[self.dependent_var]]
        ##self.split_data_tt()
        try:
            cols = list(self.data.columns)
            cols.remove(self.dependent_var)
            _d['exog'] = self.train_data[cols]
            # self.data.to_csv("t.csv")
        except Exception as err:
            self.logger.error(f"problem in defining exog vars - {err}")
        _d.update(kwargs)
        try:
            p = _d['p']
            d = _d['d']
            q = _d['q']
            _d['seasonal'] = _d['seasonal'] if 'seasonal' in _d else False
            if _d['seasonal']:
                P = _d['P']
                D = _d['D']
                Q = _d['Q']
                m = _d['m']
                self.sarimax_model = sm.tsa.statespace.SARIMAX(endog=self.train_data[[self.dependent_var]].astype(float),
                                                exog=_d['exog'].astype(float),
                                                trend='c' if self.constant else 'n',
                                                order=(p,d,q),
                                                seasonal_order=(P,D,Q,m))
                self.logger.info("set to seasonal")
            else:
                self.sarimax_model = sm.tsa.statespace.SARIMAX(endog=self.train_data[[self.dependent_var]].astype(float),
                                                    exog=_d['exog'].astype(float),
                                                    trend='c' if self.constant else 'n',
                                                    order=(p,d,q))
                self.logger.info("set to non seasonal")
            res = self.sarimax_model.fit(disp=False)
           ## pred_vol=self.get_perdicted_volume(res,cols)

            # bs = BSARIMAX(**{'result':res,
            #                 'endog':self.data[[self.dependent_var]].astype(float),
            #                 'exog':_d['exog'].astype(float),
            #                 'dependent_var': self.dependent_var,
            #                 'data': self.data
            #                 })
            # bs.parse_results()
            # bs.get_prediction_details()
            return res
        except Exception as err:
            self.logger.error(f"problem in tarimax.SARIMAX: {err}")

    def durbin_watson_test(self,resid):
        return durbin_watson(resid)

    def get_RSquared(self, resid):
        #tdf = pd.concat([self.data[[self.dependent_var]], resid])

        #tdf.to_excel("R Squared.xlsx")
        y_vals = self.train_data[[self.dependent_var]].values
        y_vals = (y_vals - np.mean(y_vals)) ** 2
        ss_tot = sum(y_vals)
        resid = resid.values
        ss_resid = sum(resid ** 2)
        return (1 - ss_resid / ss_tot)[0]

    def ARIMA(self,**kwargs):
        _d = {}
        _d.update(kwargs)
        try:
            p = _d['p']
            d = _d['d']
            q = _d['q']
            model = ARIMA( self.data[[self.dependent_var]],
                            order=(p,d,q))
            model_fit = model.fit(disp=0)
            return model_fit
        except Exception as e:
            self.logger.error(f"error in tarimax.ARIMA : {e}")
    def auto_arima(self,**kwargs):
        _d = {}
        _d['data'] = self.data[[self.dependent_var]]
        try:
            cols = list(self.data.columns)
            cols.remove(self.dependent_var)
            _d['exog'] = self.data[cols]
        except Exception as err:
            self.logger.error(f"problem in defining exog vars - {err}")
        _d.update(kwargs)
        # pp.pprint(_d)

        try:
            self.logger.info(f"tarimax.auto_arima: {_d['exog']}")
            self.auto_arima_model = pm.auto_arima(
                    _d['data'],
                    X =_d['exog'],
                    test='adf',
                    start_p =_d['start_p'],
                    start_q =_d['start_q'],
                    trace = True,
                    error_action = 'ignore',
                    seasonal = False,
                    suppress_warnings = True,
                    stepwise = True,
                    random_state = 0)
            return self.auto_arima_model
        except Exception as err:
            self.logger.error(f"probelm in running auto arima - {err}")

    def __str__(self):
        return f"{self.data_file} has been selected as the data"

    def parse_quarter(self,x):
        '''
        parse_quarter allows parsing string into quarterly dates from string including "Q1 2020",
        "Q22020","2020 Q3", "2021Q1","2020-Q2","Q2-2021" to "Q2-2021"
        '''
        x = str(x)
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
    def split_data_tt(self):
        # Train:test split of the data
        split = int(self.train_test_ratio*len(self.data))
        self.train_data = self.data.iloc[:split,]
        self.test_data = self.data.iloc[split:,]

    def decompose(self):
        '''
        seasonal data decomposition
        '''
        y = self.data.set_index(self.date_var).resample(self.period).sum()[self.dependent_var]
        self.decomposed_data = sm.tsa.seasonal_decompose(y, model= 'm', extrapolate_trend = 'freq')
        # self.decomposed_data.plot()
        df_decomposed = pd.DataFrame({
                        'trend': self.decomposed_data.trend,
                        'season': self.decomposed_data.seasonal,
                        'residual': self.decomposed_data.resid
                        })
        # df_decomposed.to_excel(f"{self.data_file_name}_decomposed.xlsx")

if __name__=='__main__':
    data_file = "Thailand_Toothpaste_Nationwide"
    d = {"data_file_name":data_file}
    tarimax = TARIMAX(**d)
    # print(tarimax)
