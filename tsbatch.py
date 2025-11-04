"""
january 19 2022
paul, samir. Pune, Maharashtra, India
  "dataset":"thai_toothpaste.csv"
  "variables": ["Date","Vol_Nationwide","P_Nationwide","Population",
                "CCPI","Tourists","HDI","CCI","R_GDP","R_Export_Goods_and_Services",
                "Employment","R_Agri_Prod","Paddy_Price_Index","Paddy_Prod_Index"],
  "var-to be exclude": ["CCI"],
  "dependent_var":['Vol_Nationwide/Population'],
  "fixed_var":["R_GDP/pop","P_Nationwide/CCPI (corrected for inflation)"],
  "date": ['Date'],
  "to be finally excluded":["Population",  "CCPI",]
  "decision param":['AIC',"MAPE"]
  "sign":['gdp positive','price_negative']
"""
import itertools
import os
import pandas as pd
import numpy as np
import json
import logging
from statsmodels.tsa.stattools import acf
import pmdarima as pm
from pmdarima.arima import ADFTest
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from collections import namedtuple
from operator import attrgetter
from think_excel import ThinkExcel
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
# Declaring namedtuple()
SARIMAXModel = namedtuple('SARIMAXModel', ['model', 'AIC', 'MAPE'])
class TSFBatch:
    """
    Handles a batch of ts forecasting based on combinations of exog variables and summarize
    top models based on selection criteria - signs of certain xvars and performance params such as
    AIC and MAPE
    """
    def __init__(self,**kwargs):
        self.logger = logging.getLogger("anuman")
        """DEBUG INFO WARNING ERROR CRITICAL"""
        self.logger.info("TSF Batch starts")
        _d = {}
        _d.update(kwargs)
        self.load_settings(_d['project_setting_file'])
        self.set_output_excel()
        self.prepare_dataset()

    def run_models(self):
        models = []
        df_test_forecasts = []
        df_train_forecasts = []
        df_Nochange_PI_GDPI_forecasts = []
        df_NPIGr_forecasts = []
        _xsets = self.get_exog_combinations()
        self.logger.info(f"A total of {len(_xsets)} combinations of exog variables are being processed")
        try:
            adf_test = ADFTest(alpha=0.05)
            p_val, should_diff = adf_test.should_diff(self.data[self.dep_var].astype(float))
            # Estimate the number of differences using an ADF test:
            n_adf = ndiffs(self.data[self.dep_var].astype(float), test='adf')
            self.logger.info(f"ADFTest: p value: {p_val} Non-stationary: {should_diff} differences req: {n_adf}")
            # adf test using statsmodel
            adft = adfuller(self.data[self.dep_var].astype(float),autolag='AIC')
            self.logger.info(f"adfuller: {adft}")
        except Exception as e:
            self.logger.error(f"adft: {e}")
        # get the elasticity datasets ready
        self.get_elasticity_dataset_prepared()

        coeff_tables = []
        for indx,vset in enumerate(_xsets):
            d_var = vset[0]
            x_list = list(vset[1:])
            if self.auto_arima:
                auto_arima_model = pm.auto_arima(
                            self.train_data[d_var],
                            exogenous = self.train_data[x_list],
                            test='adf',
                            start_p = 0,
                            start_q = 0,
                            seasonal = False,
                            trace = True,
                            error_action = 'ignore',
                            suppress_warnings = True,
                            stepwise = True,
                            random_state = 0)
                p,d,q,P,D,Q,M = self.get_auto_arima_orders(auto_arima_model)
                model_orders = []
                model_orders.append((p,d,q))
                # print(model_orders)
            else:
                #seasonal order from auto_arima
                auto_arima_model = pm.auto_arima(
                            self.train_data[d_var],
                            exogenous = self.train_data[x_list],
                            test='adf',
                            start_p = 0,
                            start_q = 0,
                            seasonal = False,
                            trace = True,
                            error_action = 'ignore',
                            suppress_warnings = True,
                            stepwise = True,
                            random_state = 0)
                p,d,q,P,D,Q,M = self.get_auto_arima_orders(auto_arima_model)
                model_orders = self.get_combinations_for_pdq(**self.d_arima_order)
            for order in model_orders:
                (p,d,q) = order
                if self.settings["tsf-batch-settings"]["intercept"] == "Yes":
                    sarimax_model = sm.tsa.statespace.SARIMAX(endog=self.train_data[d_var].astype(float),
                                                                exog=self.train_data[x_list].astype(float),
                                                                trend="c",
                                                                order=(p,d,q),
                                                                seasonal_order=(P,D,Q,M))
                else:
                    sarimax_model = sm.tsa.statespace.SARIMAX(endog=self.train_data[d_var].astype(float),
                                                              exog=self.train_data[x_list].astype(float),
                                                              order=(p, d, q),
                                                              seasonal_order=(P, D, Q, M))
                model_fit = sarimax_model.fit(disp=False)
                self.forecast_test = self.get_forecast(model_fit,x_list)
                self.resi = model_fit.resid
                #new_df=self.forecast_volume_train()
                # code for Durbin_Watson and R-Squared
                dw_test = durbin_watson(self.resi)
                r_squared = self.get_RSquared()
                MAPE = self.forecast_accuracy(self.forecast_test,self.test_data[d_var])['mape']
                RMSE = self.forecast_accuracy(self.forecast_test, self.test_data[d_var])['rmse']
                MAPE_train = self.forecast_accuracy_train(self.resi,self.train_data[d_var])['mape']
                RMSE_train = self.forecast_accuracy_train(self.resi, self.train_data[d_var])['rmse']
                    ### elasticity
                pe,ie= self.get_price_and_income_elasticities(model_fit,x_list,f'{p}{d}{q}')
                # code to get exog and model combinations
                tdf_Nochange_PI_GDPI_forecasts = self.forecast_volume_Elasticity()
                tdf_test = self.forecast_volume_test()
                tdf_train = self.forecast_volume_train()
                t_model = f"Auto Arima - SARIMAX({p}{d}{q})({P}{D}{Q}{M})" if self.auto_arima else f"SARIMAX({p}{d}{q})"
                tdf_test["model"] = t_model
                tdf_train["model"] = t_model
                tdf_Nochange_PI_GDPI_forecasts["model"] = t_model
                t_exog = self.get_label(x_list)
                tdf_test["exog"] = t_exog
                tdf_train["exog"] = t_exog
                tdf_Nochange_PI_GDPI_forecasts["exog"] = t_exog

                df_test_forecasts.append(tdf_test)
                df_train_forecasts.append(tdf_train)
                df_Nochange_PI_GDPI_forecasts.append(tdf_Nochange_PI_GDPI_forecasts)

                npi_growth, tdf_NPIGr = self.get_NPI_growth(model_fit,x_list,f'{p}{d}{q}')
                # code to get forecasted volumes for npi growth
                tdf_NPIGr['model'] = t_model
                tdf_NPIGr['exog'] = t_exog
                df_NPIGr_forecasts.append(tdf_NPIGr)
                # print(f"model AIC: {model_fit.aic} and MAPE {MAPE}")
                models.append(SARIMAXModel(model_fit,model_fit.aic,MAPE))
                # std coeff
                tab = self.results_summary_to_dataframe(model_fit) 
                # df_std_coeff = self.standardized_coefficients(model_fit)
                # df_std_coeff = self.std_coeff(model_fit)
                # tab = pd.concat([tab,df_std_coeff],axis=1)    
                df_std_coeff = self.standardized_coefficients(model_fit,x_list)
                tab = pd.concat([tab, df_std_coeff['std_coeff']], axis=1)   
                sign_checked = self.get_signs_checked(tab)
                tab['model'] =  t_model
                tab['exog'] = self.get_label(x_list)
                tab['AIC'] = model_fit.aic
                tab['MAPE_TEST'] = MAPE
                tab['RMSE_TEST'] = RMSE
                tab['MAPE_TRAIN'] = MAPE_train
                tab['RMSE_TRAIN'] = RMSE_train
                tab['Sign Checked'] = sign_checked
                tab['price elasticity'] = pe
                tab['income elasticity'] = ie
                tab['npi_growth'] = npi_growth
                tab['Durbin-Watson-Test'] = dw_test
                tab['R-Squared'] = r_squared                
                tab['std_coeff']=tab['std_coeff'].fillna("")

                # self.write_to_ws(tab,ws,disptab)
                # row = row + tab.shape[0]+1
                sig_x = self.scan_coeff(tab)
                # print(f"significant exog vars: {sig_x}\n")
                tab = self.get_var_sig(tab)
                coeff_tables.append(tab)

        dfn_test_forecasts = pd.concat(df_test_forecasts)
        dfn_train_forecasts = pd.concat(df_train_forecasts)
        dfn_Nochange_PI_GDPI_forecasts = pd.concat(df_Nochange_PI_GDPI_forecasts)
        dfn_NPIGr_forecasts = pd.concat(df_NPIGr_forecasts)

        df_coeff = self.get_model_coeff(coeff_tables)

        if self.split_ratio < 1:
            df_coeff_test = df_coeff
            self.write_to_ws(df_coeff_test, "model-coeff", 'Tab6')
            self.write_to_ws(dfn_train_forecasts, "Volume-train", 'Tab7')
            self.write_to_ws(dfn_test_forecasts, "Volume-test", 'Tab8')
            self.write_to_ws(dfn_Nochange_PI_GDPI_forecasts, "Forecasted Volume", 'Tab9')
            self.write_to_ws(dfn_NPIGr_forecasts, "Volume-NPIGrowth", 'Tab10')        
        else:
            df_coeff.drop(['MAPE_TEST', 'RMSE_TEST'], axis=1, inplace=True)
            self.write_to_ws(df_coeff, "model-coeff", 'Tab6')
            self.write_to_ws(dfn_train_forecasts, "Volume-train", 'Tab7')
            self.write_to_ws(dfn_Nochange_PI_GDPI_forecasts, "Forecasted Volume", 'Tab8')
            self.write_to_ws(dfn_NPIGr_forecasts, "Volume-NPIGrowth", 'Tab9')
        if self.seasonality == 'Yes':
            df_seasonality=pd.DataFrame({'Seasonal-Factors':self.seasonal_factors})
            self.write_to_ws(df_seasonality, "Seasonal-Factors", 'Tab11')
        self.xl.save_current_wb()
        msg = f"output has been saved to {self.output_file}"
        self.logger.info(f"{msg}")
        return msg
    def standardized_coefficients(self,results,x_list):
        std_X = self.train_data[x_list].astype(float).std()
        std_Y = self.train_data[self.dep_var].astype(float).std()
        coefficient = results.params
        df_coeff = pd.DataFrame({"coeff":coefficient})
        # df_coeff.to_excel("coefficient.xlsx")
        try:
            std_coeff = {}
            for col in x_list:
                std_coeff[col] = df_coeff.loc[col, 'coeff'] * (std_X.loc[col] / std_Y)
            self.logger.info(f"std_coeff_dict:{std_coeff}")
            df_std_coeff= pd.DataFrame(std_coeff,index=[0]).T
            df_std_coeff = df_std_coeff.rename(columns = {0:'std_coeff'})
            # df_std_coeff.to_excel("standardized coefficients.xlsx")
            return df_std_coeff
        except Exception as err:
            self.logger.error(f"Standardized_coefficients: {err}")
    def std_coeff(self,results):
        try:
            cov = results.cov_params()
            std_x = np.sqrt(cov.values.diagonal())
            std_Y = self.train_data[[self.dep_var]].astype(float).std().values[0]
            mult = [x/std_Y for x in std_x]
            coeff = results.params
            std_coeff = coeff*mult
            std_coeff.columns = ['std_coeff']
            return std_coeff
        except Exception as err:
            self.logger.error(f"std_coeff: {err}")
    # def standardized_coefficients(self,results):
    #     try:
    #         df_cov_params = pd.DataFrame(results.cov_params())
    #         variances = df_cov_params.values.diagonal()
    #         std_X = pd.Series(np.sqrt(variances),index = df_cov_params.columns)
    #         # std_X.to_excel("Standard_deviation_x.xlsx")
    #         std_Y = self.train_data[[self.dep_var]].astype(float).std()
    #         # mult = std_X/std_Y
    #         # self.logger.info(f"mult: {mult}")
    #         # coeff = results.params*mult
    #         # pd.DataFrame({"std_coeff":coeff}).to_excel("std_ceff.xlsx")
    #         # return pd.DataFrame({"std_coeff":coeff})
    #         # std_Y.to_excel("Standard_deviation_y.xlsx")
    #         coefficient = results.params
    #         df_coeff = pd.DataFrame({"coeff":coefficient})
    #         beta_coefficients =[]
    #         for col, row in df_coeff.iterrows():
    #             beta = row['coeff'] * (std_X.loc[col] / std_Y.values)
    #             beta_coefficients.append([col, beta])
    #         beta_coefficients = [[var, beta.item()] for var, beta in beta_coefficients]
    #         df_beta = pd.DataFrame(beta_coefficients, columns=["variable", "std_coeff"])
    #         df_beta = df_beta.set_index('variable')
    #         # df_beta.to_excel("standardized_coefficients.xlsx", index=False)           
    #         return df_beta
    #     except Exception as err:
    #         self.logger.error(f"Standardized_coefficients: {err}")

    def get_model_coeff(self,coeff_table):
        def format_std_coeff(x):
            try:
                return f"{x:0.6f}"
            except Exception:
                pass
        df_coeff = pd.concat(coeff_table)
        df_coeff.reset_index(inplace=True)
        df_coeff.columns = ['variable', 'coeff', 'pvals', 'conf_lower', 'conf_higher','Z values','std_coeff',
                            'model', 'exog', 'AIC', 'MAPE_TEST', 'RMSE_TEST', 'MAPE_TRAIN', 'RMSE_TRAIN',
                            'Signed Checked', 'price elasticity', 'income elasticity', 'NPI Growth(%)',
                            'Durbin-Watson-Test','R-Squared','Significant']
        df_coeff = df_coeff[['variable', 'coeff', 'std_coeff','Signed Checked', 'pvals', 'Significant',
                             'conf_lower', 'conf_higher','Z values', 'model', 'exog', 'price elasticity',
                             'income elasticity', 'NPI Growth(%)', 'AIC', 'MAPE_TEST', 'RMSE_TEST',
                             'MAPE_TRAIN', 'RMSE_TRAIN','Durbin-Watson-Test','R-Squared']]

        for col in ['coeff', 'std_coeff','pvals', 'conf_lower', 'conf_higher', 'AIC', 'MAPE_TRAIN','MAPE_TEST']:
            if col in ['coeff', 'std_coeff']:
                df_coeff[col] = df_coeff[col].apply(format_std_coeff)
            else:
                df_coeff[col] = df_coeff[col].apply(lambda x: f"{x:0.4f}")

        df_coeff['amape'] = df_coeff['MAPE_TEST'].apply(float).apply(abs)
        df_coeff['aaic'] = df_coeff['AIC'].apply(float).apply(abs)
        df_coeff.sort_values(by=['model', 'aaic', 'amape'], inplace=True, ascending=True)
        df_coeff.drop(['amape', 'aaic'], inplace=True, axis=1)

        return df_coeff

    #def model_wit_TTD:
    def get_RSquared(self):
        #tdf = pd.concat([self.data[[self.dependent_var]], resid])

        #tdf.to_excel("R Squared.xlsx")
        y_vals = self.train_data[[self.dep_var]].values
        y_vals = (y_vals - np.mean(y_vals)) ** 2
        ss_tot = sum(y_vals)
        resid = self.resi.values
        ss_resid = sum(resid ** 2)
        return (1 - ss_resid / ss_tot)[0]

    #def model_wit_TTD:

    def get_var_sig(self,tab):
        tab['significant'] = tab.apply(lambda row: 'Yes' if row['pvals'] < self.settings["tsf-batch-settings"]["signficance"]
                                            else 'No',axis=1)
        return tab
    def get_label(self,labs):
        label = ""
        for lab in labs:
            label += (lab+"-")
        return label[:-1]

    def get_combinations_for_pdq(self,**kwargs):
        _d = {}
        _d.update(**kwargs)
        
        model_orders = []
        for p in range(_d['p_max']+1):
            for d in range(_d['d_max']+1):
                for q in range(_d['q_max']+1):
                   model_orders.append((p,d,q))
        return model_orders

    def get_forecast(self,model,x_list):
        #split_ratio = self.settings["tsf-batch-settings"]["train-test-ratio"]
        if self.split_ratio < 1:
            forecast_res = model.get_forecast(self.test_data.shape[0],exog=self.test_data[x_list].astype(float))
            # self.logger.info(f"forecasts: {forecast_res.predicted_mean}")
            forecasted_vol=forecast_res.predicted_mean
            if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation':
                forecasted_vol=self.get_inversed_vol(forecasted_vol)
                #forecasted_vol.to_excel("inv-transformed-vol.xlsx")
            if self.seasonality == "Yes":
                forecasted_vol=self.get_forecasted_vol(forecasted_vol)
            return forecasted_vol

    def get_inversed_vol(self,forecasted_vol):        
        #self.logger.info(f'transformation: {self.settings["tsf-batch-settings"]["transformation"]}')
        try:
            d_transform = {'log':np.exp,'sqrt':np.square,'boxcox':inv_boxcox,'normalize':self.normalize_data}
            if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation':
                for key,val in self.settings["tsf-batch-settings"]["transformation"].items():
                    for var in val:
                        if var == self.dep_var:
                            if key=='boxcox':
                                try:
                                    forecasted_vol = d_transform[key](forecasted_vol,self.fitted_lambda)
                                    # self.data[var].to_csv('boxcox_.csv')
                                    return forecasted_vol
                                except Exception as err:
                                    self.logger.error(f"boxcox: {err}")
                            else:
                                forecasted_vol = d_transform[key](forecasted_vol)
                                return forecasted_vol
                        else:
                            return forecasted_vol
        except Exception as err:
            self.logger.error(f"get_transformed_vol:{err}")

    def get_forecasted_vol(self,forecasted_vol):
        d_sf=self.get_seasonal_factors()
        if self.model_additive:
            for index,item in forecasted_vol.iteritems():
                month=str(index).split("-")[1]
                forecasted_vol[index]=forecasted_vol[index]+d_sf[month]            
        else:
            for index,item in forecasted_vol.iteritems():
                month=str(index).split("-")[1]
                forecasted_vol[index]=forecasted_vol[index]*d_sf[month]
        return forecasted_vol

    def get_forecast_dfx(self,model,dataset,x_list):
        forecast_res = model.get_forecast(dataset.shape[0],exog=dataset[x_list].astype(float))
        forecasted_vol=forecast_res.predicted_mean
        if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation':
                forecasted_vol=self.get_inversed_vol(forecasted_vol)                
        if self.seasonality == "Yes":
                forecasted_vol=self.get_forecasted_vol(forecasted_vol)
        return forecasted_vol
        

    def load_settings(self,json_file_name):
        try:
            with open(json_file_name) as f:
                self.settings = json.load(f)
        except Exception as err:
            self.logger.error(f"problem in reading the TSFBatch settings: {err}")
    def get_other_exog(self):
        try:
            columns = list(self.data.columns)
            self.logger.info(f'columns: {columns}')
            self.logger.info(f'dep var: {self.settings["tsf-batch-settings"]["dependent-var"]}')
            columns.remove(self.settings["tsf-batch-settings"]["dependent-var"])
            self.logger.info(f'fixed vars: {self.settings["tsf-batch-settings"]["fixed-xvars"]}')
            for item in self.settings["tsf-batch-settings"]["fixed-xvars"]:
                columns.remove(item)
            # remove actual volume
            act_vol = self.settings["tsf-batch-settings"]["Actual-vol-column"]            
            columns.remove(act_vol)
            self.logger.info(f"other-exog: {columns}")
            return columns
        except Exception as err:
            self.logger.error(f"get_other_exog: {err}")
    def get_arima_order(self,ts_order):
        try:
            order = tuple(ts_order)
            return order
        except Exception as err:
            self.logger.error(f"get_arima_order: {err}")
    def get_summary_of_the_dataset(self):
        summary = self.data.describe()
        summary = summary.T
        if self.settings["tsf-batch-settings"]["round-off-summary-data"]=='Yes':
            for col in summary.columns:
                summary[col] = summary[col].apply(lambda x: f"{x:0.4f}")
        return summary
    def transform_data(self):
        # self.data.to_csv("before_transformation.csv")
        self.logger.info(f'transformation: {self.settings["tsf-batch-settings"]["transformation"]}')
        d_transform = {'log':np.log,'sqrt':np.sqrt,'boxcox':stats.boxcox,'normalize':self.normalize_data}
        if self.settings["tsf-batch-settings"]["transformation"] != 'None':
            for key,val in self.settings["tsf-batch-settings"]["transformation"].items():
                for var in val:
                    if key=='boxcox':
                        try:
                            self.data[var],self.fitted_lambda = d_transform[key](self.data[var])
                            # self.data[var].to_csv('boxcox_.csv')
                        except Exception as err:
                            self.logger.error(f"boxcox: {err}")
                    else:
                        self.data[var] = d_transform[key](self.data[var])
    def normalize_data(self,df):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data)
        return df
            # self.data.to_csv('transformed_data.csv')
    def set_data_precision(self):
        try:
            for col in self.data.columns:
                self.data[col] = self.data[col].apply('float64')
        except Exception as err:
            self.logger.error(f"set_data_precision: {err}")
    def prepare_dataset(self):
        try:
            self.logger.info(f'indexed on: {self.settings["tsf-batch-settings"]["indexed on"]}')
            self.logger.info(f'frequency: {self.settings["tsf-batch-settings"]["frequency"]}')
            self.df_data = pd.read_excel(self.settings["tsf-batch-settings"]["dataset"])
            self.logger.info(f'variables: {self.settings["tsf-batch-settings"]["variables"]}')
            # print(self.df_data.columns)
            index_var = self.settings["tsf-batch-settings"]["indexed on"]
            self.df_data = self.df_data.set_index(index_var)
            freq = self.settings["tsf-batch-settings"]["frequency"]
            self.logger.info(f'frequency code: {self.settings["tsf-batch-settings"]["frequency-options"][freq]}')
            self.df_data.index.freq = self.settings["tsf-batch-settings"]["frequency-options"][freq]
            vars = self.settings["tsf-batch-settings"]["variables"]
            vars.remove(index_var)
            self.logger.info(f"vars after removing index var: {vars}")
            self.data = self.df_data[vars]
            self.set_data_precision()
            self.dep_var = self.settings["tsf-batch-settings"]["dependent-var"]
            self.seasonality=self.settings["tsf-batch-settings"]["seasonality"]            
           #seasonality
            self.model_additive=self.settings["tsf-batch-settings"]["model-additive"]
            # if self.model_additive:
            # print(f'Additive model selected {self.model_additive}')
            if self.seasonality=='Yes': 
                try:           
                    self.decompose_seasonality()
                    if self.model_additive:
                        self.data[self.dep_var]=self.data[self.dep_var]-self.seasonal_factors
                    else:
                        self.data[self.dep_var]=self.data[self.dep_var]/self.seasonal_factors
                except Exception as err:
                    self.logger.error(f"prepare_dataset problem in seasonality: {err}")      
            
            self.exog_fixed = self.settings["tsf-batch-settings"]["fixed-xvars"]
            
            if self.settings["tsf-batch-settings"]["auto-arima"]=='yes':
                self.auto_arima = True
                self.logger.info(f"auto-arima: {self.auto_arima}")
            else:
                self.auto_arima = False
                self.logger.info(f"auto-arima: {self.auto_arima}")
                self.d_arima_order = self.settings["tsf-batch-settings"]["ts-order"]
                # self.arima_order = self.get_combinations_for_pdq(self.settings["tsf-batch-settings"]["ts-order"])
                # self.logger.info(f"arima_order: {self.arima_order}")
            self.exog_others = self.get_other_exog()
            # transformation the variables
            # self.logger.info(f"summary of the dataset before transformation\n")
            # self.logger.info(self.get_summary_of_the_dataset())
            self.write_to_ws(self.df_data,'original datset','Tab42')
            self.write_to_ws(self.data,
            'data before transformation' if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation'
            else 'data',
            'Tab1')
            self.write_to_ws(self.get_summary_of_the_dataset(),
            'data summary before transformations' if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation'
            else 'data summary',
            'Tab2')
            if self.settings["tsf-batch-settings"]["transform-variables"] != 'No Transformation':
                self.transform_data()
                self.write_to_ws(self.data,'data after transformation','Tab3')
                self.write_to_ws(self.get_summary_of_the_dataset(),'data summary after transformations','Tab4')
            # correlation matrix
            df_corr = self.data.corr()
            self.write_to_ws(df_corr,'correlation matrix','Tab41')

            self.split_data_train_test()
        except Exception as err:
            self.logger.error(f"prepare_dataset: {err}")

    def decompose_seasonality(self):
        try:               
            result=seasonal_decompose(self.data[self.dep_var],model='additive'if self.model_additive else "multiplicative")
            self.seasonal_factors = result.seasonal
            # self.seasonal_factors.to_excel("seasonal factors.xlsx")
            self.logger.info("sesonal factors completed")
        except Exception as err:
            self.logger.error("sesonal factors failed: {err}")

    def get_seasonal_factors(self):
        d_sf={}
        try:
            for dt in self.seasonal_factors.index:
                month_str=str(dt).split("-")[1]
                d_sf[month_str]=self.seasonal_factors[dt]
            return d_sf
        except Exception as err:
            self.logger.error("get_seasonal_factors: {err}")

    def set_output_excel(self):
        self.xl = ThinkExcel()
        file_name = os.path.basename(self.settings["tsf-batch-settings"]["dataset"])
        self.output_file = f'TSF {file_name}'
        self.wb = self.xl.create_wb(self.output_file)
    def write_to_ws(self,df,sheet,dispTabName):
        self.xl.add_df_to_ws(sheet,new=True,df=df)
        ref = self.xl.get_ref(start_row=1,start_col=1,end_row=df.shape[0]+2,end_col=df.shape[1]+1)
        self.xl.add_table_to_ws(dispTabName,ref)
        # cref = [f"{self.xl.d_alpha[col]}1" for col in range(df.shape[1]+2)]
        # for r in cref:
        #     self.xl.format_header(r)

    def get_exog_combinations(self):
        for i in range(len(self.exog_others)+1):
            if i==0:
                comb = list(itertools.combinations(self.exog_others,i))
            else:
                comb = comb + list(itertools.combinations(self.exog_others,i))
        for i in range(len(comb)):
            comb[i] = tuple([self.dep_var]) + tuple(self.exog_fixed) + comb[i]
        return comb
    def split_data_train_test(self):
        try:
            self.split_ratio= self.settings["tsf-batch-settings"]["train-test-ratio"]
            split = int(self.split_ratio*len(self.data))
            self.train_data = self.data.iloc[:split,]
            self.test_data = self.data.iloc[split:,]
        except Exception as e:
            self.logger.error(f"problem in splitting train and test data: {e}")

    @staticmethod
    def get_auto_arima_orders(auto_arima_model):
        res = str(auto_arima_model).strip()
        indx = [6,8,10,13,15,17,20]
        keys = ['p','d','q','P','D','Q','M']
        _d = {}
        for ind,key in enumerate(keys):
            _d[key] = int(res[indx[ind]])
        return (_d['p'],_d['d'],_d['q'],_d['P'],_d['D'],_d['Q'],_d['M'])
    @staticmethod
    def results_summary_to_dataframe(results):
        """
        takes the result of an statsmodel results table and transforms it into a dataframe
        """
        pvals = results.pvalues
        coeff = results.params
        conf_lower = results.conf_int()[0]
        conf_higher = results.conf_int()[1]
        zvalues = results.zvalues

        results_df = pd.DataFrame({"coeff":coeff,
                                   "pvals":pvals,
                                   "conf_lower":conf_lower,
                                   "conf_higher":conf_higher,
                                   "zvalues":zvalues
                                    })
        # results_df.to_csv("coeff_df.csv")
        return results_df
    def get_signs_checked(self,df_coeff):
        cols = list(df_coeff.columns)
        cols.insert(0,'feature')
        df_coeff.reset_index(inplace=True)
        df_coeff.columns = cols
        df_coeff.set_index('feature',inplace=True)
        d_sign = self.settings["tsf-batch-settings"]["coeff-signs"]
        d_sign_ops = {"positive": (lambda x: x > 0),"negative": (lambda x: x <0)}
        for k,features in d_sign.items():
            for feature in features:
                if d_sign_ops[k](df_coeff.loc[feature,'coeff'])==False:
                    return False
        return True
    def scan_coeff(self,df):
        """
        scans coeff dataframe for significant coefficients
        """
        significant_x = []
        for index,row in df.iterrows():
            if row['pvals'] < 0.05 and abs(row['coeff']) > 1.96:
                significant_x.append(index)
        return significant_x
    @staticmethod
    def forecast_accuracy(forecast, actual):

        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        rmse = np.mean((forecast - actual)**2)**.5
        return {'mape':mape,'rmse':rmse}

    @staticmethod
    def forecast_accuracy_train(residual,actual):
        try:
            mape_train = np.mean(np.abs(residual )/np.abs(actual)) * 100
            rmse_train = np.mean((residual ) ** 2) ** .5
        except Exception as err:
            return {'mape': err, 'rmse': err}

        return {'mape': mape_train, 'rmse': rmse_train}


    def get_NPI_dataset(self,no_of_datapoints):
        indexed_on = self.settings["tsf-batch-settings"]["indexed on"]
        freq = self.settings["tsf-batch-settings"]["frequency-options"][self.settings["tsf-batch-settings"]["frequency"]]
        try:
            df_NPI = self.data[-1:]
            dt = str(list(df_NPI.index)[0]).split("-")
            # df_NPI.to_csv("tdf_NPI.csv")
            # df_NPI.reset_index(inplace=True)
            # dt = str(df_NPI[indexed_on].values[0][0]).split("-")
            dts = f"{dt[0]}-{dt[1]}-{dt[2][:2]}"
            dt_range = pd.date_range(dts,periods=no_of_datapoints,freq=freq)
            df_NPI = df_NPI.loc[df_NPI.index.repeat(no_of_datapoints)].reset_index(drop=True)
            df_NPI[indexed_on] = dt_range
            df_NPI.set_index(indexed_on,inplace=True)
        except Exception as err:
            self.logger.error(f"NPI dataset: {err}")
        return df_NPI
    def get_dataset_with_feature_value_increased(self,df,feature):
        try:
            key_feature = 'RPRICE' if 'price' in feature.lower() else 'GDP_PER_CAPITA'
            d_fvc = self.settings["tsf-batch-settings"]["elasticity-feature-increase-by-percent"] # dict of feature value incr
            fvc = 1+d_fvc[key_feature]/100
            tdf = df.copy()
            # tdf.to_csv('npi growth dataset nestle.csv')
            tdf[key_feature][1:] = tdf[key_feature][1:].apply(lambda x: x*fvc)
            return tdf
        except Exception as err:
            self.logger.error(f"dataset feature value increase: {err}")

    def transform_trend(self,df):
        increaseby1 = lambda x,i: x + i
        decreaseby1 = lambda x,i: x - i
        get_squared = lambda x,i: (x+i)**2 
        _dtransform = {
            "increase-by-one":increaseby1,
            "decrease-by-one":decreaseby1,
            "squared":get_squared,
            }
        try:
            trend_var = self.settings["tsf-batch-settings"]["elasticity-trend"]
            _transform = self.settings["tsf-batch-settings"]["elasticity-trend-transformation"]

            tdf = df.copy()
            tdf.reset_index(inplace=True)
            for i,r in tdf.iterrows():
                val = r[trend_var]
                tdf.at[i,trend_var] = _dtransform[_transform ](val,i)
            tdf.set_index(self.settings["tsf-batch-settings"]["indexed on"],inplace=True)
            #tdf.to_csv('npi-dataset-nestle-after-transformation.csv')
            return tdf
        except Exception as err:
             self.logger.error(f"transform trend: {err}")
        

    def get_elasticity_dataset_prepared(self):
        try:
            no_of_rows = self.settings["tsf-batch-settings"]["elasticity-dataset-no-rows"]
            self.dataset_NPI = self.get_NPI_dataset(no_of_rows)
            if self.settings["tsf-batch-settings"]["elasticity-trend"] != "No Trend":
                self.dataset_NPI = self.transform_trend(self.dataset_NPI)
            self.dataset_price = self.get_dataset_with_feature_value_increased(self.dataset_NPI,'Price')
            self.dataset_gdp = self.get_dataset_with_feature_value_increased(self.dataset_NPI,'GDP')
        except Exception as err:
            self.logger.error(f"elasticity datasets preparation: {err}")
    def get_price_and_income_elasticities(self,model,x_list,pdq):
        try:
            """
            this is forecasting beyong test set - to fill up the gap between train and
            new datasets (NPI etc), we should add test to xvars to these
            """
            df_NPI = pd.concat([self.test_data[:-1],self.dataset_NPI])
            df_price = pd.concat([self.test_data[:-1],self.dataset_price])
            df_gdp = pd.concat([self.test_data[:-1],self.dataset_gdp])
            # df_NPI.to_csv("NPIx.csv")
            # df_price.to_csv("pricex.csv")
            # df_gdp.to_csv('gdpx.csv')
            #forecasts
            forecasts_NPI = self.get_forecast_dfx(model, df_NPI,x_list)
            forecasts_price = self.get_forecast_dfx(model, df_price,x_list)
            forecasts_gdp = self.get_forecast_dfx(model,df_gdp,x_list)
            #forecasts_NPI.to_excel("forecasts_NPI.xlsx")
            # xdf = pd.DataFrame({'npi':forecasts_NPI,'price':forecasts_price,'gdp':forecasts_gdp})
            # for k in xdf.columns:
            #     xdf[k] = xdf[k].apply(lambda x: f"{x:0.6f}")
            #     xdf.to_csv(f'forecastsx-{pdq}.csv')
            """
            filter based on the first index of NPI dataset to get population and actual volume
            """
            no_of_rows = self.settings["tsf-batch-settings"]["elasticity-dataset-no-rows"]
            df_pop = self.df_data[self.df_data.index.isin([self.dataset_NPI.index[0]])].reset_index() #.T.reset_index().to_dict#()

            self.tot_pred_vol_NPI1 = (forecasts_NPI*df_pop['Population'].values[0])
            #self.logger.info(f"forecasted volume NPI:{tot_pred_vol_NPI1}")
            tot_pred_vol_NPI =self.tot_pred_vol_NPI1.tail(no_of_rows).sum()
            #self.logger.info(f"sum total forcasted vol-no change:{tot_pred_vol_NPI}")
            #xdf.to_csv("Forecasted_Volume_NPI.csv")
            # print(f"predicted NPI Vol {forecasts_NPI} {tot_pred_vol_NPI}")

            # price elasticity        
            self.tot_pred_vol_price1 = (forecasts_price*df_pop['Population'].values[0])
            tot_pred_vol_price=self.tot_pred_vol_price1.tail(no_of_rows).sum()
            #self.logger.info(f"sum total forcasted vol-price change:{tot_pred_vol_price}")
            vol_change_price = (tot_pred_vol_price - tot_pred_vol_NPI)/np.array([tot_pred_vol_NPI,tot_pred_vol_price]).mean()
            self.logger.info(f" vol-price change:{vol_change_price}")
            prices = self.dataset_price['RPRICE'].values
            price_change = (prices[1]-prices[0])/np.array([prices[0],prices[1]]).mean()
            price_elasticity = vol_change_price/price_change

            # income elasticity
            self.tot_pred_vol_income1 = (forecasts_gdp*df_pop['Population'].values[0])
            tot_pred_vol_income=self.tot_pred_vol_income1.tail(no_of_rows).sum()
            vol_change_income = (tot_pred_vol_income - tot_pred_vol_NPI)/np.array([tot_pred_vol_NPI,tot_pred_vol_income]).mean()
            gdps = self.dataset_gdp['GDP_PER_CAPITA'].values
            income_change = (gdps[1]-gdps[0])/np.array([gdps[0],gdps[1]]).mean()
            income_elasticity = vol_change_income/income_change
            self.logger.info(f"price elasticity {price_elasticity} income elasticity {income_elasticity}")
            return (price_elasticity,income_elasticity)
        except Exception as err:
            self.logger.error(f"price and income elasticities: {err}")

    def forecast_volume_Elasticity(self):

        xdf = pd.DataFrame({'Forecasted Volume - No Change': self.tot_pred_vol_NPI1})
        ydf = pd.DataFrame({'Forecasted Volume with Price Increase': self.tot_pred_vol_price1})
        zdf = pd.DataFrame({'Forecasted Volume with GDP Increase': self.tot_pred_vol_income1})
        Forecasted_Volume=pd.concat([xdf, ydf , zdf], axis=1)
        Forecasted_Volume= Forecasted_Volume.reset_index()
        Forecasted_Volume.rename(columns={'index': 'Forecasted Time'}, inplace=True)
        return(Forecasted_Volume)

    def forecast_volume_test(self):
        Forecasted_Volume_Test = self.forecast_test
        tdf=(self.test_data).copy()
        tdf["Predicted per Capita Volume"] = Forecasted_Volume_Test
        tdf["Percentage Error - Test"]=(tdf[self.dep_var ]-tdf["Predicted per Capita Volume"])*100/tdf[self.dep_var ]

        return tdf
    def forecast_volume_train(self):
        residual = pd.DataFrame({'Residuals': self.resi})
        #residual.to_excel("Residual.xlsx")
        Data = self.train_data.copy()
        new_df = pd.concat([Data, residual], axis=1)
        new_df["Predicted per Capita Volume"]=new_df[self.dep_var ]-new_df['Residuals']
        new_df["Percentage Error - Train"]=new_df['Residuals']*100/new_df[self.dep_var ]

        return new_df


    def get_NPI_growth(self,model,x_list,pdq):
        try:
            npi_dataset = pd.read_excel(self.settings["tsf-batch-settings"]["NPI-growth-dataset"])
            npi_dataset.set_index(self.settings["tsf-batch-settings"]["indexed on"],inplace=True)
            freq = self.settings["tsf-batch-settings"]["frequency"]
            npi_dataset.index.freq = self.settings["tsf-batch-settings"]["frequency-options"][freq]
            vol_col = self.settings["tsf-batch-settings"]["Actual-vol-column"]
            no_of_row = self.settings["tsf-batch-settings"]["NPI-Growth-dataset-no-rows"]
            actutal_vol_dataset = npi_dataset[:no_of_row]
            npi_pred_dataset = npi_dataset[no_of_row:]
            npi_pred_dataset = pd.concat([self.test_data,npi_pred_dataset])
            #npi_pred_dataset.to_csv('npi-pred-ds.csv')
            #print(x_list)
            npi_preds = self.get_forecast_dfx(model, npi_pred_dataset,x_list)
            # tdf_preds = pd.DataFrame({'predicted vol':npi_preds})
            # tdf_preds.to_csv(f"pred vol {pdq}.csv")

            tot_pred_vol = npi_preds.tail(no_of_row) * npi_pred_dataset['Population'].tail(no_of_row)
            tdf_NPIGr = pd.DataFrame({'Forecasted_Volume_NPIGr': tot_pred_vol})

            # tot_pred_vol_NPIGr.to_csv(f"total pred vol {pdq}.csv")
            npi_growth_prcnt = ((tot_pred_vol.sum() / actutal_vol_dataset[vol_col].sum()) - 1) * 100.0
            return npi_growth_prcnt, tdf_NPIGr
        except Exception as err:
            self.logger.error(f"NPI-growth: {err}")
            return "check log"

if __name__=='__main__':
    d = {'project_setting_file':'forecast_and_elasticiy.afefx'}
    tsb = TSFBatch(**d)
    models = tsb.run_models()
    # df = tsb.get_NPI_dataset(5)
    # df.to_csv("NPI DATASET.CSV")
    # dft = tsb.get_incremental_change_by1(df)
    # dft.to_csv("trend new dataset.csv")
    # df1 = tsb.get_dataset_with_feature_value_increased(df,'Price')
    # df1.to_csv("price-dataset.csv")
    # df2 = tsb.get_dataset_with_feature_value_increased(df,'GDP')
    # df2.to_csv("gdp-dataset.csv")

    # _d = {'p_max':8,'d_max':2,'q_max':8}
    # print(len(tsb.get_combinations_for_pdq(**_d)))

    # print("Top 10 models:\n")
    # for model in models[:10]:
    #     print(f"model: {model.model} AIC: {model.AIC} and MAPE: {model.MAPE}\n")
