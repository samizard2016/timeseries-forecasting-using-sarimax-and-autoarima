"""
batch sarimax to generate a batch of models based on selection criteria
"""
import logging
import pandas as pd
class BSARIMAX:
    def __init__(self,**kwargs):
        self.logger = logging.getLogger("anuman")
        self.keys = {}
        self.keys.update(kwargs)
    def parse_results(self):
        #results,data, endog, exog)  :
        #Prediction on data given endog and exog
        try:
            results = self.keys['result']
            endog = self.keys['endog']
            exog = self.keys['exog']
            self.dependent_var = self.keys['dependent_var']
            self.data = self.keys['data']
            self.prediction = pd.DataFrame(results.predict(start = endog.index[0],
                                        end = endog.index[-1],
                                        exog = exog[exog.columns],
                                        dynamic = False))
            self.pcname = f'predicted value- y({endog.columns[0]})-x{list(exog.columns)}'
            self.prediction.columns = [self.pcname]
            # predict = pd.concat([predict_test,predict_train])
            # self.prediction.to_excel("prediction.xlsx")

            #Collecting Results from ARIMA fit
            df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
            df_ = pd.read_html(results.summary().tables[0].as_html())[0]
            df.columns = ['Coefficient','Std Err','z','p>z','CI_up','CI_down']
            df_.columns = ['measure','value','measure','value']
            df_ = df_.iloc[:,:2].append(df_.iloc[:,2:])
            df_.index = list(range(len(df_)))
            df_.index = df_.pop('measure')
            self.df_results = df_.copy()
            # from datetime import datetime
            # t = datetime.now()
            # self.df_results.to_excel(f'result_{t}.xlsx')
            return pd.DataFrame({'x':[1,2,3]})
        except Exception as err:
            self.logger.error(f"problem in batch processing.parse_results: {err}")
            return f"problem in batch processing.parse_results: {err}"



    def get_prediction_details(self):
        # **computing MAPE, r2, pval , tval, standardized coefficients**
        dep = self.dependent_var
        self.df_compare = self.data[dep].to_frame().merge(self.prediction, left_index = True, right_index = True)
        percnt = (self.df_compare[dep] - self.df_compare[self.pcname])/self.df_compare[dep]*100
        self.ape = abs(percnt)
        self.max_ape = max(list(self.ape))
        self.mape = self.ape.mean()
        covar = ((cov(self.df_compare[dep],self.df_compare[self.pcname])**2)/
            (cov(self.df_compare[dep],self.df_compare[dep])*
            cov(self.df_compare[self.pcname], self.df_compare[self.pcname])))[0,1]

        self.r2 = math.sqrt(covar)

        neg_err = list(percnt+self.ape).count(0)
        pos_err = len(percnt) - neg_err
        pos_err_ratio = pos_err/len(percnt)*100
        neg_err_ratio = neg_err/len(percnt)*100
        if neg_err_ratio != 0 :
            err_ratio = pos_err_ratio/neg_err_ratio
        else :
            err_ratio = 0

        cof = self.df_results['Coefficient'].to_frame()
        df_ = pd.concat([self.df_results.iloc[:2],self.df_results.iloc[7:13]])
        tes = df_.copy()
        calc_stats = [max_ape,mape,pos_err_ratio,neg_err_ratio,r2]
        calc = pd.DataFrame(calc_stats)
        calc.index = ['max APE','MAPE','Pos/Total error','Neg/Total error', 'R-Squared']
        tes.columns = cof.columns
        calc.columns = cof.columns
        self.pred_details = pd.concat([cof,tes,calc])
        mname = 'y({})-x{}'.format(endog.columns[0],list(exog.columns))
        self.pred_details.columns = [mname]
        self.df_compare.to_excel("comparison.xlsx")
        self.pred_details.to_excel('pred_details.xlsx')
