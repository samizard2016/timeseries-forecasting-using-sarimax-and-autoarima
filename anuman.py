from msilib.schema import ComboBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pandas.io import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
from tarimax import TARIMAX
from statsmodels.stats.stattools import durbin_watson
import math
import sys
import os
import openpyxl
import datetime as dt
import json
import logging
import pprint as pp
from tsbatch import TSFBatch
import scipy.spatial.transform._rotation_groups, scipy.optimize
import seaborn as sns
import seaborn.matrix, seaborn.cm
from reg import Registry


matplotlib.use("Qt5Agg")
plt.style.use("dark_background")


class Anuman(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # start logging
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
            level=logging.INFO,
            filename="anuman.log",
        )
        release = "July 2023"
        self.logger = logging.getLogger("anuman")
        """DEBUG INFO WARNING ERROR CRITICAL"""
        self.logger.info("A new session starts")
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        self.main_widget.bottom_widget.tabWidget.tabBar().setVisible(False)       
        self.showMaximized()
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        self.setWindowTitle(f"Anuman 1.4.2 - Macro-economic Time-series Forecast Modeling, {release}")
        self.setWindowIcon(QIcon("anuman.png"))
        self.statusBar = QStatusBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Status:")
        self.statusBar.setStyleSheet("color: white;")
        self._auto_arima = True
        self.load_afefx_settings("forecast_and_elasticity.afefx")
        self.main_widget.bottom_widget.afefx_settings = self.afefx_settings     
        if self.check_reg():
            self.update_status("ready to go")
            self.handle_interactions()
            self.logger.info(f"Registration check has passed")
        else:
            self.update_status(f"Registration failed. If the registration wasn't done yet, \
you need to send the registration file to samir.paul@kantar.com")
            self.logger.error(f"Registration check has failed")
    def check_reg(self):
        self.reg_file = f"{os.environ['COMPUTERNAME']}.bin"
        if os.path.isfile(self.reg_file):
            self.reg = Registry.restore(self.reg_file) 
            if self.reg.check():
                return True
            else:
                try:
                    self.logger.info(self.reg.d_mac)
                    self.reg = Registry(**{"expiry_date":str(dt.date.today()),"expired": "yes"})
                    self.reg.update()
                except Exception as err:
                    self.logger.error(f"couldn't create the reg file {self.reg_file}: {err}")
        else:
            self.reg = Registry(**{"expiry_date":str(dt.date.today()),"expired": "yes"})
            self.reg.d_mac['machine_name'] = self.reg_file.split(".")[0]
            self.reg.update()
        return False

    def handle_interactions(self):
        self.main_widget.top_widget.Home.clicked.connect(self.show_Home)
        self.main_widget.top_widget.Setup.clicked.connect(self.show_Setup)
        self.main_widget.top_widget.ds.clicked.connect(self.show_datasetView)
        self.main_widget.top_widget.dashboard.clicked.connect(self.show_dashboard)
        self.main_widget.bottom_widget.dataset.clicked.connect(self.select_dataset)
        self.main_widget.bottom_widget.list_key_vars.itemSelectionChanged.connect(
            self.load_items_for_dv
        )
        self.main_widget.bottom_widget.cb_dependent_var.clicked.connect(
            self.load_items_for_dv
        )
        self.main_widget.bottom_widget.cb_dependent_var_afefx.clicked.connect(
            self.load_items_for_dv_afefx
        )
        self.main_widget.bottom_widget.cb_auto_arima.clicked.connect(
            self.on_click_auto_arima
        )
        self.main_widget.bottom_widget.cb_seasonality.clicked.connect(
            self.on_click_seasonality
        )
        self.main_widget.top_widget.run.clicked.connect(self.on_run)     
        self.main_widget.top_widget.afefx_start.clicked.connect(self.start_afefx)
        self.main_widget.top_widget.tsfefx.clicked.connect(self.show_afefx_setup)
        # afefx
        self.main_widget.bottom_widget.dataset_afefx.clicked.connect(self.select_dataset_afefx)
        self.main_widget.bottom_widget.cb_actual_volume_var.currentTextChanged.connect(self.update_list_fix_vars)
        self.main_widget.bottom_widget.NPI_dataset.clicked.connect(self.select_NPI_dataset)
        self.main_widget.bottom_widget.btn_open_settings_afefx.clicked.connect(self.main_widget.bottom_widget.restore_settings_afefx)
        self.main_widget.bottom_widget.btn_save_settings_afefx.clicked.connect(self.save_settings_afefx_default)
        self.main_widget.bottom_widget.btn_save_as_settings_afefx.clicked.connect(self.save_as_afefx_settings)
        self.main_widget.bottom_widget.cb_transformation_var.currentTextChanged.connect(self.enable_add_transformation)
        self.main_widget.bottom_widget.btn_add_transformation.clicked.connect(self.add_transformation)
    def enable_add_transformation(self):
        if self.main_widget.bottom_widget.cb_transformation_var.currentText() != "No Transformation":
            self.main_widget.bottom_widget.btn_add_transformation.setVisible(True)
            #self.main_widget.bottom_widget.btn_add_transformation.clicked.connect(self.add_transformation)
        
    def add_transformation(self):
        try:
            trans_var = self.main_widget.bottom_widget.cb_transformation_var.currentText()
            trans_type = self.main_widget.bottom_widget.cb_transformation_type.currentText()
            self.main_widget.bottom_widget.list_selected_transformation.addItem(f"{trans_var} - {trans_type}")  
            self.main_widget.bottom_widget.list_selected_transformation.update()
            #self.logger.info(f"selected tansform variables: {}")
        except Exception as err:
            self.logger.error(f"var_selected_for_transformation: {err}")       

    def save_as_afefx_settings(self):
        fd = QFileDialog(self)
        cwd = os.getcwd()
        fileName, _ = fd.getSaveFileName(self,
            f"Save AFEFX selections",
            cwd,
            "AFEFX (*.afefx)","",QFileDialog.DontUseNativeDialog)
        if fileName == "": return None
        try:
            self.save_settings_afefx(fileName)
        except Exception as err:
            self.logger.error(f"save_afefx_selections: {err}")
            self.update_status(err)
   
    def save_settings_afefx_default(self):
        self.save_settings_afefx(file = "forecast_and_elasticity.afefx")
    def save_settings_afefx(self,file):
        try:
            self.main_widget.bottom_widget.update_settings()
            self.main_widget.bottom_widget.write_to_json(
                file,
                self.main_widget.bottom_widget.afefx_settings
                )
            # self.main_widget.top_widget.afefx_start.setEnabled(True)
            self.message_box(f"{file} has been saved")
            self.on_tsfefx(file)
        except Exception as err:
            self.logger.error(f"save_settings_afefx: {err}")
    def load_afefx_settings(self,json_file_name):
        try:
            with open(json_file_name) as f:
                self.afefx_settings = json.load(f)
        except Exception as err:
            self.logger.error(f"problem in reading the TSFBatch settings: {err}")

    def start_afefx(self):
        self.update_status("Forecast-Elasticity is being processed...")
        msg = self.tsfe_fx.run_models()
        self.update_status(msg)

    def on_tsfefx(self,setting_file):
        # setting_file = self.select_afefx()
        d = {"project_setting_file": setting_file}
        self.tsfe_fx = TSFBatch(**d)
        self.main_widget.top_widget.afefx_start.setEnabled(True)
        self.update_status(
            f"Time-series Forecast-Elasticity Framework is ready to process"
        )

    def on_run(self):
        try:
            self.prepare_dataset()
            # if self.main_widget.bottom_widget.cb_auto_arima.isChecked():
            #     self.update_auto_arima()
            # else:
            #     self.main_widget.bottom_widget.auto_arima()
            self.update_sarimax()
            # self.main_widget.bottom_widget.cmChart()
            # self.main_widget.bottom_widget.enable_model_dashboard()
            self.main_widget.bottom_widget.dashboard()
            self.main_widget.bottom_widget.tabWidget.setCurrentIndex(2)
            self.main_widget.bottom_widget.tabWidgetDashboard.repaint()
        except Exception as err:
            self.logger.error(f"problem in on_run: {err}")

    def update_auto_arima(self):
        try:
            _d = {"start_p": 0, "start_q": 0}
            model_summary = self.tarimax.auto_arima(**_d)
            self.main_widget.bottom_widget.auto_arima_summary.setText(
                str(model_summary)
            )
            # self.auto_arima_forecast()
            # self.forecast_using_auto_arima_n_periods()
        except Exception as err:
            self.logger.error(f"update auto arima: {err}")

    # def update_arima(self):
    #     try:
    #         model_summary = self.tarimax.ARIMA(
    #                 **{"p":self.main_widget.bottom_widget.arima_p.value(),
    #                     "d": self.main_widget.bottom_widget.arima_d.value(),
    #                     "q": self.main_widget.bottom_widget.arima_q.value()
    #                     }
    #                 )
    #
    #         self.main_widget.bottom_widget.arima_summary.clear()
    #         self.main_widget.bottom_widget.arima_summary.setText(str(model_summary.summary()))
    #     except Exception as err:
    #         self.logger.error(f"update_arima: {err}")
    def update_sarimax(self):
        try:
            if self.main_widget.bottom_widget.cb_seasonality.isChecked():
                model_summary = self.tarimax.SARIMAX(
                    **{
                        "p": self.main_widget.bottom_widget.arima_p.value(),
                        "d": self.main_widget.bottom_widget.arima_d.value(),
                        "q": self.main_widget.bottom_widget.arima_q.value(),
                        "seasonal": True,
                        "P": self.main_widget.bottom_widget.season_sarimax_P.value(),
                        "D": self.main_widget.bottom_widget.season_sarimax_D.value(),
                        "Q": self.main_widget.bottom_widget.season_sarimax_Q.value(),
                        "m": self.main_widget.bottom_widget.season_sarimax_m.value(),
                    }
                )
            else:
                model_summary = self.tarimax.SARIMAX(
                    **{
                        "p": self.main_widget.bottom_widget.arima_p.value(),
                        "d": self.main_widget.bottom_widget.arima_d.value(),
                        "q": self.main_widget.bottom_widget.arima_q.value(),
                        "seasonal": False,
                    }
                )
            dw_test=self.tarimax.durbin_watson_test(model_summary.resid)
            r_square_value = self.tarimax.get_RSquared(model_summary.resid)
            tt_dwt = self.get_tooltip_dwt(dw_test)
            self.main_widget.bottom_widget.sarimax_summary.setToolTip(tt_dwt)


            self.main_widget.bottom_widget.sarimax_summary.clear()
            self.main_widget.bottom_widget.sarimax_summary.setText(f"{model_summary.summary()}\n\nDurbin Watson Test: {dw_test}\n\nRSquared: {r_square_value}")


        except Exception as err:
            self.logger.error(f"update_sarimax: {err}")

    def get_tooltip_dwt(self, dwt):
        if dwt < 2:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are positively correlated"
        elif dwt == 2:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are not correlated"
        else:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are negatively correlated"


    def update_corr_heatmap(self):
        try:
            #     self.cmCanvas.axes.cla()
            # self.main_widget.bottom_widget.cmCanvas = Canvas(width=20,height=10)
            # self.main_widget.bottom_widget.cmCanvas.axes = self.cmCanvas.fig.add_subplot(1,1,1)
            # self.main_widget.bottom_widget.cmCanvas.axes.grid(False)
            # self.main_widget.bottom_widget.cmCanvas.axes.axis("off")
            corr = self.tarimax.data.corr()
            self.main_widget.bottom_widget.cmCanvas.axes.cla()
            # self.main_widget.bottom_widget.cmCanvas.axes.legend('off')
            sns.heatmap(
                corr,
                cmap="Reds",
                annot=True,
                ax=self.main_widget.bottom_widget.cmCanvas.axes,
            )
            self.main_widget.bottom_widget.cmCanvas.repaint()
        except Exception as err:
            self.logger.error(f"update_corr_heatmap: {err}")

    # def update_auto_arima(self):
    #     try:
    #         _d = {
    #             "start_p": 0,
    #             "start_q": 0
    #             }
    #         model_summary = self.tarimax.auto_arima(**_d)
    #         self.main_widget.bottom_widget.auto_arima_summary.clear()
    #         self.main_widget.bottom_widget.auto_arima_summary.setText(str(model_summary.summary()))
    #     except Exception as err:
    #         self.logger.error(f"update_auto_arima: {err}")

    def on_click_auto_arima(self):
        self._auto_arima = self.main_widget.bottom_widget.cb_auto_arima.isChecked()

    def on_click_seasonality(self):
        try:
            self.tarimax.seasonality = (
                self.main_widget.bottom_widget.cb_seasonality.isChecked()
            )
        except Exception as err:
            self.update_status(
                f"No dataset has been set yet - select key variables and others for a SARIMAX Model first"
            )
        # self.message_box(f"seasonality is {self.tarimax.seasonality}")

    def select_afefx(self):
        try:
            file = self.get_file(
                **{
                    "title": "Select the forecast-elasticity framework setting file",
                    "f_type": "AFEFX (*.afefx)",
                }
            )
            self.update_status(f"{file} has been successfully loaded")
        except Exception as err:
            self.logger.error(
                f"Problem in opening forecasting and elasticity framework"
            )
        return file
    
    def select_NPI_dataset(self):
        file = self.get_file(
            **{"title": "Select the dataset", "f_type": "XLSX (*.XLSX)"}
        )
        if file:
            self.main_widget.bottom_widget.NPI_dataset.setText(file)

    def select_dataset(self):
        file = self.get_file(
            **{"title": "Select the dataset", "f_type": "XLSX (*.XLSX)"}
        )
        if file:
            self.main_widget.bottom_widget.dataset.setText(file)
            self.main_widget.bottom_widget.dataset.setToolTip(file)
            list_vars = self.get_variables_from_csv(file)
            self.load_listWidget(
                list_vars, self.main_widget.bottom_widget.list_key_vars, True
            )
            # clear previous selections
            self.main_widget.bottom_widget.cb_date_var.clear()
            self.main_widget.bottom_widget.cb_date_var.addItems(list_vars)
            self.main_widget.bottom_widget.cb_date_var.adjustSize()
            index = self.get_index_of_date_var(list_vars)
            self.main_widget.bottom_widget.cb_date_var.setCurrentIndex(index)
            self.load_settings()
            self.main_widget.top_widget.ds.setEnabled(True)
            self.main_widget.top_widget.run.setEnabled(True)
            self.main_widget.top_widget.dashboard.setEnabled(True)
  
    def select_dataset_afefx(self):
        _d = {"title":"Select the dataset",
            "f_type": "XLSX (*.XLSX)",
            "dataset": self.main_widget.bottom_widget.dataset_afefx,
            "date_var_widget": self.main_widget.bottom_widget.cb_date_var_afefx,
            "list_widget_for_key_vars": self.main_widget.bottom_widget.list_key_var_afefx, 
            "cb_frequency_options": self.main_widget.bottom_widget.cb_period_afefx,
            "cb_transformation_type": self.main_widget.bottom_widget.cb_transformation_type 
            }
        self.select_dataset_extended(**_d)
        self.load_afefx_settings("forecast_and_elasticity.afefx")

    def select_dataset_extended(self,**kwargs):
        _d = {}
        _d.update(kwargs)
        file = self.get_file(
            **{"title": _d['title'], "f_type": _d['f_type']}
        )
        if file:
            try:
                _d['dataset'].setText(file)
                _d['dataset'].setToolTip(file)
                list_vars = self.get_variables_from_csv(file)
                self.load_listWidget(
                    list_vars, _d['list_widget_for_key_vars'], True
                )
                # clear previous selections
                _d['date_var_widget'].clear()
                _d['date_var_widget'].addItems(list_vars)
                _d['date_var_widget'].adjustSize()
                index = self.get_index_of_date_var(list_vars)
                _d['date_var_widget'].setCurrentIndex(index)
                self.reset_data_transformation_selections(list_vars)   
                _d['cb_frequency_options'].addItems(self.afefx_settings['tsf-batch-settings']["frequency-options"])  
                _d['cb_transformation_type'].addItems(self.afefx_settings['tsf-batch-settings']["transformation-options"])  
            except Exception as err:
               self.logger.error(f"select_dataset_extended: {err}")

    def reset_data_transformation_selections(self,list_vars):        
        # trend variable
        self.main_widget.bottom_widget.cb_trend.clear()
        var_list_trend = list_vars.copy()
        var_list_trend.insert(0,"No Trend")            
        self.main_widget.bottom_widget.cb_trend.addItems(var_list_trend)
        # clear transformation varaiable list
        self.main_widget.bottom_widget.cb_transformation_var.clear()
        self.main_widget.bottom_widget.cb_transformation_var.setCurrentText("No Transformation")
        self.main_widget.bottom_widget.btn_add_transformation.setVisible(False)
        var_list_transformation = list_vars.copy()
        var_list_transformation.insert(0,"No Transformation")
        self.main_widget.bottom_widget.cb_transformation_var.addItems(var_list_transformation)
        self.main_widget.bottom_widget.NPI_dataset.clear()
        self.main_widget.bottom_widget.NPI_dataset.setText("Select NPI Growth Dataset")
        self.main_widget.bottom_widget.list_selected_transformation.clear()



    def prepare_dataset(self):
        try:
            file = self.main_widget.bottom_widget.dataset.text()
            date_var = self.main_widget.bottom_widget.cb_date_var.currentText()
            period = self.main_widget.bottom_widget.cb_period.currentText()
            dependent_var = self.main_widget.bottom_widget.cb_dependent_var.currentText()
            tt_ratio = self.main_widget.bottom_widget.train_test_ratio.value()
            _d = {
                "data_file_name": file,
                "date_var": date_var,
                "period": period,
                "dependent_var": dependent_var,
                "vars": self.list_of_selected_vars,
                "train_test_ratio": tt_ratio,
                "intercept":self.main_widget.bottom_widget.cb_intercept.isChecked()
            }
            self.tarimax = TARIMAX(**_d)
            self.tarimax.set_index()
            dep_var = self.tarimax.get_dependent_var()
            self.main_widget.bottom_widget.update_dependent_var(dep_var)
            self.main_widget.bottom_widget.update_data(self.tarimax.data)
            self.logger.info("dataset prepared successfully")
        except Exception as err:
            self.logger.error(f"prepare dataset: {err}")
    def load_items_for_dv_afefx(self):
        self.main_widget.bottom_widget.cb_dependent_var_afefx.clear()
        self.main_widget.bottom_widget.cb_actual_volume_var.clear()
        #self.main_widget.bottom_widget.cb_trend.clear()
        self.selected_vars_afefx = self.get_vars_from_listbox(
            self.main_widget.bottom_widget.list_key_var_afefx
        )
        if self.selected_vars_afefx["checked"]:
            self.main_widget.bottom_widget.cb_dependent_var_afefx.addItems(
                self.selected_vars_afefx["checked"]
            )
            self.main_widget.bottom_widget.cb_actual_volume_var.addItems(
                self.selected_vars_afefx["checked"]
            )          

    def update_list_fix_vars(self):
        dep_var = self.main_widget.bottom_widget.cb_dependent_var_afefx.currentText()
        actual_vol = self.main_widget.bottom_widget.cb_actual_volume_var.currentText()       
        try:
            selected_vars = self.selected_vars_afefx['checked']
            selected_vars.remove(dep_var)
            selected_vars.remove(actual_vol)
            self.main_widget.bottom_widget.list_fix_var.clear()
            self.load_listWidget(selected_vars,
            self.main_widget.bottom_widget.list_fix_var, True)
        except Exception as err:
            self.logger.warning(f"update_list_fix_vars: {err}")
        # print(f"dep_var {dep_var} selected_vars {selected_vars}")
     
         
    def load_items_for_dv(self):
        self.main_widget.bottom_widget.cb_dependent_var.clear()
        selected_vars = self.get_vars_from_listbox(
            self.main_widget.bottom_widget.list_key_vars
        )
        if selected_vars["checked"]:
            self.main_widget.bottom_widget.cb_dependent_var.addItems(
                selected_vars["checked"]
            )
            self.list_of_selected_vars = selected_vars["checked"]
            if len(selected_vars["checked"]) > 2:
                self.main_widget.bottom_widget.lab_exogeneous.setVisible(True)
                self.main_widget.bottom_widget.exogeneous.setVisible(True)
            else:
                self.main_widget.bottom_widget.lab_exogeneous.setVisible(False)
                self.main_widget.bottom_widget.exogeneous.setVisible(False)
            self.prepare_dataset()
        else:
            self.update_status(
                f"You need to select key variables including variable to forecast first"
            )

    def load_settings(self):
        try:
            with open("settings.json") as f:
                settings = json.load(f)
                frequency = settings["anuman-settings"]["frequency"]
                transformations = settings["anuman-settings"]["transformations"]
                # clear previous selections
                self.main_widget.bottom_widget.cb_period.clear()
                self.main_widget.bottom_widget.cb_period.addItems(frequency)
                self.main_widget.bottom_widget.cb_transformation_dv.clear()
                self.main_widget.bottom_widget.cb_transformation_dv.addItems(
                    transformations
                )
                # set default frequency
                index = settings["anuman-settings"]["freq-default-index"]
                self.main_widget.bottom_widget.cb_period.setCurrentIndex(index)
                self.train_test_ratio = settings["anuman-settings"][
                    "train-test split ratio"
                ]
                self.screen_res = settings["anuman-settings"]["screen-res"]
        except Exception as err:
            self.logger.error(f"problem in reading the anuman settings: {err}")

    def get_index_of_date_var(self, vars):
        try:
            for var in vars:
                if "date" in var.lower():
                    return vars.index(var)
        except Exception as err:
            self.logger.error(f"problem in finding date var: {err}")
        return 0

    def get_file(self, **kwargs):
        title = kwargs["title"]
        file_type = kwargs["f_type"]  # "CFG (*.cfg)"
        cwd = os.getcwd()
        ofd = QFileDialog(self)
        openFileName = ofd.getOpenFileName(
            self, title, cwd, file_type, "", QFileDialog.DontUseNativeDialog
        )
        if openFileName != ("", ""):
            file = openFileName[0]
            self.logger.info(f"{file} has been selected")
            return file
        else:
            return None

    def load_key_vars(self):
        try:
            data_file = self.main_widget.bottom_widget.dataset.text()
            if data_file:
                list_vars = self.get_variables_from_csv(data_file)
                self.load_listWidget(
                    list_vars, self.main_widget.bottom_widget.list_key_vars
                )
                self.list_key_vars = list_vars
        except Exception as err:
            self.logger.error(f"problem in loading key vars: {err}")

    def get_variables_from_csv(self, file):
        try:
            df = pd.read_excel(file)
            heads = list(df.columns)
            return heads
        except Exception as err:
            self.logger.error(
                "problem in reading variables from the csv file: " + str(err)
            )
            return None

    def save_selections(self, file_to_save_in, d_selections):
        try:
            with open(file_to_save_in, "wb") as f:
                pickle.dump(d_selections, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            self.message_box(f"Problem is saving selections - {err}")

    def restore_selections(self, file_to_restore_from):
        try:
            with open(file_to_restore_from, "rb") as f:
                d_selections = pickle.load(f)
                return d_selections
        except Exception as err:
            self.message_box(f"Problem in restoring selections - {err}")

    def get_variables_from_ws(self, wb, ws):
        try:
            df = pd.read_excel(wb, sheet_name=ws)
            heads = df.columns
            return heads
        except Exception as err:
            self.logger.error("Get variables from ws : " + str(err))
            return None

    def load_listWidget(self, var_list, list_widget, checkable=True):
        list_widget.clear()
        # list_widget.setStyleSheet("scrollbar-width: thin;")
        for var in var_list:
            item = QListWidgetItem(var, list_widget)
            if checkable == True:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
        list_widget.repaint()

    def get_vars_from_listbox(self, listBox):
        vars_all = []
        vars_checked = []
        indices = []
        count = listBox.count()
        indx = 0
        for item in range(count):
            itm = listBox.item(item)
            vars_all.append(itm.text())
            if itm.checkState() == Qt.Checked:
                vars_checked.append(itm.text())
                indices.append(indx)
            indx += 1
        return {"all": vars_all, "checked": vars_checked, "indices": indices}

    def set_items_checked(self, listBox, indices_to_be_checked):
        for indx in indices_to_be_checked:
            item = listBox.item(indx)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
        listBox.repaint()

    def update_status(self, message):
        self.statusBar.showMessage(f"Status: {message}")
        self.statusBar.repaint()

    def show_Home(self):
        self.main_widget.bottom_widget.tabWidget.setCurrentIndex(0)
        self.update_status("")

    def show_afefx_setup(self):
        self.main_widget.bottom_widget.tabWidget.setCurrentIndex(4)
        self.update_status("")

    def clear_widgets(self):
        self.main_widget.bottom_widget.arima_summary.clear()
        self.main_widget.bottom_widget.w_aic.clear()

    def show_dashboard(self):
        try:
            self.prepare_dataset()

            # self.main_widget.bottom_widget.arima()
            if self.main_widget.bottom_widget.cb_auto_arima.isChecked():
                self.main_widget.bottom_widget.auto_arima()
                # self.main_widget.bottom_widget.auto_arima_forecast()
                # self.main_widget.bottom_widget.forecast_using_auto_arima_n_periods()

            self.main_widget.bottom_widget.sarimax()
        # self.main_widget.bottom_widget.durbin_watson_test()
            self.main_widget.bottom_widget.corr_matrix()
            # self.main_widget.bottom_widget.cmChart()
            self.main_widget.bottom_widget.acf_pacf()
            self.main_widget.bottom_widget.decompose()
            self.main_widget.bottom_widget.adft()
            self.main_widget.bottom_widget.seasonality()
            # self.main_widget.bottom_widget.arima_forecast()

            self.main_widget.bottom_widget.dashboard()
            self.main_widget.bottom_widget.tabWidget.setCurrentIndex(2)
            self.main_widget.bottom_widget.tabWidgetDashboard.repaint()
            self.main_widget.top_widget.dashboard.setEnabled(False)
            # self.main_widget.bottom_widget.cmChart()
            # self.main_widget.bottom_widget.update_auto_arima()

            # self.update_status("dataset has been successfully prepared")
            # self.update_aa()
            # self.main_widget.bottom_widget.refresh()
            # self.main_widget.bottom_widget.arima()
            # self.main_widget.bottom_widget.update()
        except Exception as err:
            self.logger.error(f"show_dashboard: {err}")

    def show_Setup(self):
        self.main_widget.bottom_widget.tabWidget.setCurrentIndex(3)
        self.update_status("")

    def show_datasetView(self):
        self.main_widget.bottom_widget.datasetView()
        self.main_widget.bottom_widget.tabWidget.setCurrentIndex(1)
        self.update_status("")

    @staticmethod
    def message_box(text_message):
        mb = QMessageBox()
        mb.setStyleSheet("background-color: #646665;color: white")
        mb.setIcon(QMessageBox.Information)
        mb.setWindowTitle("Anuman - Time-series Forecasting Tool")
        mb.setText(text_message)
        mb.setStandardButtons(QMessageBox.Ok)
        mb.exec()


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        self.top_widget = TopWidget(parent=self)
        self.bottom_widget = BottomWidget(parent=self)
        vlayout.addWidget(self.top_widget)
        vlayout.addWidget(self.bottom_widget)
        self.setLayout(vlayout)
        self.showMaximized()
        # self.setStyleSheet("background-color: #636963")


class TopWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.layout = QHBoxLayout(self)
        icon_size = QSize(50, 75)  # QSize(128,128) # if screen_res=='High' else
        self.Home = QPushButton("")
        self.Home.setToolTip("Anuman Home")
        self.Home.setIcon(QIcon("anuman.png"))
        self.Home.setIconSize(icon_size)
        self.Home.setStyleSheet(
            "border-width:5px;border-color: white;font-size: 10pt; height: 50px; width: 75px;background-color: #332F2F; color: red;margin: 0px;"
        )
        self.layout.addWidget(self.Home)

        # dataset
        self.ds = QPushButton("")
        self.ds.setToolTip("Anuman Dataset")
        self.ds.setIcon(QIcon("dataset.png"))
        self.ds.setIconSize(icon_size)
        self.ds.setStyleSheet(self.Home.styleSheet())
        self.layout.addWidget(self.ds)
        self.ds.setEnabled(False)

        # andaja dashboard
        self.dashboard = QPushButton("")
        self.dashboard.setToolTip("Anuman Dashboard")
        self.dashboard.setIcon(QIcon("dashboard.png"))
        self.dashboard.setIconSize(icon_size)
        self.dashboard.setStyleSheet(self.Home.styleSheet())
        self.layout.addWidget(self.dashboard)
        self.dashboard.setEnabled(False)
        self.run = QPushButton("")
        self.run.setToolTip("Run the model and Forecast")
        self.run.setIcon(QIcon("run.png"))
        self.run.setIconSize(icon_size)
        self.run.setStyleSheet(self.Home.styleSheet())
        self.layout.addWidget(self.run)
        self.run.setEnabled(False)

        # andaja settings
        self.Setup = QPushButton("")
        self.Setup.setToolTip("Anuman Settings")
        self.Setup.setIcon(QIcon("setup.png"))
        self.Setup.setIconSize(icon_size)
        self.Setup.setStyleSheet(self.Home.styleSheet())
        self.layout.addWidget(self.Setup)

        # forecat-elasticity fx
        self.tsfefx = QPushButton("")
        self.tsfefx.setToolTip("Anuman Forecast-Elasticity Workflow")
        self.tsfefx.setIcon(QIcon("tsfx.png"))
        self.tsfefx.setIconSize(icon_size)
        self.tsfefx.setStyleSheet(self.Home.styleSheet())
        self.layout.addWidget(self.tsfefx)

        # afefx start
        self.afefx_start = QPushButton("")
        self.afefx_start.setToolTip("Start Forecast-Elasticity Workflow")
        self.afefx_start.setIcon(QIcon("start afefx.png"))
        self.afefx_start.setIconSize(icon_size)
        self.afefx_start.setStyleSheet(self.Home.styleSheet())
        self.afefx_start.setEnabled(False)
        self.layout.addWidget(self.afefx_start)

        self.layout.addStretch(1)
        self.setStyleSheet("padding: 0px; margin: 0px; background-color: black;")
        self.setLayout(self.layout)


class RLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QLineEdit.mousePressEvent(self, event)


class RComboBox(QComboBox):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QComboBox.mousePressEvent(self, event)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=30, height=20, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(6, 3, 3)
        self.axes.axis("off")
        # self.fig, self.axes = plt.subplots(3,3,figsize=(width, height),dpi=dpi)
        super().__init__(self.fig)


class BottomWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        # self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#0f0f0f;}""")
        self.layout = QVBoxLayout(self)
        self.tabHome = QWidget()
        self.tabSetup = QWidget()
        self.tabDSView = QWidget()
        self.tabDashboard = QWidget()
        self.tabAFEFX = QWidget()
        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(self.tabHome, "Home")
        self.tabWidget.addTab(self.tabDSView, "DSV")
        self.tabWidget.addTab(self.tabDashboard, "Dashboard")
        self.tabWidget.addTab(self.tabSetup, "Setup")
        self.tabWidget.addTab(self.tabAFEFX,"afefx")
        self.layout.addWidget(self.tabWidget)
        self.layout.addStretch(1)
        # self.setStyleSheet("padding: 0px; margin: 0px; background-color: #4A4444;")
        self.setLayout(self.layout)
        self.logger = logging.getLogger("anuman")
        self.Home()
        self.Setup()
        self.afefx_setup()
        self.dependent_var = None
        self.cmCanvas = None
        self.arima_res = ""  # RQText("ARIMA Summary")
        self.arima_summary = QLabel("")
        self.sarimax_summary = QLabel("")
        self.auto_arima_summary = QLabel("")
        self.dashboard()
        self.afefx_settings = None
        self.flg_json_format = 1

    def Home(self):
        # self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabHome.setLayout(grid_layout)
        self.tabHome.setStyleSheet("border-color: red")
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # align grid to top and left

        # kantar_logo = QLabel(self)
        # pixmap = QPixmap('kantar.jpg')
        # kantar_logo.setPixmap(pixmap)
        # kantar_logo.setStyleSheet("margin: 10px")
        # # Optional, resize window to image size
        # self.resize(pixmap.width(),pixmap.height())
        # grid_layout.addWidget(kantar_logo,0,4,Qt.AlignTop | Qt.AlignRight)

        title = QLabel("Anuman - The Time series Forecasting Tool")
        # title.setStyleSheet("font-size: 14pt;color: red;text-align: center;")
        grid_layout.addWidget(title, 0, 0, 1, 7)
        sub_title = QLabel("Kantar Technology, South Asia")
        title.setStyleSheet("font-size: 12pt;color: white;text-align: center;")
        sub_title.setStyleSheet("font-size: 10pt;")
        grid_layout.addWidget(title, 0, 0, 1, 7)
        grid_layout.addWidget(sub_title, 1, 0, Qt.AlignTop | Qt.AlignLeft)

        empty_line = QLabel("")
        empty_line.setStyleSheet("width: 150px;")
        # str_dataset = RQText("Dataset")
        lab_dataset = QLabel("Dataset")

        self.dataset = RLineEdit("select the dataset")
        # Add the widgets to the layout
        self.dataset.setStyleSheet("font-size: 12pt; font-style: italic;width: 450px")

        self.key_vars = QLabel("Key Variables:")

        self.list_key_vars = ListWidget(self.tabHome)
        self.list_key_vars.setToolTip("Select the variables to include in the model")
        # self.list_key_vars.setStyleSheet("""
        #                                     QListWidget {width: 450 px;height: 300 px;background-color: #bac8e0;color: black;}
        #                                     QScrollBar:vertical {width: 40 px;margin: 3 px 0 3 px 0;}
        #                                     """)
        # self.list_key_vars.setStyleSheet("width: 450 px; height: 300 px;background-color: #bac8e0;color: black;")
        # self.list_key_vars.setStyleSheet("""QScrollBar:vertical {width: 40px;margin: 3px 0 3px 0;}""")
        lab_date = QLabel("Date Variable")
        self.cb_date_var = QComboBox(self.tabHome)
        self.cb_date_var.setFixedWidth(515)
        self.cb_date_var.setFixedHeight(60)
        lab_period = QLabel("Period")
        self.cb_period = QComboBox(self.tabHome)
        self.cb_period.setFixedWidth(515)
        self.cb_period.setFixedHeight(60)
        lab_dependent_var = QLabel("Variable to forecast")
        self.cb_dependent_var = RComboBox(self.tabHome)
        self.cb_dependent_var.setFixedWidth(515)
        self.cb_dependent_var.setFixedHeight(60)
        lab_df_test = QLabel("ADF Test")
        self.df_test = QCheckBox(self.tabHome)
        self.df_test.setChecked(True)
        lab_decomposition = QLabel("Seasonal Decomposition")
        self.seasonal_decomp = QCheckBox(self.tabHome)
        self.seasonal_decomp.setChecked(True)
        lab_acf_plot = QLabel("Plot ACF")
        self.acf_plot = QCheckBox(self.tabHome)
        self.acf_plot.setChecked(True)
        lab_pacf_plot = QLabel("Plot PACF")
        self.pacf_plot = QCheckBox(self.tabHome)
        self.pacf_plot.setChecked(True)
        lab_transformation = QLabel("Variable Transformation")
        self.cb_transformation_dv = RComboBox(self.tabHome)
        self.cb_transformation_dv.setFixedWidth(515)
        self.cb_transformation_dv.setFixedHeight(60)
        lab_ARIMA_order = QLabel("SARIMAX Order")
        lab_ARIMA_order.setStyleSheet("color:'black';font-size: 12pt")
        lab_auto_reg_order = QLabel("Auto Regressive Order (p)")
        lab_diff_order = QLabel("Difference Order (d)")
        lab_ma_order = QLabel("Moving Average Order (q)")
        self.arima_p = QSpinBox()
        self.arima_d = QSpinBox()
        self.arima_q = QSpinBox()
        lab_seasonal_order = QLabel("Seasonal Order:")
        lab_season_auto_reg_order = QLabel("Auto Regressive Order (P)")
        lab_season_diff_order = QLabel("Difference Order (D)")
        lab_season_ma_order = QLabel("Moving Average Order (Q)")
        lab_season_period = QLabel("Seasonal Period (M)")
        self.season_sarimax_P = QSpinBox()
        self.season_sarimax_D = QSpinBox()
        self.season_sarimax_Q = QSpinBox()
        self.season_sarimax_m = QSpinBox()
        self.season_sarimax_m.setRange(1, 12)
        self.season_sarimax_m.setValue(12)
        self.season_sarimax_m.setStyleSheet("width:60px;")
        self.season_sarimax_Q.setStyleSheet(self.season_sarimax_m.styleSheet())
        self.season_sarimax_P.setStyleSheet(self.season_sarimax_m.styleSheet())
        self.season_sarimax_D.setStyleSheet(self.season_sarimax_m.styleSheet())
        lab_auto_arima = QLabel("Auto ARIMA")
        lab_auto_arima.setStyleSheet(lab_ARIMA_order.styleSheet())
        self.cb_auto_arima = QCheckBox(self.tabHome)
        self.cb_auto_arima.setChecked(False)
        lab_seasonality = QLabel("Seasonality")
        lab_seasonality.setStyleSheet(lab_ARIMA_order.styleSheet())
        self.cb_seasonality = QCheckBox(self.tabHome)
        self.cb_seasonality.setChecked(False)
        lab_intercept = QLabel("Intercept")
        lab_intercept.setStyleSheet(lab_ARIMA_order.styleSheet())
        self.cb_intercept = QCheckBox(self.tabHome)
        self.cb_intercept.setChecked(False)
        self.lab_exogeneous = QLabel("Exogeneous")
        self.lab_exogeneous.setStyleSheet(lab_ARIMA_order.styleSheet())
        self.lab_exogeneous.setVisible(False)
        self.exogeneous = QCheckBox(self.tabHome)
        self.exogeneous.setChecked(True)
        self.exogeneous.setVisible(False)
        lab_forecast_N = QLabel("Forecast for N Periods")
        #lab_forecast_N.setStyleSheet(lab_ARIMA_order.styleSheet())
        lab_train_test  = QLabel("Train-Test Ratio")
        #lab_TT_ratio.setStyleSheet(lab_ARIMA_order.styleSheet())
        self.forecast_N = QSpinBox()
        selftrain_test_ratio= QSpinBox()
        self.train_test_ratio = QSpinBox()
        self.train_test_ratio.setRange(0, 100)
        self.train_test_ratio.setValue(80)
        self.train_test_ratio.setStyleSheet("width: 65px")
        # grid_layout.addWidget(empty_line,1,0)
        grid_layout.addWidget(lab_dataset, 2, 0, Qt.AlignRight)
        grid_layout.addWidget(self.dataset, 2, 1, Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(self.key_vars, 3, 0, Qt.AlignRight)
        grid_layout.addWidget(self.list_key_vars, 3, 1, Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(lab_date, 4, 0, Qt.AlignRight)
        grid_layout.addWidget(self.cb_date_var, 4, 1, Qt.AlignBottom | Qt.AlignLeft)
        grid_layout.addWidget(lab_period, 5, 0, Qt.AlignRight)
        grid_layout.addWidget(self.cb_period, 5, 1, Qt.AlignBottom | Qt.AlignLeft)
        grid_layout.addWidget(lab_dependent_var, 6, 0, Qt.AlignRight)
        grid_layout.addWidget(
            self.cb_dependent_var, 6, 1, Qt.AlignBottom | Qt.AlignLeft
        )
        grid_layout.addWidget(lab_df_test, 7, 0, Qt.AlignRight)
        grid_layout.addWidget(self.df_test, 7, 1, Qt.AlignCenter | Qt.AlignLeft)
        grid_layout.addWidget(lab_decomposition, 8, 0, Qt.AlignRight)
        grid_layout.addWidget(self.seasonal_decomp, 8, 1, Qt.AlignCenter | Qt.AlignLeft)
        grid_layout.addWidget(lab_acf_plot, 4, 2, Qt.AlignRight)
        grid_layout.addWidget(self.acf_plot, 4, 3, Qt.AlignCenter | Qt.AlignLeft)
        grid_layout.addWidget(lab_pacf_plot, 5, 2, Qt.AlignRight)
        grid_layout.addWidget(self.pacf_plot, 5, 3, Qt.AlignCenter | Qt.AlignLeft)
        grid_layout.addWidget(lab_transformation, 6, 2, Qt.AlignRight)
        grid_layout.addWidget(
            self.cb_transformation_dv, 6, 3, Qt.AlignBottom | Qt.AlignLeft
        )
        grid_layout.addWidget(lab_ARIMA_order, 9, 0, Qt.AlignRight)
        grid_layout.addWidget(lab_auto_reg_order, 10, 0, Qt.AlignRight)
        grid_layout.addWidget(lab_diff_order, 11, 0, Qt.AlignRight)
        grid_layout.addWidget(lab_ma_order, 12, 0, Qt.AlignRight)
        grid_layout.addWidget(self.arima_p, 10, 1, Qt.AlignLeft)
        grid_layout.addWidget(self.arima_d, 11, 1, Qt.AlignLeft)
        grid_layout.addWidget(self.arima_q, 12, 1, Qt.AlignLeft)
        grid_layout.addWidget(lab_seasonal_order, 9, 2, Qt.AlignLeft)
        grid_layout.addWidget(lab_season_auto_reg_order, 10, 2, Qt.AlignRight)
        grid_layout.addWidget(lab_season_diff_order, 11, 2, Qt.AlignRight)
        grid_layout.addWidget(lab_season_ma_order, 12, 2, Qt.AlignRight)
        grid_layout.addWidget(lab_season_period, 13, 2, Qt.AlignRight)
        grid_layout.addWidget(self.season_sarimax_P, 10, 3, Qt.AlignLeft)
        grid_layout.addWidget(self.season_sarimax_D, 11, 3, Qt.AlignLeft)
        grid_layout.addWidget(self.season_sarimax_Q, 12, 3, Qt.AlignLeft)
        grid_layout.addWidget(self.season_sarimax_m, 13, 3, Qt.AlignLeft)
        grid_layout.addWidget(lab_auto_arima, 7, 2, Qt.AlignRight)
        grid_layout.addWidget(self.cb_auto_arima, 7, 3, Qt.AlignLeft)
        grid_layout.addWidget(lab_seasonality, 10, 4, Qt.AlignRight)
        grid_layout.addWidget(self.cb_seasonality, 10, 5, Qt.AlignLeft)
        grid_layout.addWidget(lab_intercept, 11, 4, Qt.AlignRight)
        grid_layout.addWidget(self.cb_intercept, 11, 5, Qt.AlignLeft)
        grid_layout.addWidget(self.lab_exogeneous, 12, 4, Qt.AlignRight)
        grid_layout.addWidget(self.exogeneous, 12, 5, Qt.AlignLeft)
        grid_layout.addWidget(lab_forecast_N, 13, 0, Qt.AlignRight)
        grid_layout.addWidget(self.forecast_N, 13, 1, Qt.AlignLeft)
        grid_layout.addWidget(lab_train_test , 14, 0, Qt.AlignRight)
        grid_layout.addWidget(self.train_test_ratio, 14, 1, Qt.AlignLeft)

    def clearLayout(self, layout):
        for i in reversed(range(layout.count())):
            widgetToRemove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

    def update_dependent_var(self, dependent_var):
        self.dependent_var = dependent_var

    def update_data(self, data):
        self.data = data
    def afefx_setup(self):
        # self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabAFEFX.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman - Forecasts, Elasticity & NPI Growth Framework")
        title.setStyleSheet("font-size: 12pt;color: white;text-align: left;")
        sub_title = QLabel("Kantar Technology, South Asia")
        sub_title.setStyleSheet("font-size: 10pt;color: white;text-align: left;")
        grid_layout.addWidget(title, 0, 0, 1, 7, Qt.AlignLeft | Qt.AlignTop)
        grid_layout.addWidget(sub_title, 1, 0, 1, 7, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        grid_layout.addWidget(empy_line, 2, 0, 1, 7, Qt.AlignLeft | Qt.AlignTop)

        # Add the widgets to the layout
        lab_proj_des= QLabel("Project Description")
        self.proj_des = QLineEdit('Project description here')
        self.proj_des.setStyleSheet("font-size: 12pt; font-style: italic;width: 1250px")
        lab_dataset = QLabel("Dataset:")
        self.dataset_afefx = RLineEdit("select the dataset")
        self.dataset_afefx.setStyleSheet("font-size: 8pt; font-style: italic;width: 450px")
        lab_key_var = QLabel("Variables:")
        self.list_key_var_afefx = ListWidget(self.tabAFEFX)
        self.list_key_var_afefx.setToolTip("Select the variables to include in the Model")
        lab_fix_var = QLabel("Exogeneous Variables to be fixed")
        self.list_fix_var = ListWidget(self.tabAFEFX)
        self.list_fix_var.setFixedWidth(300)
        self.list_fix_var.setFixedHeight(80)
        lab_date = QLabel("Date Variable")
        self.cb_date_var_afefx = QComboBox(self.tabAFEFX)
        self.cb_date_var_afefx.setFixedWidth(515)
        self.cb_date_var_afefx.setFixedHeight(60)
        lab_period = QLabel("Period")
        self.cb_period_afefx = QComboBox(self.tabAFEFX)
        self.cb_period_afefx.setFixedWidth(515)
        self.cb_period_afefx.setFixedHeight(60)
        lab_dependent_var = QLabel("Variable to Forecast")
        self.cb_dependent_var_afefx = RComboBox(self.tabAFEFX)
        self.cb_dependent_var_afefx.setFixedWidth(515)
        self.cb_dependent_var_afefx.setFixedHeight(60)
        lab_sarimax_order = QLabel("SARIMAX Order")
        lab_sarimax_order.setStyleSheet("color:'black';font-size: 12pt")
        lab_pval = QLabel("Auto Regressive Order (p)")
        lab_dval = QLabel("Difference Order (d)")
        lab_qval = QLabel("Moving Average Order (q)")
        self.sb_pval = QSpinBox()
        self.sb_pval.setStyleSheet("width: 65px;")
        self.sb_dval = QSpinBox()
        self.sb_dval.setStyleSheet(self.sb_pval.styleSheet())
        self.sb_qval = QSpinBox()
        self.sb_qval.setStyleSheet(self.sb_pval.styleSheet())
        #seasonal order
        lab_seasonality_afefx = QLabel("Seasonality")
        lab_seasonality_afefx.setStyleSheet("color:'black';font-size: 12pt")
        self.cb_seasonality_afefx = QCheckBox(self.tabAFEFX)
        self.cb_seasonality_afefx.setChecked(False)
        lab_add = QLabel("Additive")
        lab_mul = QLabel("Multiplicative")        
        #self.cb_add = QCheckBox(self.tabAFEFX)
        self.radioButton_add = QRadioButton(self.tabAFEFX)        
        self.radioButton_mul = QRadioButton(self.tabAFEFX)
        self.radioButton_mul.setChecked(True)
        #Transformation
        lab_transformation_var = QLabel("Transformation variable")
        self.cb_transformation_var = QComboBox(self.tabAFEFX)
        self.cb_transformation_var.setFixedWidth(300)
        self.cb_transformation_var.setFixedHeight(50)

        lab_transformation_type = QLabel("Transformation Type")
        self.cb_transformation_type = QComboBox(self.tabAFEFX)
        self.cb_transformation_type.setFixedWidth(300)
        self.cb_transformation_type.setFixedHeight(50)

        self.btn_add_transformation = QPushButton("Add Transformation")
        self.btn_add_transformation.setStyleSheet("width: 150px; height: 50px;")
        self.btn_add_transformation.setVisible(False)

        lab_selected_transformation = QLabel("Selected Transformations")        
        self.list_selected_transformation = ListWidget(self.tabAFEFX)
        

       
        lab_autoarima = QLabel("Auto ARIMA")
        lab_autoarima.setStyleSheet(lab_sarimax_order.styleSheet())
        self.cb_autoarima = QCheckBox(self.tabAFEFX)

        lab_intercept = QLabel("Intercept")
        lab_intercept.setStyleSheet(lab_sarimax_order.styleSheet())
        self.cb_intercept_afefx = QCheckBox(self.tabAFEFX)

        lab_train_test = QLabel("Train-Test Ratio")
        self.train_test_ratio_afefx = QSpinBox()
        self.train_test_ratio_afefx.setRange(0, 100)
        self.train_test_ratio_afefx.setValue(80)
        self.train_test_ratio_afefx.setStyleSheet("width: 65px")
        lab_NPIdataset = QLabel("NPI growth dataset")
        self.NPI_dataset = RLineEdit("select NPI Growth dataset")
        self.NPI_dataset.setStyleSheet(self.dataset_afefx.styleSheet())
        lab_trend = QLabel("Elasticity Trend")
        self.cb_trend = RComboBox(self.tabAFEFX)
        self.cb_trend.setFixedWidth(515)
        self.cb_trend.setFixedHeight(60)
        lab_actual_volume_var = QLabel("Actual Volume")
        self.cb_actual_volume_var = RComboBox(self.tabAFEFX)
        self.cb_actual_volume_var.setFixedWidth(515)
        self.cb_actual_volume_var.setFixedHeight(60)
        lab_elasticity_inc = QLabel("Elasticity: Value Increase by(%)")
        lab_elasticity_inc.setStyleSheet("color:'black';font-size: 12pt; margin:5,0,0,0")
        lab_rprice = QLabel("Real Price")
        lab_real_gdp = QLabel("Real GDP per capita")
        self.sb_rprice = QSpinBox()
        self.sb_rprice.setStyleSheet("width: 65px")
        self.sb_rprice.setValue(10)
        self.sb_real_gdp = QSpinBox()
        self.sb_real_gdp.setStyleSheet("width: 65px")
        self.sb_real_gdp.setValue(10)
        self.btn_open_settings_afefx = QPushButton("Restore Settings")
        self.btn_open_settings_afefx.setToolTip("Restore settings from a file previously saved in")
        self.btn_open_settings_afefx.setStyleSheet("width: 300px; height: 75px")
        self.btn_save_settings_afefx = QPushButton("Save Settings")
        self.btn_save_settings_afefx.setToolTip("Save settings to the deault setting file")
        self.btn_save_settings_afefx.setStyleSheet("width: 300px; height: 75px;margin: 3,0,0,0")
        self.btn_save_as_settings_afefx = QPushButton("Save Settings As")
        self.btn_save_as_settings_afefx.setStyleSheet("width: 300px; height: 75px;")


        grid_layout.addWidget(lab_proj_des,3,0,Qt.AlignRight)
        grid_layout.addWidget(self.proj_des, 3, 1, 1,4,Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(lab_dataset, 4, 0, Qt.AlignRight)
        grid_layout.addWidget(self.dataset_afefx, 4, 1, Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(lab_key_var, 5, 0, Qt.AlignRight)
        grid_layout.addWidget(self.list_key_var_afefx, 5, 1,2,1, Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(lab_fix_var, 5, 2, Qt.AlignRight)
        grid_layout.addWidget(self.list_fix_var, 5, 3, 2,1,Qt.AlignTop)
        grid_layout.addWidget(lab_date, 7, 0,  Qt.AlignBottom | Qt.AlignRight)
        grid_layout.addWidget(self.cb_date_var_afefx, 7, 1, Qt.AlignBottom | Qt.AlignLeft)
        grid_layout.addWidget(lab_period, 8, 0,  Qt.AlignBottom | Qt.AlignRight)
        grid_layout.addWidget(self.cb_period_afefx,8, 1, Qt.AlignBottom | Qt.AlignLeft)
        grid_layout.addWidget(lab_dependent_var, 9, 0,  Qt.AlignBottom | Qt.AlignRight)
        grid_layout.addWidget(self.cb_dependent_var_afefx, 9, 1, Qt.AlignBottom | Qt.AlignLeft)
        grid_layout.addWidget(lab_trend, 10, 0,  Qt.AlignBottom | Qt.AlignRight)
        grid_layout.addWidget(self.cb_trend, 10, 1, Qt.AlignBottom | Qt.AlignLeft)

        grid_layout.addWidget(lab_actual_volume_var, 11, 0, Qt.AlignRight)
        grid_layout.addWidget(self.cb_actual_volume_var, 11, 1, Qt.AlignBottom | Qt.AlignLeft)
  
        grid_layout.addWidget(lab_NPIdataset, 7, 2, Qt.AlignRight)
        grid_layout.addWidget(self.NPI_dataset, 7, 3, Qt.AlignCenter | Qt.AlignLeft)

        grid_layout.addWidget(lab_sarimax_order, 13, 0, Qt.AlignRight)
        grid_layout.addWidget( lab_pval, 14, 0, Qt.AlignRight)
        grid_layout.addWidget( lab_dval, 15, 0, Qt.AlignRight)
        grid_layout.addWidget( lab_qval, 16, 0, Qt.AlignRight)
        grid_layout.addWidget(self.sb_pval, 14, 1, Qt.AlignLeft)
        grid_layout.addWidget(self.sb_dval, 15, 1, Qt.AlignLeft)
        grid_layout.addWidget(self.sb_qval, 16, 1, Qt.AlignLeft)

        grid_layout.addWidget(lab_autoarima,13, 4, Qt.AlignRight)
        grid_layout.addWidget(self.cb_autoarima, 13, 5, Qt.AlignLeft)
        grid_layout.addWidget(lab_intercept, 14, 4, Qt.AlignRight)
        grid_layout.addWidget(self.cb_intercept_afefx, 14, 5, Qt.AlignLeft)
        #seasonal order
        grid_layout.addWidget(lab_seasonality_afefx, 15, 4, Qt.AlignRight)
        grid_layout.addWidget(self.cb_seasonality_afefx, 15, 5, Qt.AlignLeft)
        grid_layout.addWidget(lab_add, 16, 4, Qt.AlignRight)
        grid_layout.addWidget(self.radioButton_add, 16, 5, Qt.AlignLeft)
        grid_layout.addWidget(lab_mul, 17, 4, Qt.AlignRight)
        grid_layout.addWidget(self.radioButton_mul, 17, 5, Qt.AlignLeft)

        #transformation
        grid_layout.addWidget(lab_transformation_var, 8, 2, Qt.AlignRight)
        grid_layout.addWidget(self.cb_transformation_var, 8, 3, 2,1, Qt.AlignTop)
        grid_layout.addWidget(lab_transformation_type, 8, 4, Qt.AlignLeft)
        grid_layout.addWidget(self.cb_transformation_type, 8, 5, 2,1,Qt.AlignTop)

        grid_layout.addWidget(self.btn_add_transformation, 10, 3 ,Qt.AlignLeft)

        grid_layout.addWidget(lab_selected_transformation, 11, 2, Qt.AlignRight)
        grid_layout.addWidget(self.list_selected_transformation, 11, 3, 2,1,Qt.AlignTop)


        #
        grid_layout.addWidget(lab_elasticity_inc, 13, 2, Qt.AlignRight)
        grid_layout.addWidget(lab_rprice, 14, 2, Qt.AlignRight)
        grid_layout.addWidget(self.sb_rprice, 14, 3, Qt.AlignLeft)
        grid_layout.addWidget(lab_real_gdp, 15, 2, Qt.AlignRight)
        grid_layout.addWidget(self.sb_real_gdp, 15, 3, Qt.AlignLeft)
        grid_layout.addWidget(lab_train_test, 16, 2, Qt.AlignRight)
        grid_layout.addWidget(self.train_test_ratio_afefx, 16, 3, Qt.AlignLeft)
        # grid_layout.addWidget(empy_line,16,0)
        grid_layout.addWidget(self.btn_open_settings_afefx,17,1,Qt.AlignRight)
        grid_layout.addWidget(self.btn_save_settings_afefx,17,2,Qt.AlignLeft)
        grid_layout.addWidget(self.btn_save_as_settings_afefx,17,3,Qt.AlignLeft)

    def set_values_to_UI_afefx(self):
        try:
            dataset = self.afefx_settings['tsf-batch-settings']['dataset']
            df = pd.read_excel(dataset)
            var_list = list(df.columns)
            self.proj_des.setText(self.afefx_settings['tsf-batch-settings']['description'])
            self.dataset_afefx.setText(dataset)
            vars_to_check = self.afefx_settings['tsf-batch-settings']['variables']
            self.load_listWidget_checked(self.list_key_var_afefx,var_list,vars_to_check)
            self.cb_date_var_afefx.addItems(var_list)
            self.cb_period_afefx.addItems(self.afefx_settings['tsf-batch-settings']['frequency-options'].keys())
            self.cb_period_afefx.setCurrentText(self.afefx_settings['tsf-batch-settings']['frequency'])
            self.cb_date_var_afefx.setCurrentText(self.afefx_settings['tsf-batch-settings']['indexed on'])
            self.cb_dependent_var_afefx.addItems(var_list)
            self.cb_dependent_var_afefx.setCurrentText(self.afefx_settings['tsf-batch-settings']['dependent-var']) 
            var_list_trend = var_list.copy()
            var_list_trend.insert(0,"No Trend")
            self.cb_trend.addItems(var_list_trend)
            self.cb_trend.setCurrentText(self.afefx_settings['tsf-batch-settings']['elasticity-trend'])           
            self.cb_actual_volume_var.addItems(var_list)
            self.cb_actual_volume_var.setCurrentText(self.afefx_settings['tsf-batch-settings']['Actual-vol-column'])
            vars_to_be_fixed = self.afefx_settings['tsf-batch-settings']['fixed-xvars']
            self.load_listWidget_checked(self.list_fix_var,vars_to_be_fixed,vars_to_be_fixed)
            p = self.afefx_settings['tsf-batch-settings']['ts-order']['p_max']
            d = self.afefx_settings['tsf-batch-settings']['ts-order']['d_max']
            q = self.afefx_settings['tsf-batch-settings']['ts-order']['q_max']
            self.sb_pval.setValue(p)
            self.sb_dval.setValue(d)
            self.sb_qval.setValue(q)
            ttr = int(self.afefx_settings['tsf-batch-settings']["train-test-ratio"]*100)
            self.train_test_ratio_afefx.setValue(ttr)
            self.NPI_dataset.setText(self.afefx_settings['tsf-batch-settings']["NPI-growth-dataset"])
           #transformation
            var_list_transform = var_list.copy()
            var_list_transform.insert(0,"No Transformation")
            self.cb_transformation_var.addItems(var_list_transform)           
            self.cb_transformation_var.setCurrentText(self.afefx_settings['tsf-batch-settings']["transform-variables"])

            self.cb_transformation_type.addItems(self.afefx_settings['tsf-batch-settings']['transformation-options'])
            self.cb_transformation_type.setCurrentText(self.afefx_settings['tsf-batch-settings']['transformation_type'])
            
            b_auto_arima = True if self.afefx_settings['tsf-batch-settings']["auto-arima"]=='yes' else False
            b_intercept = True if self.afefx_settings['tsf-batch-settings']["intercept"]=='Yes' else False
            self.cb_autoarima.setChecked(b_auto_arima)
            self.cb_intercept_afefx.setChecked(b_intercept)
            price_increase_by = self.afefx_settings['tsf-batch-settings']["elasticity-feature-increase-by-percent"]["RPRICE"]
            gdp_increase_by = self.afefx_settings['tsf-batch-settings']["elasticity-feature-increase-by-percent"]["GDP_PER_CAPITA"]
            self.sb_rprice.setValue(price_increase_by)
            self.sb_real_gdp.setValue(gdp_increase_by)
            self.tabAFEFX.repaint()
            self.logger.info(f"efefx_ui updated")
            # self.logger.info(f"variable list: {var_list}")
            # self.logger.info(f"variable_trend list: {var_list_trend}")
            # self.logger.info(f"variable transformation: {var_list_transform}")
            
        except Exception as err:
            self.logger.error(f"set_values_to_UI_afefx: {err}")
    def load_listWidget_checked(self,listWidget,variables,variables_to_be_checked):
        try:
            listWidget.clear()
            for var in variables:
                item = QListWidgetItem(var, listWidget)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                if var in variables_to_be_checked:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
            listWidget.repaint()
        except Exception as err:
            self.logger.error(f"load_list_widget_checked: {err}")
    def restore_settings_afefx(self):
        file = self.parent().parent().get_file( 
               **{
                    "title": "Select the forecast-elasticity framework setting file",
                    "f_type": "AFEFX (*.afefx)",
                })
        # load settings from afefx
        if file:
            try:
                self.afefx_settings = None
                self.parent().parent().load_afefx_settings(file)
                self.afefx_settings = self.parent().parent().afefx_settings
                self.set_values_to_UI_afefx()
                self.parent().parent().on_tsfefx(file)
                self.parent().parent().update_status(f"{file} has been restored into settings to run")
                self.setEnabled(True)
            except Exception as err:
                self.logger.error(f"restore_settings_afefx: {err}")

    def update_settings(self):
        selected_vars = self.parent().parent().get_vars_from_listbox(self.list_key_var_afefx)
        self.afefx_settings["tsf-batch-settings"]["description"] = self.proj_des.text()
        self.afefx_settings["tsf-batch-settings"]["dataset"] = self.dataset_afefx.text()
        self.afefx_settings["tsf-batch-settings"]["variables"] = selected_vars['checked']
        self.afefx_settings["tsf-batch-settings"]["dependent-var"] = self.cb_dependent_var_afefx.currentText()
        fixed_xvars = self.parent().parent().get_vars_from_listbox(self.list_fix_var)
        self.afefx_settings["tsf-batch-settings"]["fixed-xvars"] = fixed_xvars['checked']
        self.afefx_settings["tsf-batch-settings"]["elasticity-feature-increase-by-percent"]['RPRICE'] = self.sb_rprice.value()
        self.afefx_settings["tsf-batch-settings"]["elasticity-feature-increase-by-percent"]['GDP_PER_CAPITA'] = self.sb_real_gdp.value()
        self.afefx_settings["tsf-batch-settings"]["NPI-growth-dataset"] = self.NPI_dataset.text()
        self.afefx_settings["tsf-batch-settings"]["Actual-vol-column"] = self.cb_actual_volume_var.currentText()
        self.afefx_settings["tsf-batch-settings"]["elasticity-trend"] = self.cb_trend.currentText()
        self.afefx_settings["tsf-batch-settings"]["indexed on"] = self.cb_date_var_afefx.currentText()
        self.afefx_settings["tsf-batch-settings"]["frequency"] = self.cb_period_afefx.currentText()
        self.afefx_settings["tsf-batch-settings"]["train-test-ratio"] = self.train_test_ratio_afefx.value()/100
        self.afefx_settings["tsf-batch-settings"]["auto-arima"] = 'yes' if self.cb_autoarima.isChecked() else 'no'
        self.afefx_settings["tsf-batch-settings"]["intercept"] = 'Yes' if self.cb_intercept_afefx.isChecked() else 'No'
        self.afefx_settings['tsf-batch-settings']['ts-order']['p_max'] = self.sb_pval.value()
        self.afefx_settings['tsf-batch-settings']['ts-order']['d_max'] = self.sb_dval.value()
        self.afefx_settings['tsf-batch-settings']['ts-order']['q_max'] = self.sb_qval.value()
        #seasonality
        self.afefx_settings["tsf-batch-settings"]["seasonality"] = 'Yes' if self.cb_seasonality_afefx.isChecked() else 'No'
        self.afefx_settings["tsf-batch-settings"]["model-additive"] = True if self.radioButton_add.isChecked() else False 
        #transformation
       
        self.afefx_settings["tsf-batch-settings"]["transform-variables"] = self.cb_transformation_var.currentText()
        self.afefx_settings["tsf-batch-settings"]["transformation_type"] = self.cb_transformation_type.currentText()
        
        dict_transformation=self.get_transformed_vars()
        self.afefx_settings["tsf-batch-settings"]["transformation"] = dict_transformation

    def get_transformed_vars(self):
        try:
            transformation_list = self.parent().parent().get_vars_from_listbox(self.list_selected_transformation)["all"]
            #transformation_list = [tuple(item) for item in transformation_list]
            d_transformations={}
            for item in transformation_list:
                x=item.split("-")
                x=[ele.strip() for ele in x]
                if x[1] in d_transformations:        
                    d_transformations[x[1]].append(x[0])        
                else:
                    d_transformations[x[1]]=[x[0]] 
            self.logger.info(f"transformation-list: {transformation_list}")            
            return d_transformations
        except Exception as err:
            self.logger.error(f"get_transformed_vars: {err}")


    def write_to_json(self,file_name,d_settings):
        try:
            with open(file_name, 'w') as f:
                if self.flg_json_format==1:
                    f.write(json.dumps(d_settings, indent=2))
                else:
                    f.write(json.dumps(d_settings))
            return True, ""
        except Exception as err:
            return (False,err)

    def dashboard(self):
        try:
            self.logger.info("in the dashboard")
            layout = QVBoxLayout(self)
            self.tabDashboard.setLayout(layout)
            self.tabDashboard.setStyleSheet("color: white; background-color: black")
            self.tabCF = (
                QWidget()
            )  # Correlation Func - ACF and PACF of actual values and differences
            self.tabDecomposition = QWidget()  # seasonal trend and seasonal decomposition
            self.tabCM = QWidget()  # CM - graph of correlation matrix
            self.tabForecasts = QWidget()  # actual vs predicted values
            self.tabResiduals = QWidget()  # plots on residuals
            self.tabADF = QWidget()
            self.tabSeasonality = QWidget()
            # self.tabARIMA = QWidget()
            self.tabSARIMAX = QWidget()
            #self.tabdft = QWidget()
            # self.tabARIMAForecast = QWidget()
            # self.tabARIMAResiduals = QWidget()
            self.tabAutoARIMA = QWidget()
            self.tabAutoARIMAForecast = QWidget()
            self.tabForecast_N = QWidget()
            self.tabWidgetDashboard = DashboardTabWidget()  ##QTabWidget()
            self.tabWidgetDashboard.setStyleSheet(
                "Selection-color: white;selection-background-color: green;"
            )
            self.tabWidgetDashboard.addTab(self.tabCM, "Correlation Matrix")
            self.tabWidgetDashboard.addTab(self.tabCF, "Autocorrelation Functions")
            self.tabWidgetDashboard.addTab(self.tabDecomposition, "Decomposition")
            self.tabWidgetDashboard.addTab(self.tabADF, "Augmented Dickey Fuller Test")
            self.tabWidgetDashboard.addTab(self.tabSeasonality, "Seasonality")

            self.tabWidgetDashboard.addTab(self.tabAutoARIMA, "Auto ARIMA")
            self.tabWidgetDashboard.addTab(
                self.tabAutoARIMAForecast, "Auto ARIMA Forecasts"
            )
            self.tabWidgetDashboard.addTab(
                self.tabForecast_N, "Auto ARIMA Forecasts for N Periods"
            )
            self.tabWidgetDashboard.addTab(self.tabSARIMAX, "SARIMAX & Durbin-Watson Test")
            self.tabWidgetDashboard.setTabPosition(QTabWidget.West)
            layout.addWidget(self.tabWidgetDashboard)
            layout.addStretch(1)
            self.setStyleSheet("padding: 0px; margin: 0px; background-color: #4A4444;")
            self.setLayout(layout)
            self.logger.info("dashboard rendered successfully")
        except Exception as err:
            self.logger.error(f"problem in rendering the dashboard: {err}")

        # self.tabWidgetDashboard.addTab(self.tabdft, "Durbin Watson Test")
        # self.tabWidgetDashboard.setTabPosition(QTabWidget.West)
        # layout.addWidget(self.tabWidgetDashboard)
        # layout.addStretch(1)
        # self.setStyleSheet("padding: 0px; margin: 0px; background-color: #4A4444;")
        # self.setLayout(layout)
    def get_nlags(self):
        nlags_acf,nlags_pacf = self.dependent_var.shape[0]*.6, self.dependent_var.shape[0]*.4
        return (nlags_acf,nlags_pacf)

    def acf_pacf(self):
        # self.logger.info("in DatasetView")
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabCF.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - Autocorrelation Functions")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        sc = Canvas()
        sc.axes.cla()
        sc.setToolTip("Lags beyond the shaded areas are statistically significant")
        nlags_acf,nlags_pacf = self.get_nlags()
        for i in range(3):
            sc.axes = sc.fig.add_subplot(6, 3, i + 1)
            if i == 0:
                self.dependent_var.plot(ax=sc.axes)  
                # plt.plot(self.dependent_var.values,ax=sc.axes) 
            elif i == 1:
                sgt.plot_acf(self.dependent_var, lags=nlags_acf, zero=False, ax=sc.axes)
            else:
                sgt.plot_pacf(self.dependent_var, lags=nlags_pacf, zero=False, ax=sc.axes)
        grid_layout.addWidget(sc, 1, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)
        grid_layout.addWidget(empy_line, 2, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)

        # 1st order difference - ACF and PACF
        for i in range(2):
            sc.axes = sc.fig.add_subplot(6, 3, i + 7)
            if i == 0:
                sgt.plot_acf(
                    self.dependent_var.diff().dropna(),
                    lags=nlags_acf,
                    zero=False,
                    ax=sc.axes,
                    title="ACF - First Order Diff",
                )
                # plt.title("Autocorrelation - 1st Order Difference")
            else:
                # plt.title("Partial Autocorrelation - 1st Order Difference")
                sgt.plot_pacf(
                    self.dependent_var.diff().dropna(),
                    lags=nlags_pacf,
                    zero=False,
                    ax=sc.axes,
                    title="PACF - First Order Diff",
                )
        # 2nd order difference - ACF and PACF
        grid_layout.addWidget(empy_line, 3, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)
        for i in range(2):
            sc.axes = sc.fig.add_subplot(6, 3, i + 13)
            if i == 0:
                sgt.plot_acf(
                    self.dependent_var.diff().diff().dropna(),
                    lags=nlags_acf,
                    zero=False,
                    ax=sc.axes,
                    title="ACF - Second Order Diff",
                )
                # plt.title("Autocorrelation - 1st Order Difference")
            else:
                # plt.title("Partial Autocorrelation - 1st Order Difference")
                sgt.plot_pacf(
                    self.dependent_var.diff().diff().dropna(),
                    lags=nlags_pacf,
                    zero=False,
                    ax=sc.axes,
                    title="PACF - Second Order Diff",
                )
        grid_layout.addWidget(sc, 7, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)

    def decompose(self):
        # self.logger.info("in DatasetView")
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabDecomposition.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(
            f"Anuman Dashboard - Decomposition into Trend, Seasonality and Residuals"
        )
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        sc = Canvas()
        sc.axes = sc.fig.add_subplot(6, 3, 1)
        decompose_data = seasonal_decompose(self.dependent_var, model="additive")
        seasonality = decompose_data.seasonal
        seasonality.plot(ax=sc.axes, title="Seasonality")
        grid_layout.addWidget(sc, 3, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)
        sc.axes = sc.fig.add_subplot(6, 3, 2)
        trend = decompose_data.trend
        trend.plot(ax=sc.axes, title="Trend")
        sc.axes = sc.fig.add_subplot(6, 3, 3)
        resid = decompose_data.resid
        resid.plot(ax=sc.axes, title="Residuals")
        grid_layout.addWidget(sc, 7, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)

    def corr_matrix(self):
        self.logger.info("Dashboard cm")
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabCM.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - the Correlation Matrix")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        # plt.style.use('default')
        plt.style.use("dark_background")
        # if self.cmCanvas:
        #     self.cmCanvas.axes.cla()
        self.cmCanvas = Canvas(width=20, height=10)
        self.cmCanvas.axes = self.cmCanvas.fig.add_subplot(1, 1, 1)
        # self.cmCanvas.axes.cla()
        self.cmCanvas.axes.grid(False)
        # self.cmCanvas.axes.axis('off')
        # sc.axes.spines["right"].set_visible(False)
        # sc.axes.spines["top"].set_visible(False)
        # sc.axes.spines["bottom"].set_visible(False)

        # corr = self.data.corr()
        # # self.logger.info(corr)
        # sns.heatmap(corr, cmap="Reds", annot=True,ax=self.cmCanvas.axes)
        grid_layout.addWidget(self.cmCanvas, 7, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)
        self.cmChart()

    def cmChart(self):
        try:
            self.logger.info("updating cm chart")
            self.cmCanvas.axes.cla()
            corr = self.data.corr()
            sns.heatmap(corr, cmap="Reds", annot=True, ax=self.cmCanvas.axes)
            self.cmCanvas.draw()
        except Exception as err:
            self.logger.error(f"problem in rendering cm chart")

    def adft(self):
        try:
            adf_test = self.parent().parent().tarimax.adf_test()
            # df = pd.DataFrame(adf_test,index=[0])
            # df.to_csv('adfpyqt5.csv')
            self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
            grid_layout = QGridLayout()
            self.tabADF.setLayout(grid_layout)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

            title = QLabel(
                f"Anuman Dashboard - the ADF (Augmented Dickey-Fuller) test for Stationarity"
            )
            title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
            grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)
            empy_line = QLabel("")
            empy_line.setStyleSheet("width: 150px;")
            grid_layout.addWidget(empy_line, 1, 0, Qt.AlignLeft | Qt.AlignTop)
            i = 0
            for k, v in adf_test.items():
                lab_adf = QLabel(f"{k}: ")
                val_adf = QLabel(f"{v:.4f}")
                lab_adf.setStyleSheet(
                    "border: 1px;border-color:white;width:35px;margin:5px"
                )
                val_adf.setStyleSheet(
                    "background-color:green;border: 1px;border-color:white;width: 20px"
                )
                val_adf.setEnabled(False)
                if i == 1:  # p-value
                    tip = "Stationary" if v < 0.05 else "Non-Stationary"
                    val_adf.setToolTip(f"p-value suggests the data is {tip}")
                grid_layout.addWidget(lab_adf, 2 + i, 0, Qt.AlignRight | Qt.AlignBottom)
                grid_layout.addWidget(val_adf, 2 + i, 1, Qt.AlignBottom | Qt.AlignLeft)
                i += 1
        except Exception as err:
            self.logger.error(f"adf: {err}")

    def seasonality(self):
        try:
            self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
            grid_layout = QGridLayout()
            self.tabSeasonality.setLayout(grid_layout)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

            title = QLabel(
                f"Anuman Dashboard - Seasonality Spikes in the time-series data"
            )
            title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
            grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)
            empy_line = QLabel("")
            empy_line.setStyleSheet("width: 150px;")
            grid_layout.addWidget(empy_line, 1, 0, Qt.AlignLeft | Qt.AlignTop)
            sc = Canvas()
            sc.axes = sc.fig.add_subplot(2, 1, 1)
            data = self.parent().parent().tarimax.get_dependent_var()
            # Usual Differencing
            sc.axes.plot(data[:], label="Original Series")
            sc.axes.plot(data[:].diff(1), label="Usual Differencing")
            sc.axes.set_title("Usual Differencing")
            # sc.axes.legend(loc='upper left', fontsize=10)
            # grid_layout.addWidget(sc,1,0,Qt.AlignTop | Qt.AlignLeft)
            # seasonal diff
            sc.axes = sc.fig.add_subplot(2, 1, 2)
            sc.axes.plot(data[:], label="Original Series")
            sc.axes.plot(data[:].diff(12), label="Seasonal Differencing", color="green")
            sc.axes.set_title("Seasonal Differencing")
            # sc.axes.legend(loc='upper left', fontsize=10)
            grid_layout.addWidget(sc, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        except Exception as err:
            self.logger.error(f"seasonality: {err}")

    def enable_model_dashboard(self):
        try:
            if self.cb_auto_arima.isChecked():
                self.tabAutoARIMA.setEnabled(True)
                # self.tabAutoARIMAForecast.setEnabled(True)
                # self.self.tabForecast_N.setEnabled(True)
            else:
                self.tabSARIMAX.setEnabled(True)
        except Exception as err:
            self.logger.error(f"enable model dashboard: {err}")

    # def arima(self):
    #     self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
    #     grid_layout = QGridLayout()
    #     self.tabARIMA.setLayout(grid_layout)
    #     self.clearLayout(grid_layout)
    #     grid_layout.setContentsMargins(0,0,0,0)
    #     grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    #
    #     title = QLabel(f"Anuman Dashboard - the ARIMA Model Summary")
    #     title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
    #     grid_layout.addWidget(title,0,0,Qt.AlignLeft | Qt.AlignTop)
    #     empy_line = QLabel("")
    #     empy_line.setStyleSheet("width: 150px;")
    #     grid_layout.addWidget(empy_line,1,0,Qt.AlignLeft | Qt.AlignTop)
    #     try:
    #         # self.arima_model = None
    #         # model_summary = None
    #         self.arima_model = self.parent().parent().tarimax.ARIMA(
    #                     **{"p":self.arima_p.value(),
    #                         "d": self.arima_d.value(),
    #                         "q": self.arima_q.value()
    #                         }
    #                     )
    #         model_summary = self.arima_model.summary()
    #         # self.arima_summary = QLabel("")
    #         # self.arima_summary.clear()
    #         self.arima_summary.setText(str(model_summary))
    #         # self.arima_summary.setStyleSheet("width:500px;height:350")
    #         grid_layout.addWidget(self.arima_summary,2,0,3,3,Qt.AlignLeft | Qt.AlignTop)
    #         self.arima_summary.repaint()
    #         # self.parent().parent().message_box(f"in arima with p:{self.arima_p.value()},d:{self.arima_d.value()},q:{self.arima_res}")
    #         # self.parent().parent().update_status(f"AIC - {self.arima_model.aic}")
    #     except Exception as err:
    #         self.logger.warning(f"Problem in arima - {err}")
    def sarimax(self):
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabSARIMAX.setLayout(grid_layout)
        self.clearLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - the SARIMAX Model Summary")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        grid_layout.addWidget(empy_line, 1, 0, Qt.AlignLeft | Qt.AlignTop)
        try:
            # self.arima_model = None
            # model_summary = None
            _d = {
                "p": self.arima_p.value(),
                "d": self.arima_d.value(),
                "q": self.arima_q.value(),
            }
            if self.cb_seasonality.isChecked():
                _d["P"] = self.season_sarimax_P.value()
                _d["D"] = self.season_sarimax_D.value()
                _d["Q"] = self.season_sarimax_Q.value()
                _d["m"] = self.season_sarimax_m.value()
                _d["seasonal"] = True

            self.sarimax_model = self.parent().parent().tarimax.SARIMAX(**_d)

            model_summary = self.sarimax_model.summary()
            dw_test = self.parent().parent().tarimax.durbin_watson_test(self.sarimax_model.resid)
            r_square_value = self.parent().parent().tarimax.get_RSquared(self.sarimax_model.resid)

            tt_dwt = self.get_tooltip_dwt(dw_test)
            self.sarimax_summary.setToolTip(tt_dwt)

            # self.arima_summary = QLabel("")
            # self.arima_summary.clear()
            self.sarimax_summary.setText(f"{model_summary}\n\nDurbin Watson Test: {dw_test}\n\nR Squared: {r_square_value}")

            # self.arima_summary.setStyleSheet("width:500px;height:350")
            grid_layout.addWidget(
                self.sarimax_summary, 2, 0, 3, 3, Qt.AlignLeft | Qt.AlignTop
            )
            self.sarimax_summary.repaint()
            # self.parent().parent().message_box(f"in arima with p:{self.arima_p.value()},d:{self.arima_d.value()},q:{self.arima_res}")
        except Exception as err:
            self.logger.warning(f"Problem in sarimax - {err}")

    def get_tooltip_dwt(self, dwt):
        if dwt < 2:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are positively correlated"

        elif dwt == 2:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are not correlated"
        else:
            return f"Durbin Watson Test Statistic of {dwt} indicates the residuals are negatively correlated"


    # def arima_forecast(self):
    #     try:
    #         self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
    #         grid_layout = QGridLayout()
    #         self.tabARIMAForecast.setLayout(grid_layout)
    #         grid_layout.setContentsMargins(0,0,0,0)
    #         grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    #
    #         title = QLabel(f"Anuman Dashboard - ARIMA Model Forecast")
    #         title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
    #         grid_layout.addWidget(title,0,0,Qt.AlignLeft | Qt.AlignTop)
    #
    #         empy_line = QLabel("")
    #         empy_line.setStyleSheet("width: 150px;")
    #         sc = Canvas()
    #         sc.axes = sc.fig.add_subplot(1,1,1)
    #         # sc.axes.legend(loc='upper left', fontsize=10)
    #         grid_layout.addWidget(sc,1,0,Qt.AlignTop | Qt.AlignLeft)
    #         self.arima_model.plot_predict(ax=sc.axes,dynamic=False)
    #         grid_layout.addWidget(sc,7,0,4,5,Qt.AlignTop | Qt.AlignLeft)
    #     except Exception as err:
    #         self.logger.error(f"arima_forecast: {err}")

    def auto_arima_residuals(self):
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabARIMAResiduals.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - Auto ARIMA Model Residuals")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        sc = Canvas()
        sc.axes = sc.fig.add_subplot(2, 2, 1)
        # model = self.parent().parent().tarimax.auto_arima_model
        self.parent().parent().tarimax.auto_arima_model.plot_diagnostics(fig=sc.fig)
        # sc.axes.legend(loc='upper left', fontsize=10)
        grid_layout.addWidget(sc, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        self.tabARIMAForecast.draw()

    def auto_arima(self):
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabAutoARIMA.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - the Auto ARIMA Model Summary")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        grid_layout.addWidget(empy_line, 1, 0, Qt.AlignLeft | Qt.AlignTop)
        _d = {
            "start_p": 0,
            "start_q": 0,
        }
        self.auto_arima_model = self.parent().parent().tarimax.auto_arima(**_d)
        # model_summary = self.auto_arima_summary
        self.auto_arima_summary.setText(str(self.auto_arima_model.summary()))
        grid_layout.addWidget(self.auto_arima_summary, 2, 0, Qt.AlignLeft | Qt.AlignTop)

        # self.logger.info("updating auto arima ...")

    def auto_arima_forecast(self):
        # self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabAutoARIMAForecast.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(f"Anuman Dashboard - Auto ARIMA Model Forecast")
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        sc = Canvas()
        sc.axes = sc.fig.add_subplot(1, 1, 1)
        grid_layout.addWidget(sc, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        data = self.parent().parent().tarimax.get_dependent_var()
        n_periods = data.shape[0]
        fc, confint = (
            self.parent()
            .parent()
            .tarimax.auto_arima_model.predict(n_periods=n_periods, return_conf_int=True)
        )
        index_of_fc = data.index

        # make series for plotting purpose
        fc_series = pd.Series(fc, index=data.index)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        # Plot

        sc.axes.plot(data)
        sc.axes.plot(fc_series, color="darkgreen")
        sc.axes.fill_between(
            lower_series.index, lower_series, upper_series, color="g", alpha=0.25
        )

        # sc.axes.title("Final Forecast of Volume Sales")
        # sc.axes.legend(loc='upper left', fontsize=10)
        grid_layout.addWidget(sc, 7, 0, 4, 5, Qt.AlignTop | Qt.AlignLeft)

    def forecast_using_auto_arima_n_periods(self):

        n_periods = self.forecast_N.value()
        fitted, confint = (
            self.parent()
            .parent()
            .tarimax.auto_arima_model.predict(n_periods=n_periods, return_conf_int=True)
        )
        data = self.parent().parent().tarimax.get_dependent_var()
        index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq="MS")

        # make series for plotting purpose
        fitted_series = pd.Series(fitted, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabForecast_N.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        title = QLabel(
            f"Anuman Dashboard - Auto ARIMA Model Forecast for {n_periods} periods"
        )
        title.setStyleSheet("font-size: 10pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")

        # Plot
        sc = Canvas()
        sc.axes = sc.fig.add_subplot(1, 1, 1)
        grid_layout.addWidget(sc, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        sc.axes.plot(data)
        sc.axes.plot(fitted_series, color="darkgreen")
        # sc.axes.legend(loc='upper left', fontsize=10)
        sc.axes.fill_between(
            lower_series.index, lower_series, upper_series, color="g", alpha=0.35
        )

    def datasetView(self):
        # self.logger.info("in DatasetView")
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabDSView.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"Anuman - Dataset - {self.dataset.text()}")
        title.setStyleSheet("font-size: 14pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, 1, 7, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")
        # grid_layout.addStretch(1)

    def Setup(self):
        self.setStyleSheet(""" QWidget {background-color:#1a1a1a;color:#ffffff;}""")
        grid_layout = QGridLayout()
        self.tabSetup.setLayout(grid_layout)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Anuman - Settings")
        title.setStyleSheet("font-size: 14pt;color: white;text-align: center;")
        grid_layout.addWidget(title, 0, 0, 1, 7, Qt.AlignLeft | Qt.AlignTop)

        empy_line = QLabel("")
        empy_line.setStyleSheet("width: 150px;")

    def invoke_explorer(self):
        path = self.img_path_folder.text()
        path = os.path.realpath(path)
        os.startfile(path)

    def get_xlsx(self):
        cwd = os.getcwd()
        openFileName = QFileDialog.getOpenFileName(
            self,
            "Open File",
            cwd,
            "Excel Files (*.xlsx)",
            "",
            QFileDialog.DontUseNativeDialog,
        )
        if openFileName != ("", ""):
            file = openFileName[0]
            try:
                wb = openpyxl.load_workbook(file)
                ws = wb.worksheets
                sheets = []
                for sheet in ws:
                    sheets.append(sheet.title)
                return (file, sheets)
            except Exception as err:
                self.message_box(str(err))
                self.logger.error(err)
        else:
            return (None, None)

    def set_json_files_folder(self):
        path = QFileDialog.getExistingDirectory(
            parent=self,
            caption="Select directory for JSON Files",
            options=QFileDialog.DontUseNativeDialog,
        )
        if path != None and path.strip() != "":
            self.json_path_folder.setText(path)


class ListWidget(QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.setAcceptDrops(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self._del_item()

    def _del_item(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def sizeHint(self):
        s = QSize()
        s.setHeight(super(ListWidget, self).sizeHint().height())
        s.setHeight(200)
        s.setWidth(500)
        return s

    def mimeTypes(self):
        mimetypes = super().mimeTypes()
        mimetypes.append("text/plain")
        return mimetypes

    def dropMimeData(self, index, data, action):
        if data.hasText():
            self.addItem(data.text())
            return True
        else:
            return super().dropMimeData(index, data, action)
        ##515254


class HorizontalTabBar(QTabBar):
    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.CE_TabBarTabShape, option)
            painter.drawText(
                self.tabRect(index), Qt.AlignLeft | Qt.TextDontClip, self.tabText(index)
            )

    def tabSizeHint(self, index):
        size = QTabBar.tabSizeHint(self, index)
        if size.width() < size.height():
            size.transpose()
        return size


class DashboardTabWidget(QTabWidget):
    def __init__(self, parent=None):
        QTabWidget.__init__(self, parent)
        self.setTabBar(HorizontalTabBar())


style = """
            QWidget {background-color:white;}
            QLabel { margin: 5px; font-size: 12pt;}
            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 0;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0;
            }

            QScrollBar::handle:horizontal {
                background: #374146;
                min-width: 16px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #374146;
                min-height: 16px;
                border-radius: 5px;
            }

            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                  border: none;
                  background: none;
            }

            QListWidget QScrollBar::handle:horizontal {
                    background-color:rgb(0, 170, 171);
            }
            QListWidget QScrollBar::handle:vertical {
                    background-color:rgb(0, 170, 171);
                    margin:2;
            QListWidget { font-size: 12pt; width: 430 px; height: 300 px;}
            }
            QToolTip {font-size: 10pt;background-color: white; color: black}
        """

if __name__ == "__main__":
    # don't auto scale when drag app to a different monitor.
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    if QApplication.instance():
        app = QApplication.instance()
    else:
        app = QApplication(sys.argv)
    # screen = app.primaryScreen()
    # geometry = screen.availableGeometry()

    # app.setStyleSheet(style)
    app.setStyle("Fusion")
    anuman = Anuman()
    # anuman.setFixedSize(geometry.width(), geometry.height())
    anuman.show()
    sys.exit(app.exec_())
