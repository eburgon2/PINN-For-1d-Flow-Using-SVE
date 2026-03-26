import os
import sys
#print(sys.executable)
import pandas as pd
import datetime as dt
from dataretrieval import nwis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dataretrieval import waterdata

def get_discharge(up_ID, down_ID,start,end):
    Q_u, metadata = nwis.get_dv(
            sites= up_ID, 
            start=start, 
            end=end, 
            parameterCd='00060'
        )
    Q_u.index = Q_u.index.date
    Q_u.index = pd.to_datetime(Q_u.index)
    Q_u.index.name = "Date"
    Q_u = Q_u.drop(columns=["site_no","00060_Mean_cd"],axis=1)
    Q_u = Q_u.rename(columns={'00060_Mean':'Upstream Mean Discharge (cfs)'})
    Q_u = Q_u.sort_index()

    Q_d, metadata = nwis.get_dv(
                sites= down_ID, 
                start=start, 
                end=end, 
                parameterCd='00060'
            )
    Q_d.index = Q_d.index.date
    Q_d.index = pd.to_datetime(Q_d.index)
    Q_d.index.name = "Date"
    Q_d = Q_d.drop(columns=["site_no","00060_Mean_cd"],axis=1)
    Q_d = Q_d.rename(columns={'00060_Mean':'Downstream Mean Discharge (cfs)'})
    Q_d = Q_d.sort_index()

    discharge = pd.merge(Q_u,Q_d,how='inner',on='Date')
    discharge = discharge.sort_index()

    discharge.to_csv('../Data Files/Raw/By Parameter/Discharge.csv')
    return discharge

def get_width():
    B_u = pd.read_csv("../Data Files/Raw/up_width_raw.csv")
    B_u = B_u.drop(columns=["Activity_DepthHeightMeasure","Activity_DepthHeightMeasureUnit","Result_MeasureUnit"],axis=1)
    B_u = B_u.rename(columns={'Activity_StartDate':'Date','Result_Measure':'Upstream Width (ft)'})
    B_u["Date"] = pd.to_datetime(B_u["Date"])
    B_u.set_index("Date",inplace=True)
    B_u = B_u.sort_index()

    B_d = pd.read_csv("../Data Files/Raw/down_width_raw.csv")
    B_d = B_d.drop(columns=["Activity_DepthHeightMeasure","Activity_DepthHeightMeasureUnit","Result_MeasureUnit"],axis=1)
    B_d = B_d.rename(columns={'Activity_StartDate':'Date','Result_Measure':'Downstream Width (ft)'})
    B_d["Date"] = pd.to_datetime(B_d["Date"])
    B_d.set_index("Date",inplace=True)
    B_d = B_d.sort_index()

    width = pd.merge(B_u,B_d,how='outer',on='Date')
    width = width.sort_index()
    width.to_csv('../Data Files/Raw/By Parameter/Width.csv')
    return B_u,B_d

def get_depth():
    h_u = pd.read_csv("../Data Files/Raw/up_height_raw.csv")
    h_u = h_u.drop(columns=["Activity_DepthHeightMeasure","Activity_DepthHeightMeasureUnit","Result_MeasureUnit"],axis=1)
    h_u = h_u.rename(columns={'Activity_StartDate':'Date','Result_Measure': 'Upstream Datum (ft)'})
    h_u["Date"] = pd.to_datetime(h_u["Date"])
    h_u.set_index("Date",inplace=True)
    h_u = h_u.sort_index()

    h_d = pd.read_csv("../Data Files/Raw/down_height_raw.csv")
    h_d = h_d.drop(columns=["Activity_DepthHeightMeasure","Activity_DepthHeightMeasureUnit","Result_MeasureUnit"],axis=1)
    h_d = h_d.rename(columns={'Activity_StartDate':'Date','Result_Measure': 'Downstream Datum (ft)'})
    h_d["Date"] = pd.to_datetime(h_d["Date"])
    h_d.set_index("Date",inplace=True)
    h_d = h_d.sort_index()

    depth = pd.merge(h_u,h_d,how='outer',on='Date')
    depth = depth.sort_index()
    depth = depth.resample('D').mean()
    depth.to_csv('../Data Files/Raw/By Parameter/depth_to_datum.csv')
    return h_u, h_d

def log_transform(discharge, parameter):
    if(parameter == "depth"):
        up_depth, down_depth = get_depth()

        up_datum = pd.merge(discharge,up_depth,how='outer',on='Date')
        up_datum = up_datum.drop(columns=["Downstream Mean Discharge (cfs)"],axis=1)
        up_datum = up_datum.resample('D').mean()
        up = up_datum.dropna(subset=["Upstream Mean Discharge (cfs)"]).copy()
        
        down_datum = pd.merge(discharge,down_depth,how='outer',on='Date')
        down_datum = down_datum.drop(columns=["Upstream Mean Discharge (cfs)"],axis=1)
        down_datum = down_datum.resample('D').mean()
        down = down_datum.dropna(subset=["Downstream Mean Discharge (cfs)"]).copy()

        variable = 'Datum'

    elif(parameter == "width"):
        up_width, down_depth = get_width()

        up_datum = pd.merge(discharge,up_width,how='outer',on='Date')
        up_datum = up_datum.drop(columns=["Downstream Mean Discharge (cfs)"],axis=1)
        up_datum = up_datum.resample('D').mean()
        up = up_datum.dropna(subset=["Upstream Mean Discharge (cfs)"]).copy()
        
        down_datum = pd.merge(discharge,down_depth,how='outer',on='Date')
        down_datum = down_datum.drop(columns=["Upstream Mean Discharge (cfs)"],axis=1)
        down_datum = down_datum.resample('D').mean()
        down = down_datum.dropna(subset=["Downstream Mean Discharge (cfs)"]).copy()
        
        variable = 'Width'

    

    eps = 1e-6

    def set_data(location,variable, datum):
        training = datum[f"{location} {variable} (ft)"].notna()

        flow_train = datum.loc[training, f"{location} Mean Discharge (cfs)"].values
        gauge_train = datum.loc[training, f"{location} {variable} (ft)"].values
        nan_set = datum[f"{location} {variable} (ft)"].isna()
        flow_test = datum.loc[nan_set, f"{location} Mean Discharge (cfs)"].values
        
        eps = 1e-6
        flow_train = np.log(flow_train+eps)
        gauge_train = np.log(gauge_train+eps)
        flow_test = np.log(flow_test+eps)
        flow_test = flow_test.reshape(-1,1)

        sort = np.argsort(flow_train)
        flow_sort = flow_train[sort].reshape(-1, 1)
        gauge_sort = gauge_train[sort]

        return flow_sort, gauge_sort, flow_test, nan_set
    

    def regression(flow_sort, gauge_sort, flow_test):    
        regr = LinearRegression()
        regr.fit(flow_sort, gauge_sort)
        
        gauge_train_predict = regr.predict(flow_sort)
        mse = mean_squared_error(gauge_sort,gauge_train_predict)

        gauge_predict = regr.predict(flow_test)
        gauge_predict = np.exp(gauge_predict)-eps
        return mse, gauge_predict

    up_flow_sort, up_gauge_sort, up_flow_test, nanu_set = set_data('Upstream',variable,up)
    up_MSE, up_gauge_predict = regression(up_flow_sort, up_gauge_sort, up_flow_test)
    up.loc[nanu_set, f"Upstream {variable} (ft)"] = up_gauge_predict

    down_flow_sort, down_gauge_sort, down_flow_test, nand_set = set_data('Downstream',variable,down)
    down_MSE, down_gauge_predict = regression(down_flow_sort, down_gauge_sort, down_flow_test)
    down.loc[nand_set, f"Downstream {variable} (ft)"] = down_gauge_predict

    data = pd.merge(up,down,how='outer',on='Date')

    return data, up_MSE, down_MSE
