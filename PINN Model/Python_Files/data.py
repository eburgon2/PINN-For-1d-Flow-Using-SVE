import os
import sys
#print(sys.executable)
import pandas as pd
import datetime as dt
from dataretrieval import nwis
from dataretrieval import waterdata
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

    discharge.to_csv('../Data Files/Raw/Discharge.csv')
    return discharge

def get_width(up_ID, down_ID,start,end):
    W_u = pd.DataFrame()
    w, metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{up_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                #characteristic='Width of stream',
                usgsPCode='00004'
            )
    W_u['Date']= w['Activity_StartDate']
    W_u["Upstream Width (ft)"] = w['Result_Measure']

    W_d = pd.DataFrame()
    w, metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{down_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                #characteristic='Width of stream',
                usgsPCode='00004'
            )
    W_d['Date']= w['Activity_StartDate']
    W_d["Downstream Width (ft)"] = w['Result_Measure']

    width = pd.merge(W_d,W_u, how='outer', on='Date')
    width['Date'] = pd.to_datetime(width['Date'])
    width.set_index('Date',inplace=True)

    width.to_csv('../Data Files/Raw/Width.csv')
    return width

def get_depth_2_datum(up_ID, down_ID,start,end):
    #Upstream 
    g2d_meters, metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{up_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                usgsPCode='30207'
            )
    g2d_feet,metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{up_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                usgsPCode='00065'
            )

    g2d_meters['Result_Measure'] = g2d_meters['Result_Measure'] * 3.28 #converting to ft for the merging
    g2d_up = pd.concat([g2d_feet[['Activity_StartDate','Result_Measure']],
                    g2d_meters[['Activity_StartDate','Result_Measure']]],ignore_index=True)
    g2d_up = g2d_up.rename(columns={'Activity_StartDate':'Date','Result_Measure':'Upstream to Datum (ft)'})
    g2d_up['Date'] = pd.to_datetime(g2d_up['Date'])
    g2d_up.set_index('Date',inplace=True)
    g2d_up = g2d_up.groupby('Date', as_index=True).mean()
    g2d_up = g2d_up.sort_index()

    #Downstream
    g2d_meters, metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{down_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                usgsPCode='30207'
            )
    g2d_feet,metadata = waterdata.get_samples(
                monitoringLocationIdentifier=[f"USGS-{down_ID}"],
                activityStartDateLower= start, activityStartDateUpper= end,
                service='results',
                usgsPCode='00065'
            )

    g2d_meters['Result_Measure'] = g2d_meters['Result_Measure'] * 3.28 #converting to ft for the merging
    
    g2d_down = pd.concat([g2d_feet[['Activity_StartDate','Result_Measure']],
                    g2d_meters[['Activity_StartDate','Result_Measure']]],ignore_index=True)
    g2d_down = g2d_down.rename(columns={'Activity_StartDate':'Date','Result_Measure':'Downstream to Datum (ft)'})
    g2d_down['Date'] = pd.to_datetime(g2d_down['Date'])
    g2d_down.set_index('Date',inplace=True)
    g2d_down = g2d_down.groupby('Date', as_index=True).mean()
    g2d_down = g2d_down.sort_index()
    g2d = pd.merge(g2d_up,g2d_down,left_index=True, right_index=True,how='outer')
    
    g2d.to_csv('../Data Files/Raw/Depth_to_Datum.csv')
    
    return g2d

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

def usgs_surface(id,year1,year2):
    d, m = waterdata.get_continuous(monitoring_location_id= f"USGS-{id}",
                parameter_code='00065',
                time=f"{year1}-01-01/{year2}-12-31",
                properties=["time","value"])
    d.set_index("time",inplace=True)
    d.index = d.index.date
    d.index = pd.to_datetime(d.index)
    d.index.name = "Date"
    d = d.resample('D').first()

    return d
        

def get_surface(upid, downid, coupled_years):
    up_data = []
    down_data = []

    for i in coupled_years:
        up_data.append(usgs_surface(upid,i[0],i[1]))
        down_data.append(usgs_surface(downid,i[0],i[1]))
    
    up_total = pd.concat(up_data,axis=0,join='outer',ignore_index=False)
    up_total = up_total.rename(columns = {"value":"Upstream Gauge Depth (ft)"})
    down_total = pd.concat(down_data,axis=0,join='outer',ignore_index=False)
    down_total = down_total.rename(columns = {"value":"Downstream Gauge Depth (ft)"})
    surface = pd.merge(up_total,down_total,how='outer',on="Date")

    surface.to_csv('../Data Files/Raw/Gauge_Depth.csv')
    return surface
        

