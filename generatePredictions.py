import re
import bs4
import json
import pickle
import requests
import datetime
import dateutil

from datetime import date
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd
import yfinance as yf

import warnings; 
warnings.simplefilter('ignore')
    
def updateColumn(columnName, line, newRow):
    if '<b>'+columnName+':</b>' in line:
        item = line.replace('<b>'+columnName+':</b>',"").replace("<li>","")
        item = item.replace("</li>","").replace("m<sup>2</sup>","")
        item = item.replace("N/A","").strip()
        newRow[columnName] = item
    
def converteData(datas, monthYearOnly):

    if monthYearOnly:
        return (datas.split('-')[1] + "/" + datas.split('-')[0])
    else:
        newArray = []
        meses = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
        
        for data in datas:
            
            item = data.split("/")
            mes = str(meses.index(item[0])+1)
            mes = ("0" + mes)[len(mes)-1:len(mes)+1]
            
            newArray.append(item[1] + "-" + mes + "-01 00:00:00")
            
        return newArray

def xgboostPrediction(dataframe, column, month):

    new_df = dataframe.copy()

    predict_column = column + "_target"

    new_df[predict_column] = new_df[column].shift(-month)

    treino = new_df[:-1]
    validacao = new_df[-1:]

    if month > 1:
        treino = new_df[:-month]
        validacao = new_df[-month:-(month-1)]

    x_treino = treino.loc[:, [column]].values
    y_treino = treino.loc[:, [predict_column]].values

    modelo_xgb = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    modelo_xgb.fit(x_treino, y_treino)

    predicao = modelo_xgb.predict(validacao[column][-1])
    return (predicao[0])

def sarimaxPrediction(dataframe, columns, months):

    result_df = None

    for col in columns:
      arima_df_aux = dataframe[[col]] 
      fit_arima = auto_arima(arima_df_aux, d=1, start_p=1, start_q=1, max_p=3, mar_q=3, seasonal=True, m=6, D=1, start_P=1,start_Q=1, max_P=2, max_Q=2, information_criterion='aic', trace=False, error_action='ignore', stepwise=True)
      
      model=SARIMAX(arima_df_aux, order=fit_arima.order, seasonal_order=fit_arima.seasonal_order)
      resultado_sarimax = model.fit()

      forecast = resultado_sarimax.get_forecast(steps=months)

      if result_df is None:
        result_df = pd.DataFrame.from_dict(forecast.predicted_mean)

      result_df[col] = forecast.predicted_mean

    return (result_df[columns])

def traditionalPredict(dfs, ativo, columns, months):

  dataframe = dfs[ativo]

  result_df = sarimaxPrediction(dataframe, columns, months)

  result_df['Rentabilidade'] = result_df['Dividends']/result_df['Close']

  result_df['Códigodo fundo'] = ativo
  
  return result_df

def machineLearningPredict(dfs, ativo, columns, months_to_predict):

  xgboost_df = None

  dataframe = dfs[ativo]

  for m in range(len(months_to_predict)):

    predict_months = m+1

    result = {}

    for col in columns:

      result[col] = xgboostPrediction(dataframe,col, predict_months)

    result['Rentabilidade'] = result['Dividends']/result['Close']

    result['Códigodo fundo'] = ativo
    result['Date'] = months_to_predict[m]

    result_df = pd.DataFrame.from_dict([result])

    result_df = result_df.set_index('Date')
    result_df.index = pd.to_datetime(result_df.index)

    if xgboost_df is None:
      xgboost_df = result_df.copy()
    else:
      xgboost_df = pd.concat([xgboost_df, result_df])
      
  return xgboost_df

def predictFIIs(qnt_months_to_predict):

    first_day = pd.to_datetime('today').replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    this_month = (first_day).strftime("%Y-%m")
    last_month = (first_day - relativedelta(months=1)).strftime("%Y-%m")
    headers = {
        'User-Agent': 
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36'
            ' (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'
    }
    url = 'https://www.fundsexplorer.com.br/ranking'

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        df = pd.read_html(response.content, encoding='utf-8')[0]

    df.sort_values('Códigodo fundo', inplace=True)

    categorical_columns = ['Códigodo fundo','Setor']

    idx = df[df['Setor'].isna()].index
    df.drop(idx, inplace=True)

    df[categorical_columns] = df[categorical_columns].astype('category')

    col_floats = list(df.iloc[:,2:-1].columns)

    df[col_floats] = df[col_floats].fillna(value=0)

    df[col_floats] = df[col_floats].applymap(lambda x: str(x).replace('R$', '').replace('.0','').replace('.','').replace('%','').replace(',','.'))

    df[col_floats] = df[col_floats].astype('float')

    idx = df[np.isinf(df[col_floats]).any(1)].index
    df.drop(idx, inplace=True)

    df['P/VPA'] = df['P/VPA']/100

    indicadores = [
        'Códigodo fundo',
        'Setor', 
        'DY (12M)Acumulado', 
        'VacânciaFísica', 
        'VacânciaFinanceira', 
        'P/VPA', 
        'QuantidadeAtivos', 
        'Liquidez Diária'
    ]

    df_aux = df[indicadores]

    media_setor = df_aux.groupby('Setor').agg(['mean','std'])

    media_setor.loc['Residencial', ('DY (12M)Acumulado', 'mean')]

    dfs = {}
    remover_fundos = []

    for t in df['Códigodo fundo']:

        ticker = yf.Ticker(t + '.SA')
        aux = ticker.history(interval='1mo',period="max")

        dados_recentes = False
        for data in aux.index:
            if last_month in str(data) or this_month in str(data):
                dados_recentes = True

        if dados_recentes == False:
            print("FII " + t + " será removido por não conter dados recentes.")
            remover_fundos.append(t)
        elif aux.empty or len(aux.index) < 20:
            print("FII " + t + " será removido por pouca quantidade de dados (" + str(len(aux.index)) + ")")
            remover_fundos.append(t)
        else:
            print('Lendo FII {}...'.format(t))
            aux.reset_index(inplace=True)
            aux['ticker'] = t

            new_dates = []
            add_month = dateutil.relativedelta.relativedelta(months=1)

            for index, row in aux.iterrows():

                newDate = datetime.datetime(row['Date'].year, row['Date'].month, 1)

                if(row['Date'].day > 15):
                    newDate = newDate + add_month
                new_dates.append(newDate)

            aux['new_dates'] = new_dates

            dfs[t] = aux
            #df[df['Códigodo fundo'] == t]['Preço futuro (1m)'] = 
            #dfs[t]['mm5d'] = dfs[t]['Close'].rolling(5).mean()
            #dfs[t]['mm21d'] = dfs[t]['Close'].rolling(21).mean()
            dfs[t] = dfs[t].shift(-1)
            dfs[t].dropna(inplace=True)
            dfs[t] = dfs[t].reset_index(drop=True)
            dfs[t] = dfs[t][dfs[t]['new_dates'] < first_day]
            dfs[t] = dfs[t].set_index('new_dates')

            try:
                dfs[t].index.freq = 'MS'
            except:
                dfs[t].index.freq = None
                remover_fundos.append(t)
                del dfs[t]
                print("FII " + t + " será removido por estar com dados faltantes.")
            
    df = df[~df.isin(remover_fundos).any(axis=1)]

    print("Total de fundos restantes: " + str(len(dfs)))

    columns = {'Código': [],'Endereço': [], 'Bairro': [], 'Cidade': [], 'Área Bruta Locável': []}  
    df_ativos = pd.DataFrame(columns)   

    for fundo in dfs:

        url = 'https://www.fundsexplorer.com.br/funds/' + fundo
        response = requests.get(url, headers=headers)

        soup = bs4.BeautifulSoup(response.content, "html")
        div = soup.find("div", {"id": "dividends-chart-wrapper"})

        labels = re.findall('"labels":\[.*?\]', str(div))
        dividends = re.findall('"data":\[.*?\]', str(div))

        # parse data:
        dividends = json.loads("{" + dividends[0] + "}")['data']
        labels = json.loads("{" + labels[0] + "}")['labels']

        # converte "Março/2021" para "2021-03-01"
        datas = converteData(labels, False)

        for i in range(len(datas)):
            dfs[fundo].loc[dfs[fundo].index[dfs[fundo].index == datas[i]],'Dividends'] = dividends[i]

        if ('Ativos do ') in str(response.content):

            print("Coletando ativos de " + fundo + "...")

            soup = BeautifulSoup(response.content,"lxml")
            w3schollsList = soup.find("div",id="fund-actives-items")

            lista = w3schollsList.find_all('ul')

            for l in lista:

                newRow = {}

                itemList = l.find_all('li')
                for it in itemList:
                    line = str(it)

                    for column in ['Endereço', 'Bairro', 'Cidade', 'Área Bruta Locável']:
                        updateColumn(column, line, newRow)

                if len(newRow) > 0:
                    newRow['Código'] = fundo
                    df_ativos = df_ativos.append(newRow, ignore_index = True)
        else:
            print(fundo + " não possui ativos.")

    columns_str = ['Código','Endereço', 'Bairro', 'Cidade']
    df_ativos[columns_str] = df_ativos[columns_str].astype("string")

    # Calculando datas de interesse
    months_to_predict = []

    for i in range(qnt_months_to_predict):
        add_month = dateutil.relativedelta.relativedelta(months=i)
        months_to_predict.append((first_day + add_month).strftime("%Y-%m-%d"))

    sarimax_dfs = None

    for t in df['Códigodo fundo']:
        print("Predicting with ARIMA: " + t + "...")
        aux_df = traditionalPredict(dfs,t,['Close','Dividends'],qnt_months_to_predict)

        if sarimax_dfs is None:
            sarimax_dfs = aux_df.copy()
        else:
            sarimax_dfs = pd.concat([sarimax_dfs, aux_df])

    xgboost_dfs = None

    for t in df['Códigodo fundo']:
        print("Predicting with Machine Learning: " + t + "...")
        aux_df = machineLearningPredict(dfs,t,['Close','Dividends'],months_to_predict)

        if xgboost_dfs is None:
            xgboost_dfs = aux_df.copy()
        else:
            xgboost_dfs = pd.concat([xgboost_dfs, aux_df])

    df_medio = None

    for t in df['Códigodo fundo']:
        xgb_df = xgboost_dfs[(xgboost_dfs['Códigodo fundo'] == t)].copy()
        sar_df = sarimax_dfs[(sarimax_dfs['Códigodo fundo'] == t)].copy()
        sar_xbg_mean = xgb_df.copy()

        sar_xbg_mean['Close'] = (xgb_df['Close'] + sar_df['Close'])/2
        sar_xbg_mean['Dividends'] = (xgb_df['Dividends'] + sar_df['Dividends'])/2
        sar_xbg_mean['Rentabilidade'] = sar_xbg_mean['Dividends']/sar_xbg_mean['Close']

        if df_medio is None:
            df_medio = sar_xbg_mean.copy()
        else:
            df_medio = pd.concat([df_medio, sar_xbg_mean])

    meses = []
    for m in months_to_predict:
        meses.append(converteData(m,True))

    for m in months_to_predict:

        mesAno = converteData(m,True)

        print("Salvando melhores resultados obtidos para " + mesAno + "...")

        top3_fii = df_medio[df_medio.index == m].nlargest(3, 'Rentabilidade')
        top3_fii = top3_fii[['Códigodo fundo','Rentabilidade','Close','Dividends']]
        top3_fii.reset_index(inplace=True)

        fundos_ranking = []

        for i in range(3):
            fundos_ranking.append({
                "ordem": str(i+1),
                "codigo": top3_fii.iloc[i]["Códigodo fundo"],
                "rentabilidade": round(top3_fii.iloc[i]["Rentabilidade"],3),
                "preço futuro": "R$ " + str(round(top3_fii.iloc[i]["Close"],2)),
                "dividendos futuro": "R$ " + str(round(top3_fii.iloc[i]["Dividends"],2))
            })

        response = {
            "mes_alvo": mesAno,
            "outros_meses": meses,
            "qtd_meses": len(meses),
            "fundos": fundos_ranking
        }

        with open('outputs/'+ mesAno.replace("/","") + '.dictionary', 'wb') as config_dictionary_file:
            pickle.dump(response, config_dictionary_file)

predictFIIs(6)