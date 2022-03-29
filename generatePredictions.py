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
from dateutil.relativedelta import relativedelta

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
    
def converteData(datas):
    
    newArray = []
    meses = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    
    for data in datas:
        
        item = data.split("/")
        mes = str(meses.index(item[0])+1)
        mes = ("0" + mes)[len(mes)-1:len(mes)+1]
        
        newArray.append(item[1] + "-" + mes + "-01 00:00:00")
        
    return newArray

def realizaPredicao(dataframe, column, month):

    predict_column = column + "_target"

    dataframe[predict_column] = dataframe[column].shift(-month)

    treino = dataframe[:-1]
    validacao = dataframe[-1:]

    if month > 1:
        treino = dataframe[:-month]
        validacao = dataframe[-month:-(month-1)]

    x_treino = treino.loc[:, [column]].values
    y_treino = treino.loc[:, [predict_column]].values

    modelo_xgb = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    modelo_xgb.fit(x_treino, y_treino)

    predicao = modelo_xgb.predict(validacao[column][-1])
    return (predicao[0])

def predictFIIs():

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

    df.sort_values(by=['Dividendo'],ascending=False)

    dfs = {}
    newColumn = []
    predict_months = 2

    for index, row in df.iterrows():

        t = str(row['Códigodo fundo'])

        ticker = yf.Ticker(t + '.SA')
        aux = ticker.history(interval='1mo',period="max")

        dados_recentes = False
        for data in aux.index:
            if last_month in str(data) or this_month in str(data):
                dados_recentes = True

        if dados_recentes == False:
            print("FII " + t + " removido por não conter dados recentes.")
            df.drop(index, inplace=True)
        elif aux.empty or len(aux.index) < 20:
            print("FII " + t + " removido por pouca quantidade de dados (" + str(len(aux.index)) + ")")
            df.drop(index, inplace=True)
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

    columns = {'Código': [],'Endereço': [], 'Bairro': [], 'Cidade': [], 'Área Bruta Locável': []}  
    df_ativos = pd.DataFrame(columns)   

    print("Total de fundos restantes: " + str(len(dfs)))

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
        datas = converteData(labels)

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
    
    for y in range(6):

        predict_months = y+1

        for t in df['Códigodo fundo']:
            print("Predicting " + t + "...")
            idx = df[(df['Códigodo fundo'] == t)].index[0]
            df.loc[idx,['Preço futuro']] = realizaPredicao(dfs[t],'Close', predict_months)
            df.loc[idx,['Dividendos futuros']] = realizaPredicao(dfs[t],'Dividends', predict_months)

        df['Rentabilidade'] = df['Dividendos futuros']/df['Preço futuro']

        meses = []

        for i in range(6):
            meses.append((first_day + relativedelta(months=(i))).strftime("%m/%Y"))

        mes_alvo = meses[y]

        top3_fii = df.nlargest(3, 'Rentabilidade')
        top3_fii = top3_fii[['Códigodo fundo','Rentabilidade','Preço futuro','Dividendos futuros']]
        top3_fii.reset_index(inplace=True)

        fundos_ranking = []

        for i in range(3):
            fundos_ranking.append({
                "ordem": str(i+1),
                "codigo": top3_fii.iloc[i]["Códigodo fundo"],
                "rentabilidade": round(top3_fii.iloc[i]["Rentabilidade"],3),
                "preço futuro": "R$ " + str(round(top3_fii.iloc[i]["Preço futuro"],2)),
                "dividendos futuro": "R$ " + str(round(top3_fii.iloc[i]["Dividendos futuros"],2))
            })

        response = response = {
            "mes_alvo": mes_alvo,
            "outros_meses": meses,
            "qtd_meses": len(meses),
            "fundos": fundos_ranking
        }

        with open('outputs/'+ mes_alvo.replace("/","") + '.dictionary', 'wb') as config_dictionary_file:
            pickle.dump(response, config_dictionary_file)

predictFIIs()