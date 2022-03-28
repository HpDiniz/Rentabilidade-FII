import re
import bs4
import json
import requests
import datetime
import dateutil

from datetime import date
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

from xgboost import XGBRegressor
from flask import Flask, render_template

app = Flask(__name__)

if __name__ == "__main__":
	app.run(host='0.0.0.0')
    
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

@app.route('/')
def index():
    return predictFIIs(2)

    #response = {'fundos': [{'codigo': 'RBDS11',
    #'dividendos futuro': 'R$ 1.16',
    #'ordem': '1',
    #'preço futuro': 'R$ 7.01',
    #'rentabilidade': 0.165},
    #{'codigo': 'KNRE11',
    #'dividendos futuro': 'R$ 0.04',
    #'ordem': '2',
    #'preço futuro': 'R$ 0.91',
    #'rentabilidade': 0.044},
    #{'codigo': 'PLRI11',
    #'dividendos futuro': 'R$ 1.22',
    #'ordem': '3',
    #'preço futuro': 'R$ 41.0',
    #'rentabilidade': 0.03}],
    #'mes_alvo': '04/2022',
    #'outros_meses': ['04/2022',
    #'05/2022',
    #'06/2022',
    #'07/2022',
    #'08/2022',
    #'09/2022'],
    #'qtd_meses': 6}

    #return render_template('index.html', response=response)#response=response)

@app.route('/<int:add_months>')
def predictFIIs(add_months):

    today = date.today().strftime("%d/%m/%Y")
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

    predict_months = int(add_months)

    for t in df['Códigodo fundo']:
        print("Predicting " + t + "...")
        idx = df[(df['Códigodo fundo'] == t)].index[0]
        df.loc[idx,['Preço futuro']] = realizaPredicao(dfs[t],'Close', predict_months)
        df.loc[idx,['Dividendos futuros']] = realizaPredicao(dfs[t],'Dividends', predict_months)

    df['Rentabilidade'] = df['Dividendos futuros']/df['Preço futuro']

    meses = []

    for i in range(-1,5):
        mes_alvo = (first_day + relativedelta(months=(predict_months+i))).strftime("%m/%Y")
        meses.append(mes_alvo)

    mes_alvo = meses[0]

    top3_fii = df.nlargest(3, 'Rentabilidade')
    top3_fii = top3_fii[['Códigodo fundo','Rentabilidade','Preço futuro','Dividendos futuros']]
    top3_fii.reset_index(inplace=True)

    response = {
        "mes_alvo": mes_alvo,
        "outros_meses": meses,
        "qtd_meses": len(meses),
        "fundos": [{
            "ordem": "1",
            "codigo": top3_fii.iloc[0]["Códigodo fundo"],
            "rentabilidade": round(top3_fii.iloc[0]["Rentabilidade"],3),
            "preço futuro": "R$ " + str(round(top3_fii.iloc[0]["Preço futuro"],2)),
            "dividendos futuro": "R$ " + str(round(top3_fii.iloc[0]["Dividendos futuros"],2))
        },
        {
            "ordem": "2",
            "codigo": top3_fii.iloc[1]["Códigodo fundo"],
            "rentabilidade": round(top3_fii.iloc[1]["Rentabilidade"],3),
            "preço futuro": "R$ " + str(round(top3_fii.iloc[1]["Preço futuro"],2)),
            "dividendos futuro": "R$ " + str(round(top3_fii.iloc[1]["Dividendos futuros"],2))
        },
        {
            "ordem": "3",
            "codigo": top3_fii.iloc[2]["Códigodo fundo"],
            "rentabilidade": round(top3_fii.iloc[2]["Rentabilidade"],3),
            "preço futuro": "R$ " + str(round(top3_fii.iloc[2]["Preço futuro"],2)),
            "dividendos futuro": "R$ " + str(round(top3_fii.iloc[2]["Dividendos futuros"],2))
        }]
    }

    return render_template('index.html', response=response)
