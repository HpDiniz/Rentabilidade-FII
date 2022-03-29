import pickle
import pandas as pd
from dateutil.relativedelta import relativedelta

from flask import Flask, render_template

app = Flask(__name__)

if __name__ == "__main__":
	app.run(host='0.0.0.0')

@app.route('/')
def index():
    return predictFIIs(2)

@app.route('/<int:add_months>')
def predictFIIs(add_months):

    first_day = pd.to_datetime('today').replace(day=1,hour=0,minute=0,second=0,microsecond=0)

    mes_alvo = []

    for i in range(6):
        mes_alvo.append((first_day + relativedelta(months=(i))).strftime("%m/%Y"))

    # Step 2
    with open('outputs/'+ mes_alvo[add_months-1].replace("/","") + '.dictionary', 'rb') as config_dictionary_file:
        # Step 3
        response = pickle.load(config_dictionary_file)

    return render_template('index.html', response=response)