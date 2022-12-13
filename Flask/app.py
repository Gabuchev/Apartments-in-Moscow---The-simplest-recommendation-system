# importing flask
from flask import Flask, render_template, url_for
# importing pandas module
import numpy as np
import pandas as pd  

app = Flask(__name__) 
menu = [{"name":"Характеристики объекта недвижимости", "url": "base"},
{"name":"Дополнительные характеристики", "url": "other"}, 
{"name":"Описание", "url": "des"}]

@app.route('/')
def index():
    return render_template('index.html',title ="Рекомендательная система подбора недвижимости", menu = menu)

@app.route('/base')
def base():
    df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableA.json')
    data = df.to_dict('records')
    return render_template('table.html',title ="Характеристики объекта недвижимости", tableA = data, menu = menu)

@app.route('/other')
def other():
    df1 = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableB.json')
    data1 = df1.to_dict('records')
    return render_template('tableB.html',title ="Дополнительные характеристики", tableB = data1, menu = menu)

@app.route('/des')
def des():
    df2 = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableC.json')
    data2 = df2.to_dict('records')
    return render_template('tableC.html',title ="Описание", tableC = data2, menu = menu)


    
if __name__ == "__main__":
    #app.run(host="localhost", port=int("5000"))
    app.run(debug=True)
