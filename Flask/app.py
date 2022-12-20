# importing flask
from flask import Flask, render_template, url_for, request,send_from_directory
# importing pandas module
import pandas as pd  
import pickle 
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import re
import csv


app = Flask(__name__,static_url_path='') 
menu = [{"name":"ПРОЕКТ", "url": "base"},
{"name":"РЕКОМЕНДАЦИИ", "url": "result"},
]

@app.route('/', methods=["POST","GET"])
def index():
      return render_template('index.html',title ="Ознакомительная страница", menu = menu)

@app.route('/base')
def base():
    #df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableA.json')
    #data = df.to_dict('records')
    def send_img(path):
        return send_from_directory('static', path)
    
    return render_template('Rec.html',title ="Проект", menu = menu)

#@app.route('/other')
#def other():
    #df1 = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableB.json')
    #data1 = df1.to_dict('records')
    #return render_template('tableB.html',title ="Дополнительные характеристики квартиры и дома", tableB = data1, menu = menu)

#@app.route('/des')
#def des():
    #df2 = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableC.json')
    #data2 = df2.to_dict('records')
    #return render_template('tableC.html',title ="Описание", tableC = data2, menu = menu)

data = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Сlean.csv',encoding='utf-8')

@app.route('/result', methods=["POST", "GET"])

def result():
    
    dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Сlean.csv',encoding='utf-8')
    metro = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\metro.csv',encoding='utf-8')
    DataList = []


    #Обработка входных данных
    if request.method == 'POST':
        

        Cost = request.form['Cost']
        if Cost == '' :
            Cost = dataСlean["Cost"].median()
                
        Square = request.form['Square'] 
        if Square == '' :
            Square = dataСlean["Square"].median()
        
        Distance = request.form['Distance']
        if Distance == '' :
            Distance = dataСlean["Distance"].median()
       
        Floor = request.form['Floor']
        if Floor == '' :
            Floor = dataСlean["Floor"].median()
       
        Floors = request.form['Floors']
        if Floors == '' :
            Floors = dataСlean["Floors"].median()
        
        Description = request.form['Description']
        if Description == '' :
            Description = "Без описания"
        
        MetroName = request.form['metro_name']
        if MetroName == '_MISSING_' :
            CenterDistance = metro["center_distance/m"].median()
        else:
            CenterDistance = metro[metro['metro_name'] == MetroName]["center_distance/m"].values.astype("int")
            CenterDistance=str(CenterDistance).strip('[]')

        DataList.append({'cost(₽)': Cost,'square(м²)': Square, 'distance(м)': Distance, "center_distance/m": CenterDistance,"floor": Floor,
        "floors":Floors, "description":Description,"metro_name":MetroName})

        df = pd.DataFrame(DataList).to_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv', index=False)

        df= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv')
     
        #Получаю столбец с ценой квадратного метра
        df['cost_for_meter(₽)'] =round((df['cost(₽)'])/(df['square(м²)']).astype("float"),1)  # получаю столбец с ценой квадратного метра

        df= df[['cost(₽)','square(м²)','cost_for_meter(₽)','metro_name','distance(м)','center_distance/m','floor','floors','description']]#порядок столбцов
        df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv',index=False)

   

        dfSecond_value= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv')
        dfSecond_value.rename(columns = {'cost(₽)' : 'Цена','square(м²)' : 'Площадь','cost_for_meter(₽)' : 'Цена за кв.м','metro_name' : 'Название метро', 
        'distance(м)' : 'Расстояние до метро','center_distance/m' : 'Расстояние до центра','floor' : 'Этаж',
        'floors' : 'Этажность дома','description' : 'Описание'}, inplace = True) # меняю наименование на корректное
        dfSecond_value['Этажность дома']=dfSecond_value['Этажность дома'].astype("int")# перевожу Этажность дома в тип int
        dfSecond_value['Этаж']=dfSecond_value['Этаж'].astype("int")# перевожу Этаж в тип int
        dfSecond_value.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\dfSecond_value.csv',index=False)
     
          
###################################################################################################################################
                                            #Обрабатываю слова
    
        nlp = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\nlppickle.pkl", "rb"))
        def cleaning(doc):
            #Лемматизируется и удаляет стоп-слова
            # doc должен быть пространственным объектом Doc
            txt = [token.lemma_ for token in doc if not token.is_stop]
            return ' '.join(txt)\
            # если предложение состоит всего из одного или двух слов,
            # польза от тренинга мала, поэтому
            #if len(txt) > 0:
                #return ' '.join(txt)\

        #Удаляю неалфавитные символы:
        brief_cleaning = (re.sub("[^А-Яа-я]", ' ', str(row)).lower() for row in df['description'])

        # Атрибут space .pipe() для ускорения процесса очистки:
        # Подбираю оптимальное batch_size=2000, n_process=4
        
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=1000, n_process=1)]
        #Помещаю результаты в фрейм данных
        dfD = pd.DataFrame({'description': txt})
        #Удаляю пробелы которых больше или равно 2 и сохраняю данные
        dfD = dfD['description'].str.replace(' {2,}', ' ', regex=True)
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DD.csv',index=False)
 
###################################################################################################################################

        DDf= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DD.csv')
        
        #Загружаю модель doc2vec
        loaded_model = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\d2vpickle.pkl", "rb"))
        
        #Генерация векторов
        doc2vec = [loaded_model.infer_vector((DDf['description'][i].split(' '))) for i in range(0,len(DDf['description']))]
        Vec64 = pd.DataFrame(data = doc2vec)
      
        #Конкатенация
        Data = pd.concat([df, Vec64], axis=1)
        Data=Data.drop(['description'], axis=1)#Удаляю столбец с описанием
        
        #Сохраняю csv
        Data.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData1.csv',index=False)

        #Считываю данные из Metro_encoders.csv
        Metro_Encoder_all = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Metro_encoders.csv')

        #Сопоставлю каждому названию метро свой код из Metro_encoders.csv
        Metro_Encoder = Metro_Encoder_all[Metro_Encoder_all['metro_name'] == MetroName]["Metro_encoders"].values
        #Metro_Encoder=str(Metro_Encoder).strip('[]')

        Metro_encoders = pd.DataFrame(data = Metro_Encoder)
        Metro_encoders.columns=['Metro_encoders']

        #Конкатенация
        DData=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData1.csv')
        Data_all = pd.concat([DData, Metro_encoders], axis=1)
        #удаляю metro_name
        Data_all= Data_all.drop(['metro_name'],axis=1)
        Data_all.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv',index=False)

        #Нормировка:

        Data_all=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv')

###################################################################################################################################

        # загружаю библиотеку препроцесинга данных
        # эта библиотека автоматически приведет данные к нормальным значениям

        data1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Data2.csv')
        frame = [data1, Data_all]
        data2 = pd.concat(frame)
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler()
        dataNorm_ALL = scaler.fit_transform(data2)
        dataNorm_PD = pd.DataFrame(data = dataNorm_ALL,columns = data1.columns.values)
        
        #dataNorm = dataNorm_PD.iloc[-1]

        #print(dataNorm_PD)
        #print(dataNorm_PD.iloc[-1])

        # # преобразую матрицу в формат csr
        csr_Ddata = csr_matrix(dataNorm_PD.iloc[-1].values)
        recommendations = 9
       # Загружаю модель knn
        knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        distances, indices = knn.kneighbors(csr_Ddata[0], n_neighbors = recommendations + 1)
        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))
     
###################################################################################################################################

        # отсортировываю список по расстояниям через key = lambda x: x[1] 
        # в возрастающем порядке reverse = False
        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)
        indices_distances_sorted = indices_distances_sorted[:]
      
        #Получаю рекомендации:

    
        # создаю пустой список, в который буду помещать парметры рекомендации и "расстояние" до нее

        recom_list = []
 
        # теперь в цикле буду поочередно проходить по кортежам
        for ind_dist in indices_distances_sorted:
 
            # поиск id в clean.csv
            clean_id = dataСlean.iloc[ind_dist[0]]['id']
            # индекс этого предложения
            id = dataСlean[dataСlean['id'] == clean_id].index
 
            # параметры и расстояние до рекомендации
            rooms = dataСlean.iloc[id]["Rooms"].values[0]
            cost = dataСlean.iloc[id]["Cost"].values[0]
            square = dataСlean.iloc[id]["Square"].values[0]
            cost_for_meter= dataСlean.iloc[id]["CostForMeter"].values[0]
            address= dataСlean.iloc[id]["Addres"].values[0]
            metro_name= dataСlean.iloc[id]["MetroName"].values[0]
            distance= dataСlean.iloc[id]["Distance"].values[0]
            center_distance= dataСlean.iloc[id]["CenterDistance"].values[0]
            floor= dataСlean.iloc[id]["Floor"].values[0]
            floors= dataСlean.iloc[id]["Floors"].values[0]
            description= dataСlean.iloc[id]["Description"].values[0]
            dist = ind_dist[1]
 
            # помещаю данные в словарь
            # который будет элементом списка recom_list
            recom_list.append({"Rooms" : rooms,"Cost": cost,"Square":square,"CostForMeter":cost_for_meter,
                        "Addres":address,"MetroName":metro_name,"Distance":distance,
                        "CenterDistance":center_distance,"Floor":floor,"Floors":floors,"Description":description,'CosDis': dist})

        print(recom_list[0])

        recom_df = pd.DataFrame(recom_list, index = range(0, recommendations+1))
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\recom_df.json',orient="records")

    recom_df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\recom_df.json')
    #recom_df_D = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv')
    #recom_df_D = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.json')
    dataResult = recom_df.to_dict('records')
    #dataResult_D = recom_df_D.to_dict('records')
    
    reader = csv.reader(open(r"C:\Users\Daniel\project university\Csv_to_html\dfSecond_value.csv",encoding='utf-8') )
    return render_template('result.html',title ="Рекомендации",tableD = dataResult, csv=reader, menu = menu)

if __name__ == "__main__":
     app.run(debug=True)
