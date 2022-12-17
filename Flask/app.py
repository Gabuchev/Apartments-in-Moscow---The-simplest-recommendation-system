# importing flask
from flask import Flask, render_template, url_for, request
# importing pandas module
import pandas as pd  
import pickle 
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix



app = Flask(__name__) 
menu = [{"name":"ПОДГОТОВКА ДАННЫХ", "url": "base"},
{"name":"РЕКОМЕНДАЦИИ", "url": "result"},
]

@app.route('/', methods=["POST","GET"])
def index():
      return render_template('index.html',title ="Рекомендательная система подбора недвижимости", menu = menu)

@app.route('/base')
def base():
    #df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableA.json')
    #data = df.to_dict('records')
    return render_template('table.html',title ="ввйцвувцуауцацацуацу",  menu = menu)

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

        df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\Datapredict.csv')
             
        #Обрабатываю слова
        dfD = df['description'].str.replace(' {2,}', ' ', regex=True)
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DD.csv',index=False)
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
    dataResult = recom_df.to_dict('records')
    return render_template('result.html',title ="Рекомендации",tableD = dataResult, menu = menu)

if __name__ == "__main__":
     app.run(debug=True)
