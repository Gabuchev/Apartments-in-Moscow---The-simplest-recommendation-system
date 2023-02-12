import pandas as pd  
import pickle 
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score




###################################################################################################################################    
class DataCSV:
  # def __init__(self, Cost, Square, Distance, Floor, Description, MetroName,CenterDistance):

   def __init__(self, Cost, Square, Distance, Floor, Floors, Description, MetroName,CenterDistance):
     
      self.Cost  = Cost 
      self.Square = Square
      self.Distance = Distance
      self.Floor  = Floor 
      self.Floors = Floors
      self.Description = Description
      self.CenterDistance = CenterDistance
      self.MetroName = MetroName
     
#Запаолняю дата-лист

      DataList = []
      DataList.append({'cost(₽)': self.Cost,'square(м²)': self.Square, 'distance(м)': self.Distance, "center_distance/m": self.CenterDistance,"floor": self.Floor,
      "floors":self.Floors, "description":self.Description,"metro_name":self.MetroName})
      #DataList.append({'cost(₽)': self.Cost,'square(м²)': self.Square, 'distance(м)': self.Distance, "center_distance/m": self.CenterDistance,"floor": self.Floor,
      #"description":self.Description,"metro_name":self.MetroName})
   

#Сохраняю даталист в виде фрейма данных НА ДАЛЬНЕЙШУЮ ОБРАБОТКУ



      df = pd.DataFrame(DataList).to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv', index=False) #  **********  DFInput.csv  ********** 
      df = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')  # ЭТО НАДО!
      df = df.drop(['floors'],axis=1)
      #Получаю столбец с ценой квадратного метра

      #df['cost_for_meter(₽)'] =round((df['cost(₽)'])/(df['square(м²)']).astype("float"),1)  # получаю столбец с ценой квадратного метра
      #df = df[['cost(₽)','square(м²)','cost_for_meter(₽)','metro_name','distance(м)','center_distance/m','floor','floors','description']]#порядок столбцов
      df = df[['cost(₽)','square(м²)','metro_name','distance(м)','center_distance/m','floor','description']]#порядок столбцов
      #df['floors']=df['floors'].astype("int")# перевожу Этажность дома в тип int
      df['floor']=df['floor'].astype("int")# перевожу Этаж в тип int

      df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput1.csv',index=False)

      #df_1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
      #self.df = df_1 # ЗДЕСЬ ПРОВЕРИТЬ ЗНАЧЕНИЕ ПЕРЕДАЕТСЯ ДАЛЕЕ ИЛИ ОСТАЕТСЯ КАК ЕСТЬ

                # ТАБЛИЦА ВВЕДЕННЫХ ДАННЫХ
      df_table = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
      df_table['cost_for_meter(₽)'] =round((df['cost(₽)'])/(df['square(м²)']).astype("float"),1)
      df_table['floor']=df_table['floor'].astype("int")
      df_table['floors']=df_table['floors'].astype("int")
      df_table.rename(columns = {'cost(₽)' : 'Цена(₽)','square(м²)' : 'Площадь(м²)','cost_for_meter(₽)' : 'Цена за кв.м(₽)','metro_name' : 'Название метро', 
      'distance(м)' : 'Расстояние до метро','center_distance/m' : 'до центра','floor' : 'Этаж',
      'floors' : 'Этажность','description' : 'Описание'}, inplace = True) # меняю наименование на корректное

      df_table = df_table[['Цена(₽)','Площадь(м²)','Этаж','Этажность','Название метро','Расстояние до метро','Описание']]

      #df.rename(columns = {'cost(₽)' : 'Цена','square(м²)' : 'Площадь','metro_name' : 'Название метро', 
      #'distance(м)' : 'Расстояние до метро','center_distance/m' : 'Расстояние до центра','floor' : 'Этаж',
      #'description' : 'Описание'}, inplace = True) # меняю наименование на корректное

      df_table.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTableInput.csv',index=False) # **********  DFTableInput.csv  ********** 
      
###################################################################################################################################  





###################################################################################################################################    

   def RecAll (self): #Обрабатываю слова 

        df_1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput1.csv')

        #nlp = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl", "rb"))
        with open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl","rb") as nlp_1:
             nlp = pickle.Unpickler(nlp_1).load()
       
        def cleaning(doc):
            txt = [token.lemma_ for token in doc]
            return ' '.join(txt)
          #Удаляю неалфавитные символы:
        brief_cleaning = (re.sub("[^А-Яа-я]", ' ', str(row)).lower() for row in df_1['description'])
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=1000, n_process=1)]

        #Помещаю результаты в фрейм данных
        dfD = pd.DataFrame({'description': txt})
        #Удаляю пробелы которых больше или равно 2 
        dfD = dfD['description'].str.replace(' {2,}', ' ', regex=True)
                        # ДАТАФРЕЙМ С ОПИСАНИЕМ ДЛЯ ВВЕДЕННЫХ ДАННЫХ
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFDescription.csv',index=False) # **********  DFDescription.csv  **********
        
        ###################################################################################################################################

        DDf= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFDescription.csv')   # НАДО!     

        #Загружаю модель doc2vec


        #loaded_model = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb"))  
        with open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb") as loaded_model_1:
             loaded_model = pickle.Unpickler(loaded_model_1).load()

        #Генерация векторов
        doc2vec = [loaded_model.infer_vector((DDf['description'][i].split(' '))) for i in range(0,len(DDf['description']))]
        Vec64 = pd.DataFrame(data = doc2vec)  

        #Vec64.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)

        #КОНКАТЕНАЦИЯ (получаю набор данных с векторами по описанию)

        Data = pd.concat([df_1, Vec64], axis=1)
        Data = Data.drop(['description'], axis=1)#Удаляю столбец с описанием  

        ###################################################################################################################################

        #ДАТА СЕТ ДЛЯ ВВЕДЕННЫХ ДАННЫХ БЕЗ НОРМАЛИЗАЦИИ БЕЗ МАСШТАБИРОВАНИЯ БЕЗ ЗНАЧЕНИЯ МЕТРО

        Data.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess.csv',index=False)     # **********  DFProcess.csv  **********

        #Считываю данные из Metro_encoders.csv

        Metro_Encoder_all = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Metro_encoders.csv')

        #Сопоставлю каждому названию метро свой код из Metro_encoders.csv

        Metro_Encoder = Metro_Encoder_all[Metro_Encoder_all['metro_name'] == self.MetroName]["Metro_encoders"].values

        #Metro_Encoder=str(Metro_Encoder).strip('[]')
        Metro_encoders_Input = pd.DataFrame(data = Metro_Encoder).astype("int")
        Metro_encoders_Input.rename(columns = {0: "Metro_encoders"}, inplace = True) #ПЕРЕИМЕНОВЫВАЮ СТОЛБЕЦ НА "Metro_encoders"

        #КОНКАТЕНАЦИЯ

        DData=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess.csv')
        Data_all = pd.concat([DData, Metro_encoders_Input], axis=1) #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА

        #удаляю metro_name
        Data_all = Data_all.drop(['metro_name'],axis=1)

        #Data_all.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv',index=False)

        ###################################################################################################################################

        #Нормировка:
        #Data_all=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv')

        data1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data2.csv') #ЗНАЧЕНИЯ ДАННЫХ ВСЕГО ДАТАСЕТА

        #КОНКАТЕНАЦИЯ
        frame = [data1 , Data_all] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА
        data2 = pd.concat(frame)

        data2.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess2.csv',index=False) # **********    DFProcess2    **********

        #МАСШТАБИРОВАНИЕ

        data2 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess2.csv')
        



        scaler = preprocessing.MinMaxScaler()

        #scaler = StandardScaler()


        dataNorm_ALL = scaler.fit_transform(data2)

        dataNorm_PD = pd.DataFrame(data = dataNorm_ALL,columns = data2.columns.values)     

        csr_Ddata = csr_matrix(dataNorm_PD.values)

        print(csr_Ddata[::])

        recommendations = 10

        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata)
        distances, indices = knn.kneighbors(csr_Ddata[-1], n_neighbors = recommendations + 1)

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))    

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')

        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])
        
        recom_df = pd.merge(DFComplete, dataСlean, how='left',on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)

        recom_df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree.csv',index=False)
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')
        RecDiagnostics()
   
#############################################################################################################################################################################################################################

   def RecCost (self):

        #Загружаю данные, удаляю описание

        df_2 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput1.csv')
        ###df_2 = df_2.drop(['description'],axis=1)
        df_2 = df_2.drop(['description'],axis=1)
        
        ###################################################################################################################################

        #ДАТА СЕТ ДЛЯ ВВЕДЕННЫХ ДАННЫХ БЕЗ НОРМАЛИЗАЦИИ БЕЗ МАСШТАБИРОВАНИЯ БЕЗ ЗНАЧЕНИЯ МЕТРО

        df_2.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost.csv',index=False)    # **********  DFProcessCost.csv  **********

        #Считываю данные из Metro_encoders.csv

        Metro_Encoder_all = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Metro_encoders.csv')

        #Сопоставлю каждому названию метро свой код из Metro_encoders.csv

        Metro_Encoder = Metro_Encoder_all[Metro_Encoder_all['metro_name'] == self.MetroName]["Metro_encoders"].values

        #Metro_Encoder=str(Metro_Encoder).strip('[]')
        Metro_encoders_Input = pd.DataFrame(data = Metro_Encoder).astype("int")
        Metro_encoders_Input.rename(columns = {0: "Metro_encoders"}, inplace = True) #ПЕРЕИМЕНОВЫВАЮ СТОЛБЕЦ НА "Metro_encoders"

        #КОНКАТЕНАЦИЯ

        DData=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost.csv')
        Data_all = pd.concat([DData, Metro_encoders_Input], axis=1) #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ СТРОКИ ДАТАСЕТА axis=1

        #удаляю metro_name
        Data_all = Data_all.drop(['metro_name'],axis=1)

        Data_all.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\Proverka.csv',index=False)

        ###################################################################################################################################

        #Нормировка:
        #Data_all=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv')

        data3 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data2.csv') #ЗНАЧЕНИЯ ДАННЫХ ВСЕГО ДАТАСЕТА
       ################# data3 = data3[['cost(₽)', 'square(м²)', 'cost_for_meter(₽)', 'distance(м)','center_distance/m', 'floor','floors', 'Metro_encoders']]
        data3 = data3[['cost(₽)', 'square(м²)', 'distance(м)','center_distance/m', 'floor','Metro_encoders']]

        #КОНКАТЕНАЦИЯ
        frame = [data3 , Data_all] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА
        data4 = pd.concat(frame)

        data4.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost2.csv',index=False) # **********    DFProcessCost2.csv   **********

        #МАСШТАБИРОВАНИЕ

        data4 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost2.csv')
        print(data4)
        print('____________________________________________________________________')

        scaler = preprocessing.MinMaxScaler()
        #scaler = StandardScaler()

        dataNorm_ALL = scaler.fit_transform(data4)
        print(dataNorm_ALL)
        dataNorm_PD = pd.DataFrame(data = dataNorm_ALL,columns = data4.columns.values)       
        csr_Ddata1 = csr_matrix(dataNorm_PD.values)

        #АЛГОРИТМ - KNN 

        recommendations = 10
        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata1)
        distances, indices = knn.kneighbors(csr_Ddata1[-1], n_neighbors = recommendations + 1) # -1 потому что погследний элемент добавлен конкатенацией

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')
                
        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])

        #DFComplete.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFComplete.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)
      
        recom_df = pd.merge(DFComplete, dataСlean, how='left', on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)
        recom_df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree.csv',index=False)
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')
        RecDiagnostics()
   
###################################################################################################################################

   def RecDescription (self):

        df_Description = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput1.csv')
        df_Description = df_Description[['description']]

        with open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl", "rb") as nlp_1:
             nlp = pickle.Unpickler(nlp_1).load()
        
       # nlp = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl", "rb"))

        def cleaning(doc):
          txt = [token.lemma_ for token in doc if not token.is_stop]
          return ' '.join(txt)
          #Удаляю неалфавитные символы:

        brief_cleaning = (re.sub("[^А-Яа-я]", ' ', str(row)).lower() for row in df_Description['description'])
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=1000, n_process=1)]

        #Помещаю результаты в фрейм данных
        dfD = pd.DataFrame({'description': txt})
        #Удаляю пробелы которых больше или равно 2 
        dfD = dfD['description'].str.replace(' {2,}', ' ', regex=True)
                        # ДАТАФРЕЙМ С ОПИСАНИЕМ ДЛЯ ВВЕДЕННЫХ ДАННЫХ
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DF-DefRecDes.csv',index=False) # **********  DF-DefRecDes.csv  **********

        ###################################################################################################################################
        
        DDf= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DF-DefRecDes.csv')   # НАДО! 

        #Загружаю модель doc2vec
        loaded_model = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb")) 

        with open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb") as loaded_model_1:
             loaded_model = pickle.Unpickler(loaded_model_1).load()


        #Генерация векторов
        doc2vec = [loaded_model.infer_vector((DDf['description'][i].split(' '))) for i in range(0,len(DDf['description']))]

        Vec64 = pd.DataFrame(data = doc2vec) 

        Vec64.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)

        df_Vec = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64.csv') 

        Vec64= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv') 

        #КОНКАТЕНАЦИЯ (получаю набор данных с векторами по описанию)

        frame = [df_Vec , Vec64] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА

        Data = pd.concat(frame)

        Data.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data_Des_Input.csv',index=False)
        
        Data =pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data_Des_Input.csv') # **********  Data_Des_Inpu.csv  **********

        ###################################################################################################################################

        csr_Ddata2 = csr_matrix(Data.values)

        #АЛГОРИТМ - KNN 

        recommendations = 10

        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata2)
        distances, indices = knn.kneighbors(csr_Ddata2[-1], n_neighbors = recommendations + 1) # -1 потому что погследний элемент добавлен конкатенацией

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')
                
        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])

        #DFComplete.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFComplete.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)
      
        recom_df = pd.merge(DFComplete, dataСlean, how='left', on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)

        recom_df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree.csv',index=False)
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')       

        RecDiagnostics()


 #####################################################################################################################################################################################################      
   
def RecDiagnostics():

        Data = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree.csv')
        Metro_Encoder= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Metro_encoders.csv')
        res = Data.merge(Metro_Encoder, how="left") 
        res.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree-1.csv',index=False)
        res = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTree-1.csv')

        Data1=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
        Data1 = Data1.drop(['description'],axis=1)
        Data1 = Data1.merge(Metro_Encoder, how="left")
        Data1['CostForMeter'] =round((Data1['cost(₽)'])/(Data1['square(м²)']).astype("float"),1)
        Data1 = Data1.drop(['metro_name',"center_distance/m",'cost(₽)'],axis=1)
        Data1.rename(columns = {'square(м²)' : 'Square','distance(м)' : 'Distance','floor' : 'Floor','floors' : 'Floors'}, inplace = True)
        #Data1 = Data1[["Metro_encoders",'Square','Distance',"Floor","Floors","CostForMeter"]]#порядок столбцов
        Data1 = Data1[["Metro_encoders",'Square','Distance',"Floor","Floors"]]#порядок столбцов

        #X = res[["Metro_encoders",'Square','Distance',"Floor","Floors","CostForMeter"]]
        X = res[["Metro_encoders",'Square','Distance',"Floor","Floors"]]
        y = res['Cost']
   
        model = LinearRegression()
        y_pred = model.fit(X, y).predict(X)

 #         def diagnostics(y, y_pred):  
        residuals = y - y_pred
        residuals_mean = np.round(np.mean(y - y_pred), 3)
 
        #f, ((ax_rkde, ax_prob), (ax_ry, ax_auto), (ax_yy, ax_ykde)) = plt.subplots(nrows = 3, ncols = 2,figsize = (12, 18))
        f, ((ax_rkde, ax_prob, ax_ry), (ax_auto, ax_yy, ax_ykde)) = plt.subplots(nrows = 2, ncols = 3,figsize = (12.9, 11))
 
  # в первом подграфике построим график плотности остатков
        sns.kdeplot(residuals, fill = True, ax = ax_rkde)
        ax_rkde.set_title('ГРАФИК ПЛОТНОСТИ ОСТАТКОВ', fontsize = 9)
        ax_rkde.set(xlabel = f'Остатки, среднее значение: {residuals_mean}')
        ax_rkde.set(ylabel = 'Плотность')
 
  # во втором - график нормальной вероятности остатков
        probplot(residuals/10000, dist = 'norm', plot = ax_prob)
        ax_prob.set_title('НОРМАЛЬНЫЙ ВЕРОЯТНОСТНЫЙ\n ГРАФИК ОСТАТКОВ', fontsize = 9)
        #ax_prob.set_title('', fontsize = 14)
 
  # в третьем - график остатков относительно прогноза
        ax_ry.scatter(y_pred, residuals/10000)
        #ax_ry.set_title('Predicted vs. Residuals', fontsize = 14)
        ax_ry.set_title('ГЕТЕРОСКЕДАСТИЧНОСТЬ', fontsize = 9)
        ax_ry.set(xlabel = 'Прогнозируемая цена')
        ax_ry.set(ylabel = 'Остатки/1000')
 
  # в четвертом - автокорреляцию остатков
        plot_acf(residuals, lags = 9, ax = ax_auto)
        ax_auto.set_title('АВТОКОРРЕЛЯЦИЯ ОСТАТКОВ', fontsize = 9)
        ax_auto.set(xlabel = f'Lags\nКритерий Дарбина-Уотсона: {durbin_watson(residuals).round(2)}')
        ax_auto.set(ylabel = 'Автокорреляция')
 
  # в пятом - сравним прогнозные и фактические значения
        ax_yy.scatter(y, y_pred)
        ax_yy.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw = 1)
        ax_yy.set_title('ФАКТ И ПРОГНОЗ', fontsize = 9)
        ax_yy.set(xlabel = 'Фактическая цена')
        ax_yy.set(ylabel = 'Прогнозируемая цена')
 
  # в шестом - сравним распределение истинной и прогнозной целевой переменных
        sns.kdeplot(y, fill = True, ax = ax_ykde, label = 'y_true')
        sns.kdeplot(y_pred, fill = True, ax = ax_ykde, label = 'y_pred')
        ax_ykde.set_title('ФАКТИЧЕСКОЕ И ПРОГНОЗИРУЕМОЕ РАСПРЕДЕЛЕНИЯ', fontsize = 8.5)
        ax_ykde.set(xlabel = 'Факт и Прогноз')
        ax_ykde.set(ylabel = 'Плотность')
        ax_ykde.legend(loc = 'upper right', prop = {'size': 10})

        plt.savefig(r'C:\Users\Daniel\project university\Csv_to_html\static\saved_figure.png',bbox_inches = 'tight')

        Diagnostics_list = []
                                                 #ЦЕНА 

        Predict_Cost1 = np.round(model.predict(Data1))
        Predict_Cost = Predict_Cost1[0]
               
                                                #МЕТРИКИ  
# squared = False дает RMSE
        RMSE = mean_squared_error(y, y_pred, squared = False)
        MAE = mean_absolute_error(y, y_pred)
        MAPE = mean_absolute_percentage_error(y, y_pred)
        model.score(X, y)
        r2=r2_score(y, y_pred)

        Diagnostics_list.append({"Predict_Cost" : Predict_Cost,"RMSE" : RMSE, "MAE": MAE, "MAPE" : MAPE, "r2":r2 } )
        Diagnostics_df = pd.DataFrame(Diagnostics_list)
        Diagnostics_df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Diagnostics_df.csv',index=False)

 