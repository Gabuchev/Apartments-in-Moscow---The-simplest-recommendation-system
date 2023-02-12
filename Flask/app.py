# importing flask
from flask import Flask, render_template, url_for, request,send_from_directory
# importing pandas module
import pandas as pd  
import csv
from ClassCSV import DataCSV

app = Flask(__name__,static_url_path='') 

#app.run()
menu = [{"name":"ПРОЕКТ", "url": "base"},
{"name":"РЕКОМЕНДАЦИИ", "url": "result"},
]
@app.route('/', methods=["POST","GET"])
def index():
      return render_template('index.html',title ="Ознакомительная страница", menu = menu)
###################################################################################################################################  
      
@app.route('/base')
def base():
    #df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\TableA.json')
    #data = df.to_dict('records')
    def send_img(path):
        return send_from_directory('static', path)
  
    return render_template('Rec.html',title ="Проект", menu = menu)

###################################################################################################################################  

#data = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\Сlean.csv',encoding='utf-8')

@app.route('/result', methods=["POST", "GET"])

def result():

    dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')
    metro = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\metro.csv',encoding='utf-8')
    #Обработка входных данных
    if request.method == 'POST':
       
        Cost = request.form['Cost']
        if Cost == '' :
            Cost = dataСlean["cost(₽)"].median()               
        Square = request.form['Square'] 
        if Square == '' :
            Square = dataСlean["square(м²)"].median()        
        Distance = request.form['Distance']
        if Distance == '' :
            Distance = dataСlean["distance(м)"].median()      
        Floor = request.form['Floor']
        if Floor == '' :
            Floor = dataСlean["floor"].median()      
        Floors = request.form['Floors']
        if Floors == '' :
            Floors = dataСlean["floors"].median()      
        Description = request.form['Description']
        if Description == '' :
            Description = "Без описания"       
        MetroName = request.form['metro_name']
        if MetroName == '_MISSING_' :
            CenterDistance = metro["center_distance/m"].median()
        else:
            CenterDistance = metro[metro['metro_name'] == MetroName]["center_distance/m"].values.astype("int")
            CenterDistance=str(CenterDistance).strip('[]') 
       
#################################################################################################################################  
            
        if request.form['button'] == 'all':

            DataCSV(Cost,Square, Distance, Floor, Floors, Description, MetroName,CenterDistance).RecAll()   
           #DataCSV(Cost,Square, Distance, Floor, Description, MetroName,CenterDistance).RecAll()   

        if request.form['button'] == 'cost':

            DataCSV(Cost,Square, Distance, Floor, Floors, Description, MetroName,CenterDistance).RecCost()
           #DataCSV(Cost,Square, Distance, Floor, Description, MetroName,CenterDistance).RecCost()  

        if request.form['button'] == 'description':

            DataCSV(Cost,Square, Distance, Floor, Floors, Description, MetroName,CenterDistance).RecDescription()
           # DataCSV(Cost,Square, Distance, Floor, Description, MetroName,CenterDistance).RecDescription()  

         
    #################################################################################################################################    


        
     #plt.tight_layout()

    recom_df = pd.read_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')
    dataResult = recom_df.to_dict('records') 
    #output1 = recom_df.head(5)
    #output = round(output1['Cost'].mean())
    Diagnostics_df = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Diagnostics_df.csv')
    output_Cost = Diagnostics_df["Predict_Cost"][0].round(decimals=5)
    output_Cost1 = Diagnostics_df["RMSE"][0].round(decimals=5)
    output_Cost2 = Diagnostics_df["MAE"][0].round(decimals=5)
    output_Cost3 = Diagnostics_df["MAPE"][0].round(decimals=5)
    output_Cost4 = Diagnostics_df["r2"][0].round(decimals=5)

    reader = csv.reader(open(r"C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTableInput.csv",encoding='utf-8'))
    return render_template('result.html',title ="Рекомендации", tableD = dataResult, csv = reader, menu = menu, output=output_Cost,output1=output_Cost1,
    output2=output_Cost2,output3=output_Cost3,output4=output_Cost4,user_image="saved_figure.png")
    #################################################################################################################################    

if __name__ == "__main__":
     app.run(debug=True)
