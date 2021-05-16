import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from pandas import DataFrame
from pycaret import regression


app = Flask(__name__)
model = regression.load_model(model_name = 'catboost')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    print(features)
    year = int(features[0])
    month = int(features[1])
    day = int(features[2])
    hour = int(features[3])
    minute = int(features[4])
    cloudtype = int(features[5])
    dewpoint = float(features[6])
    solarzenithangle = float(features[7])
    surfacealdebo = float(features[8])
    windspeed = float(features[9])
    precwater = float(features[10])
    winddirection = int(features[11])
    relativehumidity = float(features[12])
    temperature = float(features[13])
    pressure = int(features[14])
    final_features = []
    # a = [final_features]
    #a=[]
    #a =
    #print(a)
    final_features.append(year)
    final_features.append(month)
    final_features.append(day)
    final_features.append(hour)
    final_features.append(minute)
    final_features.append(cloudtype)
    final_features.append(dewpoint)
    final_features.append(solarzenithangle)
    final_features.append(surfacealdebo)
    final_features.append(windspeed)
    final_features.append(precwater)
    final_features.append(winddirection)
    final_features.append(relativehumidity)
    final_features.append(temperature)
    final_features.append(pressure)
    print(len(final_features))
    # df = DataFrame(a, columns=['Year', 'Month', 'Day', Hour, Minute, Cloud Type, Dew Point, Solar Zenith Angle, Surface Albedo, Wind Speed, Precipitable Water, Wind Direction, Relative Humidity, Temperature, Pressure])
    df = DataFrame({"Year": [final_features[0]],
                    "Month": [final_features[1]],
                    "Day": [final_features[2]],
                    "Hour": [final_features[3]],
                    "Minute": [final_features[4]],
                    "Cloud Type": [final_features[5]],
                    "Dew Point": [final_features[6]],
                    "Solar Zenith Angle": [final_features[7]],
                    "Surface Albedo": [final_features[8]],
                    "Wind Speed": [final_features[9]],
                    "Precipitable Water": [final_features[10]],
                    "Wind Direction": [final_features[11]],
                    "Relative Humidity": [final_features[12]],
                    "Temperature": [final_features[13]],
                    "Pressure": [final_features[14]]
                    })

    prediction = int(model.predict(df))
    output = abs(round(prediction, 2))
    return render_template('index.html',prediction_text = output)



if __name__ == "__main__":
    app.run(debug=True)
