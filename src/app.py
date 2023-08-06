import numpy as np
import pandas as pd
from src.logger import logging
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting new data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("Rendering html form: (GET)")
        return render_template('home.html')
    else:
        logging.info("Getting data from html form: (POST)")
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.convert_raw_data_to_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        res = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = res[0])        
        
if __name__ == "__main__":
    app.run(host='0.0.0.0')