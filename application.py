from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predictdata():
    if request.method == 'GET':
        return render_template('home.html')
    else :
        # Here we are reading all the vales
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_df()
        print(pred_df)

        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred_df)
        
        return render_template('home.html', results = round(results[0], 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0')