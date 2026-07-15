from flask import Flask, request, render_template
import pandas as pd
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

application = Flask(__name__, template_folder='src/templates')
app = application


def get_form_options():
    df = pd.read_csv('notebook/data/Salary.csv')
    return {
        'gender_options': sorted(df['Gender'].dropna().unique().tolist()),
        'education_options': sorted(df['Education Level'].dropna().unique().tolist()),
        'job_title_options': sorted(df['Job Title'].dropna().unique().tolist()),
        'experience_options': sorted(df['Years of Experience'].dropna().unique().tolist()),
        'country_options': sorted(df['Country'].dropna().unique().tolist()),
        'race_options': sorted(df['Race'].dropna().unique().tolist()),
        'senior_options': [0, 1]
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    form_options = get_form_options()

    if request.method == 'GET':
        return render_template('home.html', **form_options)

    try:
        logging.info("Prediction request received")
        data = request.form.to_dict()

        input_df = pd.DataFrame([
            {
                'Age': float(data['Age']),
                'Gender': data['Gender'],
                'Education Level': int(data['Education Level']),
                'Job Title': data['Job Title'],
                'Years of Experience': float(data['Years of Experience']),
                'Country': data['Country'],
                'Race': data['Race'],
                'Senior': int(data['Senior'])
            }
        ])

        preprocessor = load_object('artifacts/preprocessor.pkl')
        model = load_object('artifacts/model.pkl')

        transformed_data = preprocessor.transform(input_df)
        prediction = model.predict(transformed_data)[0]

        return render_template(
            'home.html',
            predicted_salary=round(float(prediction), 2),
            **form_options
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)