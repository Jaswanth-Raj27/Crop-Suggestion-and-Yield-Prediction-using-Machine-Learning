from flask import Flask, render_template, request
import pandas as pd
from Yield import (
    load_and_preprocess_data,
    train_and_evaluate_classifier,
    train_and_evaluate_regressor
)

app = Flask(__name__)

# Load models once
file_path = "indian_agriculture_data.csv"

X_train_cls, X_test_cls, y_train_cls, y_test_cls, label_encoder, all_features_cls = load_and_preprocess_data(
    file_path, 'crop')
classifier = train_and_evaluate_classifier(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

X_train_reg, X_test_reg, y_train_reg, y_test_reg, _, all_features_reg = load_and_preprocess_data(
    file_path, 'yield_tons_per_acre')
regressor, _, _, _ = train_and_evaluate_regressor(X_train_reg, X_test_reg, y_train_reg, y_test_reg)


@app.route('/')
def home():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form['state']
        district = request.form['district']
        soil_type = request.form['soil_type']
        pest_problem = request.form['pest_problem']

        new_data_point = {
            'N': N, 'P': P, 'K': K,
            'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
            'state': state, 'district': district, 'soil_type': soil_type, 'pest_problem': pest_problem
        }

        new_data_df = pd.DataFrame([new_data_point])
        processed_new_data_cls = pd.get_dummies(new_data_df, columns=['state', 'district', 'soil_type', 'pest_problem'])
        final_new_data_cls = processed_new_data_cls.reindex(columns=all_features_cls, fill_value=0)

        # Predict crop
        prediction_encoded_cls = classifier.predict(final_new_data_cls)
        predicted_crop = label_encoder.inverse_transform(prediction_encoded_cls)[0]

        # Predict yield
        new_data_df['crop'] = predicted_crop
        processed_new_data_reg = pd.get_dummies(new_data_df,
                                                columns=['state', 'district', 'soil_type', 'pest_problem', 'crop'])
        final_new_data_reg = processed_new_data_reg.reindex(columns=all_features_reg, fill_value=0)

        predicted_yield = regressor.predict(final_new_data_reg)[0]

        result = {
            "crop": predicted_crop,
            "yield": f"{predicted_yield:.2f} tons per acre"
        }

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
