import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_and_preprocess_data(file_path, target_label):

    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("-" * 30)
        print("First 5 rows of the data:")
        print(data.head())
        print("-" * 30)
        print("Data info:")
        data.info()
        print("-" * 30)

        numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        categorical_features = ['state', 'district', 'soil_type', 'pest_problem']

        X_numerical = data[numerical_features]
        X_categorical = pd.get_dummies(data[categorical_features])
        X = pd.concat([X_numerical, X_categorical], axis=1)

        y = data[target_label]

        all_features = X.columns
        label_encoder = None
        y_encoded = y

        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            print("Labels encoded successfully.")
            print("-" * 30)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        print("Data split into training and testing sets.")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        print("-" * 30)

        return X_train, X_test, y_train, y_test, label_encoder, all_features
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None, None



def train_and_evaluate_classifier(X_train, X_test, y_train, y_test):

    print("Starting crop recommendation model training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    print("-" * 30)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on the test set: {accuracy:.2f}")
    print("-" * 30)
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)
    return model


def train_and_evaluate_regressor(X_train, X_test, y_train, y_test):

    print("Starting yield prediction model training...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Yield prediction model training complete.")
    print("-" * 30)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")
    print(f"R-squared (R2) score on test set: {r2:.2f}")
    print("-" * 30)
    return model, X_test, y_test, y_pred



def make_full_prediction(classifier, regressor, new_data_input, label_encoder, all_features_cls, all_features_reg):

    print("Preparing new data for a full prediction...")
    new_data_df = pd.DataFrame([new_data_input])


    processed_new_data_cls = pd.get_dummies(new_data_df, columns=['state', 'district', 'soil_type', 'pest_problem'])
    final_new_data_cls = processed_new_data_cls.reindex(columns=all_features_cls, fill_value=0)

    prediction_encoded_cls = classifier.predict(final_new_data_cls)
    predicted_crop = label_encoder.inverse_transform(prediction_encoded_cls)[0]

    new_data_df['crop'] = predicted_crop

    processed_new_data_reg = pd.get_dummies(new_data_df,
                                            columns=['state', 'district', 'soil_type', 'pest_problem', 'crop'])
    final_new_data_reg = processed_new_data_reg.reindex(columns=all_features_reg, fill_value=0)

    predicted_yield = regressor.predict(final_new_data_reg)[0]

    print("-" * 30)
    print("Full Prediction for new data:")
    print(f"Input features: {new_data_input}")
    print(f"Predicted crop: {predicted_crop}")
    print(f"Predicted yield: {predicted_yield:.2f} tons per acre")
    print("-" * 30)



def plot_confusion_matrix(model, X_test, y_test, label_encoder):

    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xticks(np.arange(len(label_encoder.classes_)) + 0.5, label_encoder.classes_, rotation=45, ha='right')
    plt.yticks(np.arange(len(label_encoder.classes_)) + 0.5, label_encoder.classes_, rotation=0)
    plt.title("Confusion Matrix for Crop Recommendation", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_yield_predictions(y_test, y_pred):

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='darkgreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs. Predicted Yield", fontsize=16)
    plt.xlabel("Actual Yield (tons/acre)", fontsize=14)
    plt.ylabel("Predicted Yield (tons/acre)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_profit_histogram(file_path):

    data = pd.read_csv(file_path)
    plt.figure(figsize=(10, 8))
    sns.histplot(data['profit_or_loss_in_rupees'], bins=20, kde=True, color='purple')
    plt.title("Distribution of Profit/Loss", fontsize=16)
    plt.xlabel("Profit/Loss (in Rupees)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    file_path = "indian_agriculture_data.csv"

    X_train_cls, X_test_cls, y_train_cls, y_test_cls, label_encoder, all_features_cls = load_and_preprocess_data(
        file_path, 'crop')
    if X_train_cls is not None:
        trained_classifier = train_and_evaluate_classifier(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

        plot_confusion_matrix(trained_classifier, X_test_cls, y_test_cls, label_encoder)

        X_train_reg, X_test_reg, y_train_reg, y_test_reg, _, all_features_reg = load_and_preprocess_data(file_path,
                                                                                                         'yield_tons_per_acre')
        if X_train_reg is not None:
            trained_regressor, X_test_reg, y_test_reg, y_pred_reg = train_and_evaluate_regressor(X_train_reg,
                                                                                                 X_test_reg,
                                                                                                 y_train_reg,
                                                                                                 y_test_reg)

            plot_yield_predictions(y_test_reg, y_pred_reg)

            plot_profit_histogram(file_path)

            new_data_point = {
                'N': 95, 'P': 45, 'K': 48,
                'temperature': 27.5, 'humidity': 86.2, 'ph': 6.6, 'rainfall': 225.5,
                'state': 'West Bengal', 'district': 'Murshidabad', 'soil_type': 'Alluvial',
                'pest_problem': 'Brown planthopper'
            }

            make_full_prediction(trained_classifier, trained_regressor, new_data_point, label_encoder, all_features_cls,
                                 all_features_reg)


if __name__ == "__main__":
    main()
