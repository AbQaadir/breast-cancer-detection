import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle as pkl


def create_model(data: pd.DataFrame):
    # Splitting the data into training and testing sets
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]

    # scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)

    print(f"Accuracy: \n{accuracy_score(y_test, y_pred)}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

    return model, scaler


def clean_data(data: pd.DataFrame):
    data = pd.read_csv(data)
    data = data.drop(["id", "Unnamed: 32"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def main():
    data = "data/data.csv"
    data = clean_data(data)
    model, scaler = create_model(data)
    print("Model created successfully!")

    # save model
    with open("model.pkl", "wb") as model_file:
        pkl.dump(model, model_file)

    # save scaler
    with open("scaler.pkl", "wb") as scaler_file:
        pkl.dump(scaler, scaler_file)
    print("Model and scaler saved successfully!")


if __name__ == "__main__":
    main()
