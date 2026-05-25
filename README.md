# Car Price Prediction using Machine Learning

This project predicts the selling price of used cars using Machine Learning techniques.  
The model is trained on car-related features such as present price, fuel type, transmission type, kilometers driven, ownership history, and car age.

The project also includes an interactive Streamlit web application where users can enter car details and get real-time price predictions.

## Features

- Data preprocessing and feature engineering
- Machine Learning regression model training
- Random Forest Regressor implementation
- Model evaluation using R² Score and MAE
- Interactive Streamlit web app
- Data visualization and analytics

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Dataset Features

- Car_Name
- Year
- Selling_Price
- Present_Price
- Driven_kms
- Fuel_Type
- Selling_type
- Transmission
- Owner

##  How to Run

## Install dependencies
pip install -r requirements.txt

## Train the model
python train_model.py

## Run Streamlit App
streamlit run app.py
