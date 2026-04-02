from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st

st.title("Prediktera bilpriser")

df = pd.read_csv("car_price_dataset.csv", sep=";")

models = df["Model"].unique()

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

columns = X.columns

year = st.slider("År", 2000, 2025, 2015)
engine = st.number_input("Motorstorlek", 1.0, 5.0, 2.0)
mileage = st.number_input("Mileage", 0, 500000, 100000)
doors = st.selectbox("Antal dörrar", [2, 3, 4, 5])
owners = st.selectbox("Antal ägare", [1, 2, 3, 4, 5])

if st.button("Prediktera pris"):
    input_data = pd.DataFrame({"Year": [year], "Engine_Size": [engine], "Mileage": [mileage], "Doors": [doors], "Owner_Count": [owners]})
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    prediction = model.predict(input_data)

    st.success(f"Predikterat pris: {int(prediction[0])} kr")
