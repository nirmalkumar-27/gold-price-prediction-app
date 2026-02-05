import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Gold Price Prediction", layout="centered")

st.title("💰 Gold Price Prediction System")

uploaded_file = st.file_uploader("Upload historical gold price data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df[['Year']]
    y = df['Gold_Price']

    model = LinearRegression()
    model.fit(X, y)

    year = st.number_input("Enter year to predict", min_value=2025, max_value=2050, step=1)

    if st.button("Predict Gold Price"):
        prediction = model.predict([[year]])[0]

        st.success(f"Predicted Gold Price for {year}: ₹{prediction:.2f}")

        st.subheader("Visualization")
        plt.figure()
        plt.plot(df['Year'], df['Gold_Price'], label="Historical Prices")
        plt.scatter(year, prediction, color='red', label="Predicted Price")
        plt.xlabel("Year")
        plt.ylabel("Gold Price")
        plt.legend()
        st.pyplot(plt)
