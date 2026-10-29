
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["Male","Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
parch = st.slider("Parents/Children", 0, 6, 0)
fare = st.slider("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S","C","Q"])

sex_value = 1 if sex == "Female" else 0
embarked_value = {"S":0,"C":1,"Q":2}[embarked]

data = pd.DataFrame(
    [[pclass, sex_value, age, sibsp, parch, fare, embarked_value]],
    columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
)

if st.button("Predict"):
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("Likely to Survive")
    else:
        st.error("Likely Not to Survive")
