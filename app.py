import pandas as pd
import streamlit as st
import joblib

scaler = joblib.load('scaler.pkl')
rfModel = joblib.load('rf_model.pkl')
lrModel = joblib.load('lr_model.pkl')
svmModel = joblib.load('svm_model.pkl')

st.title("Diabetes Prediction App")

# df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
# print(df.sample(1))

bpRate       = st.checkbox("High BP")
cholRate     = st.checkbox("High Cholestrol")
cholCheck    = st.checkbox("Cholestrol Check in the last 5 years")
bmiIndex     = st.number_input("Body Mass Index (BMI)")
smoker       = st.checkbox("Smoker? (at least 100 cigarettes in your life)")
toldStroke   = st.checkbox("Ever been told you had a stroke")
heartDisease = st.checkbox("Has Coronary Heart Disease (CHD) or Myocardial Infection (MI)")
physActivity = st.checkbox("Physical Activity in the last 30 days")
eatsFruits   = st.checkbox("Consume fruit 1 or more times per day")
eatsVeggies  = st.checkbox("Consume vegetables 1 or more times per day")
heavyAlcohol = st.checkbox("Heavy drinkers (> 14 drinks a week for men and > 7 drinks for women)")
healthCare   = st.checkbox("Health care coverage (insurance, plans, etc.)")
noDoctor     = st.checkbox("Couldn't visit a doctor because of cost in the last 12 months")
genHealth    = st.number_input("On the scale of 1-5, your health is (1: excellent, 2: very good, 3: good, 4: fair, 5: poor)", 1, 5)
mentalHealth = st.number_input("How many days during the past 30 days is your mental health bad?", 0, 30)
physHealth   = st.number_input("How many days during the past 30 days is your physical health bad?", 0, 30)
diffWalk     = st.checkbox("Serious difficulty walking or climbing stairs?")
sex          = st.checkbox("Sex (check if male, unchecked if female)")
age          = st.number_input("13-level age category (1: 18-24, 13: 80+)", 1, 13)
education    = st.number_input("Education level (1: never attended school, 2: grades 1-8, 3: grades 9-11, 4: grade 12 or GED, 5: college 1-3 years, 6: college 4+)", 1, 6)
income       = st.number_input("Income scale (1: less than 10K, 8: 75K or more)", 1, 8)

if st.button("Predict"):
    inputs = [[int(bpRate), int(cholRate), int(cholCheck), bmiIndex, int(smoker), int(toldStroke), int(heartDisease),
        int(physActivity), int(eatsFruits), int(eatsVeggies), int(heavyAlcohol), int(healthCare), int(noDoctor),
        genHealth, mentalHealth, physHealth, int(diffWalk), int(sex), age, education, income]]
    inputScaled = scaler.transform(inputs)
    
    rf_pred = rfModel.predict(inputs)
    lr_pred = lrModel.predict(inputScaled)
    svm_pred = svmModel.predict(inputScaled)
    
    st.write(f"Random Forest Prediction: {rf_pred[0]}")
    st.write(f"Logistic Regression Prediction: {lr_pred[0]}")
    st.write(f"SVM Prediction: {svm_pred[0]}")