import streamlit as st
import joblib
import pandas as pd

vectorizer = joblib.load("vectorizer.pkl")
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
nb_model = joblib.load("nb_model.pkl")


st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

news_input = st.text_area("News Article:", "")

model_choice = st.selectbox(
    "Select Model",
    ("Logistic Regression", "Random Forest", "Naive Bayes", "All Models")
)

if st.button("Check News"):
    if not news_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        transformed_input = vectorizer.transform([news_input])

        if model_choice == "Logistic Regression":
            pred = lr_model.predict(transformed_input)[0]
            st.success("The News is Real!" if pred == 1 else "The News is Fake!")

        elif model_choice == "Random Forest":
            pred = rf_model.predict(transformed_input)[0]
            st.success("The News is Real!" if pred == 1 else "The News is Fake!")

        elif model_choice == "Naive Bayes":
            pred = nb_model.predict(transformed_input)[0]
            st.success("The News is Real!" if pred == 1 else "The News is Fake!")

        else:  # ALL MODELS
            results = {
                "Model": ["Logistic Regression", "Random Forest", "Naive Bayes"],
                "Prediction": [
                    "Real" if lr_model.predict(transformed_input)[0] == 1 else "Fake",
                    "Real" if rf_model.predict(transformed_input)[0] == 1 else "Fake",
                    "Real" if nb_model.predict(transformed_input)[0] == 1 else "Fake",
                ]
            }

            df = pd.DataFrame(results)
            st.subheader("🔍 Model-wise Predictions")
            st.table(df)   