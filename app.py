import streamlit as st
import pandas as pd
import pickle
import shap



with open('label_encoder.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)



def process(user_input):
    df = user_input

    # label encode
    cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in cols:
        df[col] = encoders[col].transform(df[col])

    # standardization
    df_scaled = scaler.transform(df)
    return df_scaled



def predict(user_input):
    input_processed = process(user_input)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_processed)
    first = shap_values[0,:,0].reshape(1, -1)
    second = shap_values[0,:,1].reshape(1, -1)
    result = [first, second]
    feature_names = list(user_input.columns)
    shap_val = result[1][0]

    influences = list(zip(feature_names, shap_val))
    influences_sorted = sorted(influences, key=lambda x: abs(x[1]), reverse=True)

    predicted_class = model.predict(input_processed)[0]
    label_text = "High Risk" if predicted_class == 1 else "Low Risk"

    lines = ["\n"]
    cnt = 0

    for fname, shapv in influences_sorted[:5]:
        if cnt == 3:
            break
        if (predicted_class == 1 and shapv > 0) or (predicted_class != 1 and shapv <= 0):
            cnt += 1
            direction = "increased" if shapv > 0 else "decreased"
            lines.append(f"\n-> '{fname}' {direction} the risk score by {abs(shapv):.3f}")

    explanation = f"Predicted class: {label_text}\n" + "\n".join(lines)
    return explanation



# App
st.title("Credit Risk Analysis")

age = st.text_input("Age : ")
sex = st.text_input("Sex : ")
job = st.text_input("Job : ")
housing = st.text_input("Housing : ")
saving_accounts = st.text_input("Saving Accounts : ")
checking_account = st.text_input("Checking Account : ")
credit_amount = st.text_input("Credit Amount : ")
duration = st.text_input("Duration : ")
purpose = st.text_input("Purpose : ")

user_input = pd.DataFrame([[age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose]], columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'])

if st.button("Predict"):
    res = predict(user_input)
    st.write(res)
