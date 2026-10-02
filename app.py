import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. DATASET (For Semester 4, small is okay) ---
# 0 = SAFE, 1 = FRAUD
data = {
    'message': [
        'You have received 20000 UGX from 078... Balance 150000',
        'You have received 50000 UGX. Your new balance is 100000 UGX. Ref: 12345',
        'Yello, you have received 10000 from JOHN DOE',
        'Airtel Money: You have received 30000 UGX from 070...',
        'You have won 500000 UGX! Claim now call 0701234567',
        'CONGRATULATIONS! You won 1000000 UGX. Send PIN to claim',
        'Dear customer your account blocked. Send MM PIN to 0780... to unblock',
        'Urgent: Your MoMo account will be closed. Verify with *165*123#',
        'You have been selected for loan 2000000 UGX. Pay 50000 to receive',
        'Mama send me 20000 on this number 077... it is urgent fraudsters'
    ],
    'label': [0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

# --- 2. TRAIN MODEL ---
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['message'])
model = MultinomialNB()
model.fit(X, df['label'])

# --- 3. WEBSITE UI ---
st.set_page_config(page_title="MoMo Fraud Detector UG", page_icon="🛡️")
st.title("🛡️ MoMo Fraud Detector - Uganda")
st.write("Paste any Mobile Money SMS below to check if it's FRAUD or SAFE")

sms = st.text_area("Paste SMS here:", placeholder="You have won 500,000 UGX! Call...")

if st.button("Check Message"):
    if sms.strip() == "":
        st.warning("Please paste a message")
    else:
        vec = vectorizer.transform([sms])
        pred = model.predict(vec)[0]
        prob = max(model.predict_proba(vec)[0]) * 100

        if pred == 1:
            st.error(f"⚠️ FRAUD DETECTED - {prob:.0f}% confidence")
            st.write("This message is likely a scam. Do NOT share PIN or send money.")
        else:
            st.success(f"✅ SAFE MESSAGE - {prob:.0f}% confidence")
            st.write("No threat detected. Message looks legit.")

st.caption("Semester 4 Mini Project | AI-Powered Detection | Built for MTN & Airtel Money")