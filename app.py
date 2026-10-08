import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "url": [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "https://www.github.com",
        "https://secure-login-verify-account.com",
        "http://paypal-login-security-check.com",
        "http://verify-your-account-now.com",
        "http://free-gift-card-claim.com",
        "http://bank-account-verification-login.com",
        "http://login-confirm-security-alert.com",
        "http://update-payment-information-now.com"
    ],
    "label": [0,0,0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2,5)
)

X = vectorizer.fit_transform(df["url"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

st.set_page_config(
    page_title="AI Phishing Link Detector",
    page_icon="🛡️"
)

st.title("🛡️ AI-Based Phishing Link Detector")

st.write(
    "Enter a website URL below to check whether it is "
    "likely safe or potentially phishing."
)

url = st.text_input(
    "🔗 Enter Website URL",
    placeholder="https://example.com"
)

if st.button("🔍 Check Link"):

    if url.strip() == "":
        st.warning("Please enter a URL.")

    else:
        features = vectorizer.transform([url])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("⚠️ LIKELY PHISHING LINK")
            st.write(
                "This URL contains patterns commonly associated "
                "with phishing websites."
            )

        else:
            st.success("✅ LIKELY SAFE LINK")
            st.write(
                "No strong phishing pattern was detected "
                "by the current model."
            )

st.markdown("---")

st.caption(
    "M.Sc. Mathematics Research Project | "
    "AI & Mathematical Modelling"
)
