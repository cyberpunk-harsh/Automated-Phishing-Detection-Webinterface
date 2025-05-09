from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)

# ✅ Load the trained model & vectorizer correctly
with open("model/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Updated feature extraction with explanations
def extract_features_and_explanations(url):
    explanations = []

    # Length of URL
    url_length = len(url)
    if url_length > 75:
        explanations.append("🔸 URL is very long, which is often used to hide malicious content.")
    else:
        explanations.append("✅ URL length is normal.")

    # Count of special characters
    special_chars = len(re.findall(r"[!@#$%^&*(),.?\":{}|<>]", url))
    if special_chars > 5:
        explanations.append("🔸 Contains many special characters, which may be used to trick users.")
    else:
        explanations.append("✅ Limited special characters, which is typical for legitimate sites.")

    # Count of digits
    digits = sum(c.isdigit() for c in url)
    if digits > 10:
        explanations.append("🔸 Excessive use of digits can indicate a suspicious or autogenerated URL.")
    else:
        explanations.append("✅ Number of digits looks normal.")

    # Count of hyphens (-)
    hyphens = url.count('-')
    if hyphens > 3:
        explanations.append("🔸 Too many hyphens, which are often used in phishing URLs.")
    else:
        explanations.append("✅ Hyphen usage is acceptable.")

    # Count of dots (.)
    dots = url.count('.')
    if dots > 3:
        explanations.append("🔸 Too many dots indicating multiple subdomains, which is suspicious.")
    else:
        explanations.append("✅ Dot count is within normal range.")

    # Presence of "https"
    https = 1 if "https" in url.lower() else 0
    if https == 1:
        explanations.append("✅ Uses HTTPS, which is more secure.")
    else:
        explanations.append("🔸 Does not use HTTPS, which can be a security risk.")

    # Number of subdomains
    parsed_url = urlparse(url)
    subdomains = len(parsed_url.netloc.split('.')) - 2
    if subdomains > 2:
        explanations.append("🔸 Contains many subdomains, a tactic often used in phishing.")
    else:
        explanations.append("✅ Subdomain count is typical.")

    # Vectorizer transformation
    url_features = vectorizer.transform([url]).toarray()[0]

    return url_features, explanations

@app.route("/")
def home():
    return render_template("index.html", prediction_text="", alert="no", url_entered="", feature_explanations=[], phishing_losses=[], safety_tips=[])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form["url"].strip()
        url_features, explanations = extract_features_and_explanations(url)
        url_features = np.array(url_features).reshape(1, -1)

        prediction = model.predict(url_features)[0]

        phishing_losses = [
            "Identity Theft: Your personal data can be stolen.",
            "Financial Loss: Bank details can be misused.",
            "Credential Theft: Username & password can be compromised.",
            "Device Infection: Malware or ransomware can be installed.",
            "Privacy Violation: Your private emails and messages can be accessed.",
            "Data Breach: Your sensitive data can be sold on the dark web.",
            "Fake Transactions: Fraudulent purchases in your name.",
            "Social Engineering Attacks: You may be tricked into further scams.",
            "Legal Trouble: Misuse of your stolen identity.",
            "Reputation Damage: Your social media can be hijacked."
        ]

        safety_tips = [
            "Always check the URL before clicking.",
            "Use multi-factor authentication (MFA).",
            "Keep your browser & antivirus updated.",
            "Do not enter sensitive info on suspicious websites.",
            "Look for HTTPS in the URL.",
            "Verify sender emails before clicking links.",
            "Do not download attachments from unknown emails.",
            "Use password managers for secure logins.",
            "Avoid entering details on pop-up windows.",
            "Report phishing attempts to authorities."
        ]

        if prediction == 1:
            result = "🚨 Phishing Link Detected! Do not proceed further."
            alert_flag = "yes"
            return render_template(
                "index.html",
                prediction_text=result,
                alert=alert_flag,
                url_entered=url,
                feature_explanations=explanations,
                phishing_losses=phishing_losses,
                safety_tips=[]
            )
        else:
            result = "✅ Legitimate Link"
            alert_flag = "no"
            return render_template(
                "index.html",
                prediction_text=result,
                alert=alert_flag,
                url_entered=url,
                feature_explanations=explanations,
                phishing_losses=[],
                safety_tips=safety_tips
            )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
