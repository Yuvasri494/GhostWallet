# 👻 GhostWallet  
### Real-Time Fraud Detection Using Behavioral Biometrics (Flask Prototype)

GhostWallet is a **hackathon-ready security prototype** that demonstrates how real-time fraud detection can be achieved using **behavioral biometrics** and **encrypted behavior patterns**.  
The system focuses on identifying abnormal user behavior during sensitive actions like logins or transactions.

This project is designed to be **simple, runnable, and explainable**, while still showcasing **innovation in security engineering**.

---

## 🚨 Problem Statement

Traditional authentication systems rely heavily on:
- Passwords
- OTPs
- Device-based checks

These methods **fail to detect account takeover attacks** when credentials are compromised.  
Once an attacker logs in successfully, **the system assumes the user is legitimate**.

There is a need for a **continuous, invisible security layer** that can:
- Detect abnormal behavior
- Identify fraud in real time
- Reduce financial and security risks

---

## 🎯 Project Objective

The objective of GhostWallet is to:

- Detect fraudulent behavior **after login**
- Compare live user behavior against a trusted baseline
- Flag abnormal activity in real time
- Demonstrate how behavioral data can improve security without impacting UX

---

## 🧠 Solution Overview

GhostWallet uses **behavioral biometrics**, such as:
- Typing speed
- Interaction timing
- Navigation patterns (simulated)

A baseline behavior vector is created during registration.  
During subsequent actions, live behavior is compared against this baseline using **cosine similarity**.

If similarity drops below a threshold → **Fraud is detected**.

---

## 🏗️ Architecture

Client (HTML/CSS/JS)
↓
Flask Backend (app.py)
↓
Behavior Vector Creation
↓
Similarity Comparison (NumPy)
↓
Fraud / Normal Decision

> Note: For demo simplicity, frontend (HTML, CSS, JS) is embedded directly inside `app.py`.

---

## ⚙️ Technology Stack

- **Backend:** Python, Flask
- **Security Logic:** Behavioral Biometrics
- **Math & Similarity:** NumPy
- **CORS Handling:** Flask-CORS
- **Frontend:** HTML, CSS, JavaScript (embedded)
- **Database:** In-memory (demo purpose)

## 📦 Requirements

Your `requirements.txt`:

```txt
Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3

🚀 How to Run on Any New System
1️⃣ Clone the Repository
git clone https://github.com/your-username/ghostwallet.git
cd ghostwallet

2️⃣ Create Virtual Environment (Recommended)

python -m venv venv
Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000

🔍 How GhostWallet Works (Flow)

User Registration

Behavioral data is collected

Baseline vector is created

User Action / Login

New behavior data is captured

Converted into a behavior vector

Comparison

Cosine similarity is calculated

Compared against stored baseline

Decision

✅ Normal Behavior → Allowed

🚨 Abnormal Behavior → Fraud Alert

🧪 Reliability & Completeness

Fully runnable on any system with Python

No external services required

Deterministic behavior for demo

Clear success/failure outputs

💡 Innovation & Creativity

Uses behavioral biometrics, not credentials

Detects fraud after login

Demonstrates encrypted behavior embeddings (conceptual)

Lightweight but extensible security model

🔐 Security Impact (ROI)

Reduces account takeover risk

Detects fraud even when credentials are valid

Prevents high-cost financial fraud

Improves security without hurting user experience

📈 Scope & Future Enhancements

Integration with CyborgDB for encrypted vector storage

Real-time transaction monitoring

ML-based adaptive thresholds

Browser-based behavioral tracking

Cloud deployment (AWS / Azure)

⚠️ Limitations (Demo Scope)

In-memory storage (not persistent)

Simulated behavioral signals

Not production-hardened

No real encryption (conceptual for demo)

🏁 Conclusion

GhostWallet demonstrates how continuous authentication using behavioral biometrics can dramatically improve security systems.
It highlights a practical, innovative approach to fraud detection that goes beyond passwords and OTPs.
