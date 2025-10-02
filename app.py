from flask import Flask, request, render_template, redirect, jsonify, url_for, flash
from flask_mail import Mail, Message
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import os

# ---------------- Flask setup ----------------
app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecret"

# ---------------- Mail setup ----------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'tamara.khachatryan.2001@gmail.com'
app.config['MAIL_PASSWORD'] = 'yngy clnw ebhu bexa'  # ⚠️ Google App Password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# ---------------- Uploads ----------------
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Լիցքավորել արդեն պատրաստի մոդելը
MODEL_FILE = 'rf_loan_model.pkl'
rf_model = joblib.load('rf_loan_model.pkl')
model_features = joblib.load('rf_features.pkl')

# ----------------- Encode input function -----------------
def encode_input(data):
    """
    Encode input data for Random Forest prediction.
    - Ստանում է dictionary
    - Ստեղծում DataFrame
    - Label encode categorical fields օգտագործելով պահված le_dict
    - Ավելացնում բացակա columns 0-ով
    - Խնայվում միայն model_features
    """
    # Բեռնավորում ենք le_dict և model_features
    le_dict = joblib.load('le_dict.pkl')
    model_features = joblib.load('rf_features.pkl')

    df_input = pd.DataFrame([data])

    # Categorical encoding
    for col, le in le_dict.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # Ավելացնենք բացակա columns 0-ով
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Ջնջենք ավելորդ columns, եթե հայտնվեն
    df_input = df_input[model_features]

    return df_input



# ----------------- Predict route -----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'status':'error', 'message':'No data received'}), 400

        df_input = encode_input(data)
        pred = rf_model.predict(df_input)[0]

        # Խորհուրդ՝ օգտագործել string values, որ JS comparison-ը աշխատի
        if int(pred) == 1:
            status = 'approved'
            message = "Ձեր վարկը կհաստատվի ✅"
        else:
            status = 'rejected'
            message = "Ձեր վարկը չի հաստատվի ❌"

        return jsonify({'status': status, 'message': message})
    except Exception as e:
        return jsonify({'status':'error', 'message': str(e)}), 500

# --- Static pages ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('indexabout.html')

@app.route('/calculating')
def calculating():
    return render_template('indexcalculating.html')

@app.route('/contact')
def kap_mez_het():
    return render_template('kap_mez_het.html')

# --- Email sending ---
@app.route('/send-email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    message_body = request.form['message']

    # Send to admin
    msg = Message(
        subject=f"Նոր հաղորդագրություն {name}-ից",
        sender=app.config['MAIL_USERNAME'],
        recipients=[app.config['MAIL_USERNAME']],
        body=f"Name: {name}\nEmail: {email}\nPhone: {phone}\nMessage:\n{message_body}"
    )
    mail.send(msg)

    # Auto reply
    reply = Message(
        subject="Շնորհակալություն հաղորդագրության համար",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email],
        body=f"Բարև Ձեզ {name},\n\nՇնորհակալություն Ձեր հաղորդագրության համար։ Մենք կապ կհաստատենք Ձեզ հետ մոտակա րոպեների ընթացքում։"
    )
    mail.send(reply)

    return render_template("thanks.html")

# --- Loan workflow ---
@app.route("/varki_hayt")
def varki_hayt():
    return render_template("varki_hayt.html")


@app.route('/upload_documents', methods=['GET', 'POST'])
def upload_docs():
    if request.method == 'POST':
        for f in ['passport','soc_card','bank_statement','picture']:
            file = request.files.get(f)
            if file:
                file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        return redirect('/face_verification')
    return render_template('upload_documents.html')

@app.route('/face_verification')
def face_verification():
    return render_template('face_verification.html')

@app.route('/verify_face', methods=['POST'])
def verify_face():
    live_file = request.files.get('live')
    if not live_file:
        return jsonify({'status':'fail', 'message':'Նկար չի ստացվել'})
    live_path = os.path.join(UPLOAD_FOLDER, 'live.jpg')
    live_file.save(live_path)
    # Demo only
    return jsonify({'status':'success', 'message':'Նույնականացումը հաջողվեց'})

@app.route('/final')
def final():
    return render_template('final.html')


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
