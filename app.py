from flask import Flask, render_template, request, redirect, session, flash, jsonify, send_from_directory
import sqlite3
import joblib
import pandas as pd
import json
import csv
import numpy as np
from sklearn.impute import SimpleImputer
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import re

import cv2
import keras
from keras.applications.efficientnet import preprocess_input, EfficientNetB0
from keras.preprocessing import image as keras_image

shap = None
SHAP_AVAILABLE = False
SHAP_IMPORT_ERROR = None
SHAP_LOAD_ATTEMPTED = False

# ===============================
# App Config
# ===============================
app = Flask(__name__)
app.secret_key = "secret123"

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'uploads/vehicle_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================
# Load trained model
# ===============================
model = joblib.load("fraud_xgboost_model.pkl")

def _iter_estimators(estimator):
    yield estimator
    steps = getattr(estimator, "steps", None)
    if steps:
        for _, step in steps:
            if step is not None:
                yield from _iter_estimators(step)

    transformers = getattr(estimator, "transformers", None)
    if transformers:
        for _, transformer, _ in transformers:
            if hasattr(transformer, "transform"):
                yield from _iter_estimators(transformer)

    transformers_fitted = getattr(estimator, "transformers_", None)
    if transformers_fitted:
        for _, transformer, _ in transformers_fitted:
            if hasattr(transformer, "transform"):
                yield from _iter_estimators(transformer)

def _ensure_simpleimputer_fill_dtype(estimator):
    # Backward-compat: models trained with older sklearn may miss _fill_dtype
    for est in _iter_estimators(estimator):
        if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
            if hasattr(est, "_fit_dtype"):
                est._fill_dtype = est._fit_dtype
            elif hasattr(est, "statistics_") and est.statistics_ is not None:
                est._fill_dtype = est.statistics_.dtype
            else:
                est._fill_dtype = np.float64

_ensure_simpleimputer_fill_dtype(model)

# ===============================
# Load Car Damage Models
# ===============================
print("Loading car damage models...")

FINETUNED_MODEL_PATH = "./models/damage_classifier_best.h5"
LABEL_ENCODER_PATH = "./models/label_encoder.pkl"
XGB_FINETUNED_PATH = "./models/repair_cost_xgb_finetuned.pkl"
XGB_OLD_PATH = "./models/repair_cost_xgb.pkl"
SEVERITY_MODEL_PATH = "./models/severity_classifier.h5"

USE_FINETUNED_DAMAGE = os.path.exists(FINETUNED_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH)

if USE_FINETUNED_DAMAGE:
    print("✓ Using fine-tuned car damage model")
    cnn_model = keras.models.load_model(FINETUNED_MODEL_PATH, compile=False)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if os.path.exists(SEVERITY_MODEL_PATH):
        try:
            severity_model = keras.models.load_model(SEVERITY_MODEL_PATH, compile=False)
        except:
            severity_model = None
    else:
        severity_model = None
    
    if os.path.exists(XGB_FINETUNED_PATH):
        xgb_cost_model = joblib.load(XGB_FINETUNED_PATH)
    else:
        xgb_cost_model = joblib.load(XGB_OLD_PATH)
else:
    print("⚠ Fine-tuned model not found. Using basic model.")
    cnn_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    feature_model = cnn_model
    label_encoder = None
    severity_model = None
    try:
        knn = joblib.load("./models/damage_knn.pkl")
    except:
        knn = None
    xgb_cost_model = joblib.load(XGB_OLD_PATH)


explainer = None

def _load_shap():
    global shap, SHAP_AVAILABLE, SHAP_IMPORT_ERROR, SHAP_LOAD_ATTEMPTED
    if SHAP_LOAD_ATTEMPTED:
        return SHAP_AVAILABLE
    SHAP_LOAD_ATTEMPTED = True
    try:
        import shap as _shap
        shap = _shap
        SHAP_AVAILABLE = True
        SHAP_IMPORT_ERROR = None
    except Exception as exc:
        shap = None
        SHAP_AVAILABLE = False
        SHAP_IMPORT_ERROR = str(exc)
    return SHAP_AVAILABLE

def get_explainer():
    global explainer
    if explainer is not None:
        return explainer
    if not _load_shap():
        return None
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    return explainer

# ===============================
# Fraud decision threshold
# ===============================
FRAUD_THRESHOLD = 0.20   # IMPORTANT: lower threshold

# ===============================
# Full training schema
# ===============================
MODEL_COLUMNS = [
    'months_as_customer','age','policy_state','policy_csl','policy_deductable',
    'policy_annual_premium','umbrella_limit','insured_sex',
    'insured_education_level','insured_occupation','insured_relationship',
    'capital-gains','capital-loss','incident_type','collision_type',
    'incident_severity','authorities_contacted','incident_state','incident_city',
    'incident_hour_of_the_day','number_of_vehicles_involved','property_damage',
    'bodily_injuries','witnesses','police_report_available',
    'total_claim_amount','injury_claim','property_claim','vehicle_claim',
    'auto_make','auto_model','auto_year','insured_hobbies','_c39'
]

# ===============================
# Validation helpers
# ===============================
def validate_username(username):
    """
    Validate username:
    - Length: 3-30 characters
    - Allowed: alphanumeric, underscore, hyphen
    - Must start with letter or number
    """
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(username) > 30:
        return False, "Username must be at most 30 characters long"
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', username):
        return False, "Username must start with a letter or number and contain only letters, numbers, underscores, or hyphens"
    return True, ""

def validate_password(password):
    """
    Validate password strength:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 128:
        return False, "Password must be at most 128 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    return True, ""

def check_username_exists(username):
    """Check if username already exists in database"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=?", (username,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

# ===============================
# Image Analysis Helpers (Car Damage)
# ===============================
def predict_damage_type(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    if USE_FINETUNED_DAMAGE:
        predictions = cnn_model(x, training=False).numpy()
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        damage_type = label_encoder.inverse_transform([predicted_class_idx])[0]
        return damage_type, confidence
    else:
        features = feature_model(x, training=False).numpy()[0]
        damage_type = knn.predict(features.reshape(1, -1))[0]
        return damage_type, 0.0

def extract_features_for_cost(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    if USE_FINETUNED_DAMAGE:
        features = x
        for i in range(5):
            features = cnn_model.layers[i](features, training=False)
        return features.numpy()[0]
    else:
        return feature_model(x, training=False).numpy()[0]

def get_severity_score_from_probs(probs):
    weights = np.array([1.0, 3.0, 5.0]) 
    return np.sum(probs * weights)

def predict_severity(img_path, damage_type=None, confidence=0.0):
    if USE_FINETUNED_DAMAGE and severity_model:
        try:
            img = keras_image.load_img(img_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            probs = severity_model(x, training=False).numpy()[0]
            score = get_severity_score_from_probs(probs)
            return float(score)
        except:
            pass
            
    base_severity = {
        "scratch": 1.2, "crack": 2.5, "dent": 2.8,
        "glass_shatter": 4.2, "lamp_broken": 3.5, "tire_flat": 3.0
    }
    severity = base_severity.get(damage_type, 2.0)
    if confidence > 0:
        severity = severity * (0.7 + 0.3 * confidence)
    return float(np.clip(severity, 0.5, 5.0))

def get_severity_level(score):
    if score < 1.5: return "Minor"
    elif score < 3.0: return "Moderate"
    elif score < 4.0: return "Major"
    else: return "Severe"

def get_damage_icon(damage_type):
    icons = {
        "scratch": "🔧", "crack": "⚡", "dent": "🔨",
        "glass_shatter": "💥", "lamp_broken": "💡", "tire_flat": "🛞"
    }
    return icons.get(damage_type, "🔍")

def get_recommendations(damage_type, severity_level, repair_cost):
    recommendations = []
    if severity_level in ["Major", "Severe"]:
        recommendations.append("⚠️ Immediate professional inspection recommended")
        recommendations.append("🚗 Vehicle may not be safe to drive")
    elif severity_level == "Moderate":
        recommendations.append("📅 Schedule repair within 1-2 weeks")
        recommendations.append("✅ Vehicle is likely safe to drive carefully")
    else:
        recommendations.append("✅ Minor damage - repair at your convenience")
    if repair_cost > 3000:
        recommendations.append("💰 Consider getting multiple repair quotes")
    if damage_type == "glass_shatter":
        recommendations.append("🪟 Glass damage can spread - repair soon")
    elif damage_type == "tire_flat":
        recommendations.append("🛞 Replace tire before driving long distances")
    return recommendations

# ===============================
# Database helpers
# ===============================
def get_db():
    return sqlite3.connect("database.db")

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT,
        role TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        claim_data TEXT,
        status TEXT,
        fraud_prediction TEXT,
        fraud_probability REAL,
        shap_reason TEXT,
        decision TEXT DEFAULT 'Pending',
        repair_cost_total REAL
    )
    """)

    # Migration: add new columns to existing databases
    for col_def in [
        "ALTER TABLE claims ADD COLUMN repair_cost_total REAL",
        "ALTER TABLE claims ADD COLUMN payout_sent INTEGER DEFAULT 0",
    ]:
        try:
            cur.execute(col_def)
        except Exception:
            pass  # Column already exists

    cur.execute("""
    CREATE TABLE IF NOT EXISTS bank_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        account_holder TEXT,
        bank_name TEXT,
        account_number TEXT,
        ifsc_code TEXT,
        account_type TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        filename TEXT,
        original_filename TEXT,
        damage_type TEXT,
        severity_score REAL,
        repair_cost REAL,
        analysis_json TEXT,
        annotated_filename TEXT,
        FOREIGN KEY(claim_id) REFERENCES claims(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        message TEXT NOT NULL,
        claim_id INTEGER,
        is_read INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ===============================
# Load dropdowns from dataset
# ===============================
def load_dropdowns():
    fields = [
        "policy_state","insured_education_level","insured_occupation",
        "incident_type","collision_type","incident_severity",
        "incident_state","incident_city","authorities_contacted",
        "insured_hobbies","auto_make","auto_model"
    ]
    data = {f: set() for f in fields}

    with open("data/insurance_claims.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for f in fields:
                if row[f]:
                    data[f].add(row[f])

    return {k: sorted(v) for k, v in data.items()}

# ===============================
# Login
# ===============================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        # Basic input validation
        if not username or not password:
            return render_template("login.html", error="Please enter both username and password")
        
        # Query database for user
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT password, role FROM users WHERE username=?",
            (username,)
        )
        user = cur.fetchone()
        conn.close()
        
        # Check if user exists and password matches
        if user:
            stored_password = user[0]
            role = user[1]
            
            # Check if password is hashed
            if stored_password.startswith('pbkdf2:') or stored_password.startswith('scrypt:'):
                # Use secure password verification
                if check_password_hash(stored_password, password):
                    session["user"] = username
                    session["role"] = role
                    return redirect("/admin" if role == "admin" else "/user")
            else:
                # Legacy plain text password (for backward compatibility)
                # This should be migrated to hashed passwords
                if stored_password == password:
                    session["user"] = username
                    session["role"] = role
                    
                    # Update to hashed password
                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE users SET password=? WHERE username=?",
                        (generate_password_hash(password), username)
                    )
                    conn.commit()
                    conn.close()
                    
                    return redirect("/admin" if role == "admin" else "/user")
        
        # Invalid credentials
        return render_template("login.html", error="Invalid username or password")
    
    return render_template("login.html")

# ===============================
# Register
# ===============================
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "user")
        
        # Validate username
        is_valid, error_msg = validate_username(username)
        if not is_valid:
            return render_template("register.html", error=error_msg)
        
        # Check if username already exists
        if check_username_exists(username):
            return render_template("register.html", error="Username already exists. Please choose a different username.")
        
        # Validate password
        is_valid, error_msg = validate_password(password)
        if not is_valid:
            return render_template("register.html", error=error_msg)
        
        # Check password confirmation
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match")
        
        # Validate role
        if role not in ["user", "admin"]:
            role = "user"  # Default to user if invalid role
        
        # Hash password and insert user
        hashed_password = generate_password_hash(password)
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role)
            )
            conn.commit()
            conn.close()
            return render_template("login.html", success="Registration successful! Please login.")
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Username already exists")
    
    return render_template("register.html")

# ===============================
# USER: Submit claim
# ===============================
@app.route("/user", methods=["GET","POST"])
def user_dashboard():
    dropdowns = load_dropdowns()

    if request.method == "POST":
        # Extract form data excluding files into JSON
        form_data = {k: v for k, v in request.form.items()}
        claim_json = json.dumps(form_data)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO claims (user, claim_data, status)
            VALUES (?, ?, ?)
        """, (session["user"], claim_json, "Pending"))
        
        claim_id = cur.lastrowid
        
        # Handle file uploads
        files = request.files.getlist('vehicle_images')
        for file in files:
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                # Create a unique filename using claim_id to avoid overwrites
                import uuid
                filename = f"claim_{claim_id}_{uuid.uuid4().hex[:8]}_{original_filename}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                cur.execute("""
                    INSERT INTO vehicle_images (claim_id, filename, original_filename)
                    VALUES (?, ?, ?)
                """, (claim_id, filename, original_filename))

        conn.commit()
        conn.close()

        return redirect("/my_claims")

    return render_template("user_dashboard.html", dropdowns=dropdowns)

# ===============================
# USER: View my claims
# ===============================
@app.route("/my_claims")
def my_claims():
    if "user" not in session:
        return redirect("/")
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.user, c.status, c.fraud_prediction, c.fraud_probability, c.decision,
               COALESCE(
                   NULLIF(c.repair_cost_total, 0),
                   (SELECT SUM(vi.repair_cost) FROM vehicle_images vi WHERE vi.claim_id = c.id AND vi.repair_cost IS NOT NULL)
               ) as effective_repair_cost,
               c.payout_sent
        FROM claims c
        WHERE c.user = ?
        ORDER BY c.id DESC
    """, (session["user"],))
    claims = cur.fetchall()

    # Fetch unread notifications for the user
    cur.execute("""
        SELECT id, message, claim_id, created_at
        FROM notifications
        WHERE username=? AND is_read=0
        ORDER BY id DESC
    """, (session["user"],))
    notifications = cur.fetchall()

    # Mark fetched notifications as read
    if notifications:
        ids = [n[0] for n in notifications]
        cur.execute(
            "UPDATE notifications SET is_read=1 WHERE id IN ({})".format(",".join("?" * len(ids))),
            ids
        )
        conn.commit()

    conn.close()
    
    converted_claims = []
    for claim in claims:
        claim_list = list(claim)
        fraud_prob = claim_list[4]
        
        if fraud_prob is not None:
            if isinstance(fraud_prob, bytes):
                try:
                    import struct
                    fraud_prob = struct.unpack('<f', fraud_prob)[0] * 100
                    fraud_prob = round(fraud_prob, 2)
                except:
                    fraud_prob = None
            elif isinstance(fraud_prob, (int, float)):
                fraud_prob = float(fraud_prob)
            else:
                fraud_prob = None
        
        claim_list[4] = fraud_prob
        converted_claims.append(tuple(claim_list))
    
    return render_template("my_claims.html", claims=converted_claims, notifications=notifications)

# ===============================
# Service Centers (Kerala Map)
# ===============================
@app.route("/service_centers")
def service_centers():
    if "user" not in session:
        return redirect("/")
    return render_template("service_centers.html")

# ===============================
# ADMIN dashboard
# ===============================
@app.route("/admin")
def admin_dashboard():
    return render_template("admin_dashboard.html")

# ===============================
# ADMIN: Claims queue
# ===============================
@app.route("/claims")
def claims():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, user, status, fraud_prediction, fraud_probability, decision
        FROM claims
    """)
    claims = cur.fetchall()
    conn.close()
    return render_template("admin_claims.html", claims=claims)

# ===============================
# ADMIN: View claim details
# ===============================
@app.route("/claim_details/<int:cid>")
def claim_details(cid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, user, claim_data, status, fraud_prediction, fraud_probability, shap_reason
        FROM claims
        WHERE id = ?
    """, (cid,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return "Claim not found", 404
    
    claim_id, user, claim_data_json, status, fraud_pred, fraud_prob, shap_reason = row
    data = json.loads(claim_data_json)
    
    # Add default values for fields that might be missing
    default_fields = {
        'capital-gains': 0,
        'capital-loss': 0,
        'bodily_injuries': 0,
        'witnesses': 0,
        'injury_claim': 0,
        'property_claim': 0,
        'vehicle_claim': 0,
        'insured_relationship': 'self'
    }
    
    for field, default_value in default_fields.items():
        if field not in data:
            data[field] = default_value
    
    # Convert numeric string fields to proper types for template formatting
    numeric_fields = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
        'auto_year'
    ]
    
    for field in numeric_fields:
        if field in data and data[field] is not None:
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                data[field] = 0
    
    # Handle fraud_probability data type conversion
    # It might be stored as bytes (old data) or float (new data)
    if fraud_prob is not None:
        if isinstance(fraud_prob, bytes):
            try:
                import struct
                # Try to unpack as float
                fraud_prob = struct.unpack('<f', fraud_prob)[0] * 100
                fraud_prob = round(fraud_prob, 2)
            except:
                fraud_prob = None
        elif isinstance(fraud_prob, (int, float)):
            fraud_prob = float(fraud_prob)
        else:
            fraud_prob = None
            
    # Fetch images
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT filename, original_filename FROM vehicle_images WHERE claim_id=?", (cid,))
    images = cur.fetchall()
    conn.close()
    
    return render_template("claim_details.html",
        claim_id=claim_id,
        claim_data={
            "user": user,
            "status": status,
            "fraud_prediction": fraud_pred,
            "fraud_probability": fraud_prob,
            "shap_reason": shap_reason
        },
        data=data,
        images=images
    )

# ===============================
# ADMIN: View claims
# ===============================
@app.route("/view_claims")
def view_claims():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, user, status, fraud_prediction, fraud_probability
        FROM claims
    """)
    claims = cur.fetchall()
    conn.close()
    return render_template("view_claims.html", claims=claims)

# ===============================
# ADMIN: Predict fraud + SHAP
# ===============================
@app.route("/predict/<int:cid>")
def predict(cid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT claim_data FROM claims WHERE id=?", (cid,))
    row = cur.fetchone()

    claim = json.loads(row[0])

    # --------------------------------
    # Build FULL input with smart defaults
    # --------------------------------
    # Derive sub-claim amounts from total if not supplied
    total_claim = float(claim.get("total_claim_amount", [3000])[0] if isinstance(claim.get("total_claim_amount"), list) else claim.get("total_claim_amount", 3000))
    default_sub_claim = round(total_claim / 3, 2) if total_claim > 0 else 1000

    SMART_DEFAULTS = {
        "injury_claim":    default_sub_claim,
        "property_claim":  default_sub_claim,
        "vehicle_claim":   default_sub_claim,
        "bodily_injuries": 1,
        "witnesses":       1,
        "capital-gains":   0,
        "capital-loss":    0,
        "insured_relationship": "self",
        "policy_csl":      "250/500",
        "_c39":            0,
    }

    full_input = {}
    for c in MODEL_COLUMNS:
        val = claim.get(c)
        if val is not None:
            # form values come as lists when parsed from JSON-encoded form
            full_input[c] = val[0] if isinstance(val, list) else val
        elif c in SMART_DEFAULTS:
            full_input[c] = SMART_DEFAULTS[c]
        else:
            full_input[c] = 0

    # Convert all numeric columns from strings to numbers
    NUMERIC_COLUMNS = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year', '_c39'
    ]
    for col in NUMERIC_COLUMNS:
        if col in full_input:
            try:
                full_input[col] = float(full_input[col])
            except (ValueError, TypeError):
                full_input[col] = 0.0

    df_input = pd.DataFrame([full_input])

    # --------------------------------
    # Predict (ML model)
    # --------------------------------
    prob = model.predict_proba(df_input)[0][1]

    # --------------------------------
    # Rule-based fraud boosters
    # These catch financial inconsistencies the ML model may miss
    # --------------------------------
    rule_flags = []
    rule_boost = 0.0

    total_claim  = full_input.get("total_claim_amount", 0) or 0
    annual_prem  = full_input.get("policy_annual_premium", 0) or 0
    injury_c     = full_input.get("injury_claim", 0) or 0
    property_c   = full_input.get("property_claim", 0) or 0
    vehicle_c    = full_input.get("vehicle_claim", 0) or 0
    sub_total    = injury_c + property_c + vehicle_c

    # Rule 1: Claim is very disproportionate to annual premium
    if annual_prem > 0 and total_claim > 0:
        ratio = total_claim / annual_prem
        if ratio > 20:           # claim > 20× annual premium — very suspicious
            rule_boost += 0.45
            rule_flags.append(f"Claim amount (${total_claim:,.0f}) is {ratio:.0f}× the annual premium (${annual_prem:,.0f})")
        elif ratio > 10:         # claim > 10× annual premium — suspicious
            rule_boost += 0.25
            rule_flags.append(f"Claim amount (${total_claim:,.0f}) is {ratio:.0f}× the annual premium (${annual_prem:,.0f})")

    # Rule 2: Sub-claims (injury + property + vehicle) don't add up to total claim
    if total_claim > 0 and sub_total > 0:
        mismatch_pct = abs(sub_total - total_claim) / total_claim
        if mismatch_pct > 0.30:   # >30% discrepancy
            rule_boost += 0.35
            rule_flags.append(f"Sub-claims sum (${sub_total:,.0f}) does not match total claim (${total_claim:,.0f}) — {mismatch_pct*100:.0f}% discrepancy")
        elif mismatch_pct > 0.15:  # >15% discrepancy
            rule_boost += 0.15
            rule_flags.append(f"Sub-claims sum (${sub_total:,.0f}) has a {mismatch_pct*100:.0f}% discrepancy vs total claim (${total_claim:,.0f})")

    # Rule 3: One sub-claim is disproportionately large vs others (e.g., all in vehicle, nothing in injury for an injury claim)
    if sub_total > 0:
        sub_claims = [("injury", injury_c), ("property", property_c), ("vehicle", vehicle_c)]
        max_share = max(v for _, v in sub_claims) / sub_total if sub_total > 0 else 0
        if max_share > 0.95 and total_claim > 10000:  # 95%+ in a single sub-claim
            dominant = [n for n, v in sub_claims if v / sub_total > 0.95][0]
            rule_boost += 0.20
            rule_flags.append(f"95%+ of claim is concentrated in {dominant} sub-claim only")

    # Combine ML probability with rule boost (cap at 0.99)
    if rule_boost > 0:
        # Blend: ensure rules can push genuine claims over the threshold
        prob = min(0.99, prob + rule_boost * (1 - prob))

    pred = "Fraud" if prob >= FRAUD_THRESHOLD else "Genuine"
    print(f"ML prob: {float(prob):.4f} | Rule boost: {rule_boost:.2f} | Final: {prob*100:.2f}% | Verdict: {pred}")
    if rule_flags:
        print("Rule flags:", rule_flags)

    # --------------------------------
    # SHAP explanation (optional)
    # --------------------------------
    top_reasons = []
    explainer = get_explainer()
    if explainer is not None:
        try:
            X_transformed = model.named_steps["preprocessor"].transform(df_input)
            shap_values = explainer.shap_values(X_transformed)

            pre = model.named_steps["preprocessor"]
            num_features = pre.transformers_[0][2]
            cat_encoder = pre.transformers_[1][1].named_steps["onehot"]
            cat_features = cat_encoder.get_feature_names_out(
                pre.transformers_[1][2]
            )
            feature_names = np.concatenate([num_features, cat_features])

            shap_vals = shap_values[0]
            min_len = min(len(feature_names), len(shap_vals))

            shap_df = pd.DataFrame({
                "feature": feature_names[:min_len],
                "impact": shap_vals[:min_len]
            })

            top_reasons = (
                shap_df
                .assign(abs_impact=lambda x: abs(x.impact))
                .sort_values("abs_impact", ascending=False)
                .head(5)[["feature","impact"]]
                .to_dict(orient="records")
            )
        except Exception:
            top_reasons = []

    # Prepend business rule flags so they appear first in the explanation
    if rule_flags:
        for flag in reversed(rule_flags):
            top_reasons.insert(0, {"feature": "⚠ Business Rule", "impact": flag})

    # --------------------------------
    # Save result
    # --------------------------------
    cur.execute("""
        UPDATE claims
        SET status=?, fraud_prediction=?, fraud_probability=?, shap_reason=?
        WHERE id=?
    """, (
        "Reviewed",
        pred,
        float(round(prob * 100, 2)),
        json.dumps(top_reasons),
        cid
    ))

    conn.commit()
    conn.close()
    return redirect("/claims")

# ===============================
# ADMIN: Approve claim
# ===============================
@app.route("/approve/<int:cid>")
def approve_claim(cid):
    conn = get_db()
    cur = conn.cursor()
    # Sum all analyzed image repair costs for this claim
    cur.execute("""
        SELECT SUM(repair_cost) FROM vehicle_images
        WHERE claim_id=? AND repair_cost IS NOT NULL
    """, (cid,))
    row = cur.fetchone()
    total_repair_cost = float(row[0]) if row and row[0] is not None else 0.0
    cur.execute(
        "UPDATE claims SET decision=?, repair_cost_total=? WHERE id=?",
        ("Approved", total_repair_cost, cid)
    )
    conn.commit()
    conn.close()
    return redirect("/claims")

# ===============================
# ADMIN: Reject claim
# ===============================
@app.route("/reject/<int:cid>")
def reject_claim(cid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE claims SET decision=? WHERE id=?", ("Rejected", cid))
    conn.commit()
    conn.close()
    return redirect("/claims")

# ===============================
# ADMIN: Explain prediction
# ===============================
@app.route("/explain/<int:cid>")
def explain(cid):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT fraud_prediction, fraud_probability, shap_reason,
               user, claim_data, decision
        FROM claims WHERE id=?
    """, (cid,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Claim not found", 404

    reasons = json.loads(row[2]) if row[2] else []

    # Parse claim data for context
    claim_data_raw = json.loads(row[4]) if row[4] else {}
    def _get(field, default=None):
        val = claim_data_raw.get(field, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val

    claim_context = {
        "user": row[3],
        "decision": row[5] or "Pending",
        "total_claim_amount": _get("total_claim_amount", 0),
        "incident_type": _get("incident_type", "—"),
        "incident_severity": _get("incident_severity", "—"),
        "auto_make": _get("auto_make", "—"),
        "auto_model": _get("auto_model", "—"),
        "auto_year": _get("auto_year", "—"),
        "authorities_contacted": _get("authorities_contacted", "—"),
        "police_report_available": _get("police_report_available", "—"),
        "witnesses": _get("witnesses", 0),
        "property_damage": _get("property_damage", "—"),
    }

    # Separate business rule flags from SHAP numeric reasons
    rule_flags = [r for r in reasons if isinstance(r.get("impact"), str)]
    shap_reasons = [r for r in reasons if not isinstance(r.get("impact"), str)]

    return render_template(
        "explain.html",
        claim_id=cid,
        prediction=row[0],
        probability=float(row[1]) if row[1] is not None else 0.0,
        reasons=reasons,
        rule_flags=rule_flags,
        shap_reasons=shap_reasons,
        claim_context=claim_context,
        shap_available=SHAP_AVAILABLE
    )

# ===============================
# Logout
# ===============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ===============================
# USER: Bank details
# ===============================
@app.route("/bank_details", methods=["GET", "POST"])
def bank_details():
    if "user" not in session:
        return redirect("/")

    username = session["user"]
    conn = get_db()
    cur = conn.cursor()

    # Check if user has at least one approved claim
    cur.execute("SELECT COUNT(*) FROM claims WHERE user=? AND decision='Approved'", (username,))
    has_approved_claim = cur.fetchone()[0] > 0
    
    if not has_approved_claim:
        conn.close()
        return render_template("bank_details.html", requires_approval=True)

    if request.method == "POST":
        account_holder = request.form.get("account_holder", "").strip()
        bank_name = request.form.get("bank_name", "").strip()
        account_number = request.form.get("account_number", "").strip()
        ifsc_code = request.form.get("ifsc_code", "").strip().upper()
        account_type = request.form.get("account_type", "Savings")

        if not all([account_holder, bank_name, account_number, ifsc_code]):
            cur.execute("SELECT * FROM bank_details WHERE username=?", (username,))
            existing = cur.fetchone()
            conn.close()
            return render_template("bank_details.html", error="All fields are required.", existing=existing)

        cur.execute("""
            INSERT INTO bank_details (username, account_holder, bank_name, account_number, ifsc_code, account_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                account_holder=excluded.account_holder,
                bank_name=excluded.bank_name,
                account_number=excluded.account_number,
                ifsc_code=excluded.ifsc_code,
                account_type=excluded.account_type
        """, (username, account_holder, bank_name, account_number, ifsc_code, account_type))
        conn.commit()

        # Recalculate total payout for the POST response too
        cur.execute("""
            SELECT SUM(repair_cost_total) FROM claims
            WHERE user=? AND decision='Approved' AND repair_cost_total IS NOT NULL AND repair_cost_total > 0
        """, (username,))
        payout_row = cur.fetchone()
        total_payout = float(payout_row[0]) if payout_row and payout_row[0] else None

        conn.close()
        return render_template("bank_details.html",
            success="Bank details saved successfully!",
            total_payout=total_payout,
            existing=(None, username, account_holder, bank_name, account_number, ifsc_code, account_type))

    cur.execute("SELECT * FROM bank_details WHERE username=?", (username,))
    existing = cur.fetchone()

    # Get total repair payout from all approved claims
    cur.execute("""
        SELECT SUM(repair_cost_total) FROM claims
        WHERE user=? AND decision='Approved' AND repair_cost_total IS NOT NULL AND repair_cost_total > 0
    """, (username,))
    payout_row = cur.fetchone()
    total_payout = float(payout_row[0]) if payout_row and payout_row[0] else None

    conn.close()
    return render_template("bank_details.html", existing=existing, total_payout=total_payout)


# ===============================
# ADMIN: View user bank details + initiate payout
# ===============================
@app.route("/admin/bank_details/<username>")
def admin_bank_details(username):
    if "user" not in session or session.get("role") != "admin":
        return redirect("/")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bank_details WHERE username=?", (username,))
    details = cur.fetchone()

    # Get all approved claims with LIVE repair cost sums for this user
    cur.execute("""
        SELECT c.id, c.repair_cost_total, c.payout_sent,
               (SELECT COUNT(*) FROM vehicle_images vi WHERE vi.claim_id = c.id AND vi.repair_cost IS NOT NULL) as analyzed_count,
               (SELECT COUNT(*) FROM vehicle_images vi WHERE vi.claim_id = c.id) as total_images,
               (SELECT COALESCE(SUM(vi.repair_cost), 0) FROM vehicle_images vi WHERE vi.claim_id = c.id AND vi.repair_cost IS NOT NULL) as live_repair_total
        FROM claims c
        WHERE c.user=? AND c.decision='Approved'
        ORDER BY c.id DESC
    """, (username,))
    approved_claims = cur.fetchall()
    conn.close()

    return render_template("admin_bank_details.html",
        details=details,
        claimant=username,
        approved_claims=approved_claims
    )

# ===============================
# ADMIN: Send payout for a claim
# ===============================
@app.route("/admin/send_payout/<int:cid>", methods=["POST"])
def send_payout(cid):
    if "user" not in session or session.get("role") != "admin":
        return redirect("/")
    conn = get_db()
    cur = conn.cursor()

    # Fetch claim owner
    cur.execute("SELECT user FROM claims WHERE id=?", (cid,))
    owner_row = cur.fetchone()
    claimant = owner_row[0] if owner_row else ""

    # Block payout if user has not submitted bank details
    cur.execute("SELECT id FROM bank_details WHERE username=?", (claimant,))
    bank_row = cur.fetchone()
    if not bank_row:
        conn.close()
        return redirect(f"/admin/bank_details/{claimant}?error=no_bank_details")

    # Re-sum image costs at payout time in case more images were analyzed since approval
    cur.execute("""
        SELECT SUM(repair_cost) FROM vehicle_images
        WHERE claim_id=? AND repair_cost IS NOT NULL
    """, (cid,))
    row = cur.fetchone()
    total = float(row[0]) if row and row[0] is not None else 0.0
    cur.execute(
        "UPDATE claims SET payout_sent=1, repair_cost_total=? WHERE id=?",
        (total, cid)
    )

    # Notify the claimant that their payout has been sent
    amount_str = f"₹{total:,.2f}" if total > 0 else "the approved amount"
    notification_message = (
        f"💸 Your payout of {amount_str} for Claim #{cid} has been sent to your registered bank account. "
        "Please allow 1-3 business days for the funds to reflect."
    )
    cur.execute("""
        INSERT INTO notifications (username, message, claim_id)
        VALUES (?, ?, ?)
    """, (claimant, notification_message, cid))

    conn.commit()
    conn.close()
    return redirect(f"/admin/bank_details/{claimant}")

from flask import send_from_directory

@app.route('/download_image/<filename>')
def download_image(filename):
    if "user" not in session or session.get("role") != "admin":
        return "Unauthorized", 403
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# ===============================
# ADMIN: Analyze Image API
# ===============================
@app.route('/analyze_image/<int:claim_id>/<filename>', methods=['GET', 'POST'])
def analyze_image(claim_id, filename):
    if "user" not in session or session.get("role") != "admin":
        return "Unauthorized", 403
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    # Check database cache first to avoid re-running heavy models on page refresh
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT analysis_json, annotated_filename FROM vehicle_images WHERE claim_id=? AND filename=?", (claim_id, filename))
    row = cur.fetchone()
    conn.close()
    
    if row and row[0]: # If analysis_json exists
        analysis_result = json.loads(row[0])
        return render_template("analysis_result.html", claim_id=claim_id, filename=filename, annotated_filename=row[1], result=analysis_result)

    try:
        # Default metadata since we don't have driver properties stored natively in the same way
        driver_age = 35 
        car_year = 2015
        mileage = 60000

        # Try mapping actual claim data
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT claim_data FROM claims WHERE id=?", (claim_id,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            claim_data = json.loads(row[0])
            driver_age = int(claim_data.get('age', 35))
            car_year = int(claim_data.get('auto_year', 2015))

        damage_type, confidence = predict_damage_type(filepath)
        severity_score = predict_severity(filepath, damage_type, confidence)
        severity_level = get_severity_level(severity_score)
        features = extract_features_for_cost(filepath)

        X_cost = np.hstack([
            features.reshape(1, -1),
            [[driver_age, car_year, mileage, severity_score]]
        ])
        repair_cost = float(xgb_cost_model.predict(X_cost)[0])

        damage_icon = get_damage_icon(damage_type)
        recommendations = get_recommendations(damage_type, severity_level, repair_cost)

        detected_parts = []
        annotated_filename = None

        if severity_score >= 4.5:
             damage_type = "Total Loss / Severe Wreck"
             damage_icon = "⛔"
             if "🚗 Vehicle considered a Total Loss" not in recommendations:
                 recommendations.insert(0, "🚗 Vehicle considered a Total Loss")
             if "💰 Repair costs likely exceed vehicle value" not in recommendations:
                 recommendations.insert(1, "💰 Repair costs likely exceed vehicle value")
        elif severity_score >= 3.8 and damage_type in ['Broken Lamp', 'Flat Tire', 'Scratch']:
             damage_type = "Severe Wreck / Total Loss"
             damage_icon = "⛔"
             if "🚗 Vehicle likely a Total Loss" not in recommendations:
                 recommendations.insert(0, "🚗 Vehicle likely a Total Loss")
             if "💰 Repair costs may exceed vehicle value" not in recommendations:
                 recommendations.insert(1, "💰 Repair costs may exceed vehicle value")

        analysis_result = {
            'damage_type': damage_type,
            'damage_icon': damage_icon,
            'severity_score': round(severity_score, 2),
            'severity_level': severity_level,
            'repair_cost': round(repair_cost, 2),
            'confidence': round(confidence * 100, 1) if confidence > 0 else None,
            'detected_parts': detected_parts,
            'recommendations': recommendations
        }

        # Save to DB
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE vehicle_images 
            SET damage_type=?, severity_score=?, repair_cost=?, analysis_json=?, annotated_filename=?
            WHERE claim_id=? AND filename=?
        """, (damage_type, severity_score, repair_cost, json.dumps(analysis_result), annotated_filename, claim_id, filename))
        conn.commit()
        conn.close()

        return render_template("analysis_result.html", claim_id=claim_id, filename=filename, annotated_filename=annotated_filename, result=analysis_result)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

# ===============================
# LINKEDIN PROJECT SHOWCASE
# ===============================
@app.route("/linkedin_project")
def linkedin_project():
    return render_template("linkedin_project.html")

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=5000)

