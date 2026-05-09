import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import json
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Microfinance Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    :root {
        --primary-color: #1f4d6d;
        --secondary-color: #2d7a99;
        --accent-color: #4CAF50;
        --danger-color: #ff6b6b;
    }
    
    .main {
        background: linear-gradient(135deg, #1f4d6d 0%, #2d7a99 100%);
    }
    
    .sidebar .sidebar-content {
        background: #1f4d6d;
    }
    
    .metric-card {
        background: rgba(45, 122, 153, 0.2);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .objective-card {
        background: rgba(45, 122, 153, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid;
    }
    
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# DB CONNECTION
# =========================
def db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="fintech_db",
            autocommit=True
        )
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

# =========================
# SESSION
# =========================
if "user" not in st.session_state:
    st.session_state.user = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

# =========================
# HELPER FUNCTIONS
# =========================
def safe_int(value):
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except:
        return 0

def safe_float(value):
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except:
        return 0.0

def clean_data(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    return df

# =========================
# BULK UPLOAD FUNCTION
# =========================
def bulk_upload_borrowers(df):
    """Upload multiple borrowers from dataframe"""
    inserted = 0
    errors = []
    
    required_columns = ['name', 'age', 'income']
    
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            return 0, errors
    
    for idx, row in df.iterrows():
        try:
            name = str(row.get('name', ''))
            age = safe_int(row.get('age', 18))
            income = safe_float(row.get('income', 0))
            repayment_history = safe_float(row.get('repayment_history', 80))
            previous_loans = safe_int(row.get('previous_loans', 0))
            defaults = safe_int(row.get('defaults', 0))
            transaction_freq = safe_float(row.get('transaction_freq', 5))
            
            if name and income > 0:
                if add_borrower((name, age, income, repayment_history, previous_loans, defaults, transaction_freq)):
                    inserted += 1
                else:
                    errors.append(f"Row {idx+1}: Failed to add {name}")
            else:
                errors.append(f"Row {idx+1}: Invalid name or income")
        except Exception as e:
            errors.append(f"Row {idx+1}: {str(e)}")
    
    return inserted, errors

# =========================
# AUTH
# =========================
def register_user(username, password, role):
    conn = db()
    if not conn:
        return False
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        if cur.fetchone():
            conn.close()
            return False

        hashed = generate_password_hash(password)
        cur.execute("""
            INSERT INTO users(username,password,role,status,created_at)
            VALUES(%s,%s,%s,'pending',%s)
        """, (username, hashed, role, datetime.now()))
        conn.close()
        return True
    except:
        conn.close()
        return False

def login_user(username, password):
    conn = db()
    if not conn:
        return None
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        conn.close()

        if not user:
            return None
        if user["status"] != "approved":
            return "PENDING"
        if check_password_hash(user["password"], password):
            return user
        return None
    except:
        conn.close()
        return None

# =========================
# BORROWERS
# =========================
def add_borrower(data):
    try:
        conn = db()
        if not conn:
            return False
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO borrowers(name,age,income,repayment_history,previous_loans,defaults,transaction_freq,created_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
        """, (*data, datetime.now()))
        conn.close()
        return True
    except:
        return False
        
def get_borrowers():
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()

        df = pd.read_sql("SELECT * FROM borrowers", conn)
        conn.close()

        if len(df) > 0:
            df = clean_data(df)

            for col in ['age', 'income', 'repayment_history', 'previous_loans', 'defaults', 'transaction_freq']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except:
        return pd.DataFrame()

def get_loans():
    """
    FIXED: Now retrieves ALL loans (approved + rejected + pending_review)
    """
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()

        df = pd.read_sql("SELECT * FROM loans", conn)
        conn.close()

        if len(df) > 0:
            for col in ['risk_score', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'status' in df.columns:
                df['status'] = df['status'].astype(str)

        return df
    except:
        return pd.DataFrame()

# =========================
# PREDICTION LOGGING
# =========================
def log_prediction(bid, model_name, risk_score, features_json, decision="REVIEW", reason=""):
    try:
        conn = db()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_logs(borrower_id, model_name, risk_score, features, decision, reason, created_at)
            VALUES(%s, %s, %s, %s, %s, %s, %s)
        """, (bid, model_name, risk_score, features_json, decision, reason, datetime.now()))
        conn.close()
    except:
        pass

def get_prediction_logs(days=30):
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()
        query = f"""
            SELECT * FROM prediction_logs 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY)
            ORDER BY created_at DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# =========================
# LOAN DECISION TRACKING
# =========================
def calculate_loan_amount(borrower, risk):
    income = safe_float(borrower["income"])
    age = safe_float(borrower["age"])
    repayment = safe_float(borrower["repayment_history"])
    prev_loans = safe_float(borrower["previous_loans"])
    defaults = safe_float(borrower["defaults"])
    txn_freq = safe_float(borrower["transaction_freq"])

    base = income * 0.5

    repayment_score = repayment / 100
    txn_score = min(txn_freq / 50, 1)
    experience_score = min(prev_loans / 10, 1)

    behavior_score = (repayment_score * 0.5) + (txn_score * 0.3) + (experience_score * 0.2)

    default_penalty = max(0.2, 1 - (defaults * 0.25))
    risk_penalty = max(0.1, 1 - risk / 100)

    if age < 25:
        age_factor = 0.8
    elif age < 60:
        age_factor = 1.0
    else:
        age_factor = 0.7

    amount = base * behavior_score * default_penalty * risk_penalty * age_factor

    amount = max(10000, amount)
    amount = min(amount, income * 1.5)

    return round(amount, 2)


def calculate_duration(amount, borrower):
    income = safe_float(borrower["income"])

    if amount < income * 0.3:
        return 6
    elif amount < income:
        return 12
    else:
        return 18


def create_loan_with_decision(bid, amount, duration, risk_score, model_choice, decision, reason):
    """
    FIXED: Now saves ALL loans (Approved + Rejected) regardless of decision outcome
    """
    try:
        conn = db()
        if not conn:
            return False

        cur = conn.cursor()
        
        # Map decision to database status - use lowercase to match MySQL ENUM
        status_map = {
            'APPROVE': 'approved',
            'REJECT': 'rejected',
            'REVIEW': 'pending_review'
        }
        status = status_map.get(decision, 'pending_review')

        cur.execute("""
            INSERT INTO loans(borrower_id, amount, duration, risk_score, model_name, status, decision_reason, created_at)
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            safe_int(bid),
            safe_float(amount) if decision == 'APPROVE' else 0,
            safe_int(duration) if decision == 'APPROVE' else 0,
            safe_float(risk_score),
            model_choice,
            status,
            reason,
            datetime.now()
        ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error creating loan: {e}")
        return False


def get_loan_decisions():
    """
    FIXED: Now retrieves ALL loan decisions (approved + rejected)
    """
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()

        query = """
            SELECT 
                l.id,
                b.name AS borrower_name,
                b.id AS borrower_id,
                b.income,
                b.age,
                b.repayment_history,
                b.previous_loans,
                b.defaults,
                b.transaction_freq,
                l.amount,
                l.duration,
                l.risk_score,
                l.model_name,
                l.status,
                COALESCE(l.decision_reason, 'No reason') AS decision_reason,
                l.created_at,
                COALESCE(l.actual_default, 0) AS actual_default
            FROM loans l
            JOIN borrowers b ON l.borrower_id = b.id
            ORDER BY l.created_at DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()
        return df

    except:
        return pd.DataFrame()


def get_decision_statistics():
    """
    FIXED: Now counts ALL loan statuses
    """
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()

        query = """
            SELECT 
                status,
                COUNT(*) as count,
                AVG(risk_score) as avg_risk,
                AVG(amount) as avg_amount
            FROM loans
            GROUP BY status
        """

        df = pd.read_sql(query, conn)
        conn.close()
        return df

    except:
        return pd.DataFrame()


def auto_generate_loans():
    """
    FIXED: Now saves ALL loan decisions (Approved + Rejected)
    """
    borrowers = get_borrowers()

    if len(borrowers) < 20:
        st.warning(f"Need at least 20 borrowers. Current: {len(borrowers)}")
        return

    results, X_test, y_test = train_models(borrowers)

    if not results:
        st.error("Model training failed")
        return

    best_model_name = max(results.items(), key=lambda x: x[1]["ROC-AUC"])[0]
    best_model_info = results[best_model_name]

    st.success(f"Using model: {best_model_name} (ROC-AUC: {best_model_info['ROC-AUC']:.3f})")

    progress_bar = st.progress(0)
    status_text = st.empty()

    approved_count = 0
    rejected_count = 0
    error_count = 0

    for idx, (_, borrower) in enumerate(borrowers.iterrows()):
        progress_bar.progress((idx + 1) / len(borrowers))
        status_text.text(f"Processing {idx+1}/{len(borrowers)}: {borrower['name']}")

        try:
            bid = borrower["id"]

            data = [
                safe_float(borrower["income"]),
                safe_float(borrower["age"]),
                safe_float(borrower["repayment_history"]),
                safe_float(borrower["previous_loans"]),
                safe_float(borrower["transaction_freq"])
            ]

            risk = predict_risk(
                best_model_info["model"],
                best_model_info["scaler"],
                data,
                best_model_name,
                best_model_info["features"]
            )

            decision, reason = get_decision_and_reason(
                risk,
                safe_float(borrower["repayment_history"]),
                safe_float(borrower["defaults"])
            )

            if decision == "APPROVE":
                amount = calculate_loan_amount(borrower, risk)
                duration = calculate_duration(amount, borrower)
                approved_count += 1
            else:
                amount = 0
                duration = 0
                rejected_count += 1

            if create_loan_with_decision(
                bid, amount, duration, risk,
                best_model_name, decision, reason
            ):
                log_prediction(bid, best_model_name, risk, json.dumps(data), decision, reason)
            else:
                error_count += 1

        except Exception as e:
            print(f"Error processing borrower {bid}: {e}")
            error_count += 1

    progress_bar.empty()
    status_text.empty()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", approved_count + rejected_count)
    with col2:
        st.metric("Approved", approved_count)
    with col3:
        st.metric("Rejected", rejected_count)
    with col4:
        rate = (approved_count / (approved_count + rejected_count) * 100) if (approved_count + rejected_count) else 0
        st.metric("Approval Rate", f"{rate:.1f}%")

    st.success("Loan generation completed!")

# =========================
# MODEL TRAINING
# =========================
def train_models(df):
    if len(df) < 20:
        return {}, None, None
    
    try:
        df["default_flag"] = (df["defaults"] > 0).astype(int)
        features = ["income", "age", "repayment_history", "previous_loans", "transaction_freq"]
        
        X = df[features].copy()
        y = df["default_flag"].copy()
        
        X = X.fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, max(1, (y_train == 1).sum() - 1)))
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        except:
            X_train_smote, y_train_smote = X_train, y_train
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            if model_name == "Logistic Regression":
                model.fit(X_train_scaled, y_train_smote)
                pred = model.predict(X_test_scaled)
                pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                cv_scores = cross_val_score(model, X_train_scaled, y_train_smote, cv=5, scoring='roc_auc')
                X_test_use = X_test_scaled
            else:
                model.fit(X_train_smote, y_train_smote)
                pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)[:, 1]
                cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
                X_test_use = X_test
            
            results[model_name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, pred_proba),
                "CV Mean": cv_scores.mean(),
                "CV Std": cv_scores.std(),
                "model": model,
                "scaler": scaler if model_name == "Logistic Regression" else None,
                "pred_proba": pred_proba,
                "y_test": y_test,
                "predictions": pred,
                "X_test": X_test_use,
                "features": features,
                "y_pred": pred,
                "y_prob": pred_proba
            }
        
        return results, X_test, y_test
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return {}, None, None

def predict_risk(model, scaler, data, model_name, features):
    if model is None:
        return 50
    
    try:
        if model_name == "Logistic Regression":
            if scaler is not None:
                data_scaled = scaler.transform([data])
                risk = model.predict_proba(data_scaled)[0][1] * 100
            else:
                risk = model.predict_proba([data])[0][1] * 100
        else:
            risk = model.predict_proba([data])[0][1] * 100
        return risk
    except Exception as e:
        st.error(f"Risk prediction error: {str(e)}")
        return 50

def get_decision_and_reason(risk_score, repayment_history, defaults):
    reasons = []
    decision = "APPROVE"
    
    if risk_score >= 60:
        decision = "REJECT"
        reasons.append(f"High risk: {risk_score:.2f}%")
    elif risk_score >= 30:
        decision = "REVIEW"
        reasons.append(f"Medium risk: {risk_score:.2f}%")
    else:
        reasons.append(f"Low risk: {risk_score:.2f}%")
    
    if repayment_history >= 80:
        reasons.append("Excellent repayment (≥80%)")
    elif repayment_history >= 60:
        reasons.append("Good repayment (60-80%)")
    else:
        reasons.append("Poor repayment (<60%)")
    
    if defaults > 0:
        decision = "REJECT"
        reasons.append(f"Previous defaults: {safe_int(defaults)}")
    
    return decision, " | ".join(reasons)

def get_pending_users():
    try:
        conn = db()
        if not conn:
            return pd.DataFrame()
        df = pd.read_sql("SELECT * FROM users WHERE status='pending'", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def approve_user(uid):
    try:
        conn = db()
        if not conn:
            return False
        cur = conn.cursor()
        cur.execute("UPDATE users SET status='approved' WHERE id=%s", (uid,))
        conn.close()
        return True
    except:
        return False

# =========================
# ENHANCED FORECASTING FUNCTIONS (LEARNS FROM DATABASE)
# =========================
def get_database_stats():
    """
    Get comprehensive statistics from database
    """
    try:
        conn = db()
        if not conn:
            return None
        
        borrowers = pd.read_sql("SELECT * FROM borrowers", conn)
        loans = pd.read_sql("SELECT * FROM loans", conn)
        predictions = pd.read_sql("SELECT * FROM prediction_logs", conn)
        
        conn.close()
        
        return {
            'borrowers_count': len(borrowers),
            'loans_count': len(loans),
            'predictions_count': len(predictions),
            'borrowers_df': borrowers,
            'loans_df': loans,
            'predictions_df': predictions
        }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return None


def forecast_default_risk(days=90):
    """
    UPDATED: Learns from loans table to forecast default risk
    Uses risk_score trends from actual loan decisions
    """
    try:
        conn = db()
        if not conn:
            return None
        
        # Query loans with risk scores
        query = """
            SELECT created_at, risk_score, status
            FROM loans 
            WHERE risk_score IS NOT NULL 
            ORDER BY created_at ASC
        """
        loans_df = pd.read_sql(query, conn)
        conn.close()
        
        if loans_df is None or len(loans_df) < 2:
            return None
        
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df = loans_df.sort_values('created_at')
        
        # Group by hour for faster trend detection
        hourly_risk = loans_df.groupby(loans_df['created_at'].dt.floor('H')).agg({
            'risk_score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        hourly_risk.columns = ['date', 'avg_risk', 'std_risk', 'min_risk', 'max_risk', 'count']
        hourly_risk = hourly_risk.sort_values('date')
        
        if len(hourly_risk) < 2:
            return None
        
        x = np.arange(len(hourly_risk))
        y = hourly_risk['avg_risk'].values
        
        # Use linear fit for 2 points, quadratic for more
        if len(hourly_risk) == 2:
            z = np.polyfit(x, y, 1)
        else:
            z = np.polyfit(x, y, min(2, len(hourly_risk) - 1))
        p = np.poly1d(z)
        
        # Generate forecast
        future_x = np.arange(len(hourly_risk), len(hourly_risk) + days)
        forecast_y = p(future_x)
        forecast_y = np.clip(forecast_y, 0, 100)
        
        # Calculate R-squared
        if len(y) > 1 and np.std(y) > 0:
            r_squared = 1 - (np.sum((y - p(x))**2) / np.sum((y - y.mean())**2))
        else:
            r_squared = 0.5
        
        # Calculate trend analysis
        trend_direction = "Increasing" if forecast_y[-1] > y[-1] else "Decreasing"
        
        return {
            'historical_dates': hourly_risk['date'].astype(str).tolist(),
            'historical_values': hourly_risk['avg_risk'].tolist(),
            'forecast_dates': [(hourly_risk['date'].max() + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(1, min(days+1, 50))],
            'forecast_values': forecast_y[:min(days, 49)].tolist(),
            'r_squared': r_squared,
            'data_points': len(hourly_risk),
            'total_loans_analyzed': len(loans_df),
            'trend': trend_direction,
            'current_avg_risk': float(y[-1]),
            'forecast_avg_risk': float(forecast_y[-1])
        }
    except Exception as e:
        print(f"Forecast error: {e}")
        return None


def forecast_approval_rate(days=30):
    """
    UPDATED: Learns from loans table to forecast approval rate
    Analyzes approved vs rejected loans
    """
    try:
        conn = db()
        if not conn:
            return None
        
        query = """
            SELECT created_at, status
            FROM loans
            ORDER BY created_at ASC
        """
        loans_df = pd.read_sql(query, conn)
        conn.close()
        
        if loans_df is None or len(loans_df) < 2:
            return None
        
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df = loans_df.sort_values('created_at')
        
        # Group by hour and calculate approval rate
        hourly_approvals = loans_df.groupby(loans_df['created_at'].dt.floor('H')).apply(
            lambda x: (len(x[x['status'] == 'approved']) / len(x) * 100) if len(x) > 0 else 0
        ).reset_index()
        hourly_approvals.columns = ['date', 'approval_rate']
        hourly_approvals = hourly_approvals.sort_values('date')
        
        if len(hourly_approvals) < 2:
            return None
        
        x = np.arange(len(hourly_approvals))
        y = hourly_approvals['approval_rate'].values
        
        # Use linear fit for 2 points, quadratic for more
        if len(hourly_approvals) == 2:
            z = np.polyfit(x, y, 1)
        else:
            z = np.polyfit(x, y, min(2, len(hourly_approvals) - 1))
        p = np.poly1d(z)
        
        # Generate forecast
        future_x = np.arange(len(hourly_approvals), len(hourly_approvals) + days)
        forecast_y = p(future_x)
        forecast_y = np.clip(forecast_y, 0, 100)
        
        # Calculate R-squared
        if len(y) > 1 and np.std(y) > 0:
            r_squared = 1 - (np.sum((y - p(x))**2) / np.sum((y - y.mean())**2))
        else:
            r_squared = 0.5
        
        # Calculate trend analysis
        approved_count = len(loans_df[loans_df['status'] == 'approved'])
        rejected_count = len(loans_df[loans_df['status'] == 'rejected'])
        
        return {
            'historical_dates': hourly_approvals['date'].astype(str).tolist(),
            'historical_values': hourly_approvals['approval_rate'].tolist(),
            'forecast_dates': [(hourly_approvals['date'].max() + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(1, min(days+1, 50))],
            'forecast_values': forecast_y[:min(days, 49)].tolist(),
            'r_squared': r_squared,
            'data_points': len(hourly_approvals),
            'total_decisions': len(loans_df),
            'approved': approved_count,
            'rejected': rejected_count,
            'current_approval_rate': float(y[-1]),
            'forecast_approval_rate': float(forecast_y[-1])
        }
    except Exception as e:
        print(f"Forecast error: {e}")
        return None


def forecast_loan_amount(days=30):
    """
    UPDATED: Learns from loans table to forecast loan amounts
    Uses approved loan amounts to predict future trends
    """
    try:
        conn = db()
        if not conn:
            return None
        
        query = """
            SELECT created_at, amount
            FROM loans
            WHERE status = 'approved' AND amount > 0
            ORDER BY created_at ASC
        """
        loans_df = pd.read_sql(query, conn)
        conn.close()
        
        if loans_df is None or len(loans_df) < 2:
            return None
        
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df['amount'] = pd.to_numeric(loans_df['amount'], errors='coerce')
        loans_df = loans_df[loans_df['amount'] > 0]
        
        if len(loans_df) < 2:
            return None
        
        loans_df = loans_df.sort_values('created_at')
        
        # Group by hour and calculate average amount
        hourly_amounts = loans_df.groupby(loans_df['created_at'].dt.floor('H')).agg({
            'amount': ['mean', 'sum', 'count', 'min', 'max']
        }).reset_index()
        hourly_amounts.columns = ['date', 'avg_amount', 'total_amount', 'count', 'min_amount', 'max_amount']
        hourly_amounts = hourly_amounts.sort_values('date')
        
        if len(hourly_amounts) < 2:
            return None
        
        x = np.arange(len(hourly_amounts))
        y = hourly_amounts['avg_amount'].values
        
        # Use linear fit for 2 points, quadratic for more
        if len(hourly_amounts) == 2:
            z = np.polyfit(x, y, 1)
        else:
            z = np.polyfit(x, y, min(2, len(hourly_amounts) - 1))
        p = np.poly1d(z)
        
        # Generate forecast
        future_x = np.arange(len(hourly_amounts), len(hourly_amounts) + days)
        forecast_y = p(future_x)
        forecast_y = np.maximum(forecast_y, 0)
        
        # Calculate R-squared
        if len(y) > 1 and np.std(y) > 0:
            r_squared = 1 - (np.sum((y - p(x))**2) / np.sum((y - y.mean())**2))
        else:
            r_squared = 0.5
        
        # Calculate trend analysis
        trend_direction = "Increasing" if forecast_y[-1] > y[-1] else "Decreasing"
        
        return {
            'historical_dates': hourly_amounts['date'].astype(str).tolist(),
            'historical_values': hourly_amounts['avg_amount'].tolist(),
            'forecast_dates': [(hourly_amounts['date'].max() + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(1, min(days+1, 50))],
            'forecast_values': forecast_y[:min(days, 49)].tolist(),
            'r_squared': r_squared,
            'data_points': len(hourly_amounts),
            'total_approved_loans': len(loans_df),
            'current_avg_amount': float(y[-1]),
            'forecast_avg_amount': float(forecast_y[-1]),
            'trend': trend_direction,
            'total_approved_value': float(hourly_amounts['total_amount'].sum())
        }
    except Exception as e:
        print(f"Forecast error: {e}")
        return None

# =========================
# AUTH SCREEN
# =========================
if st.session_state.user is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏦 Microfinance Loan Default Prediction")
        st.divider()
        
        menu = st.radio("Choose Action", ["Login", "Register"], horizontal=True)
        
        if menu == "Register":
            st.subheader("Register New Account")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            r = st.selectbox("Role", ["loan_officer", "risk_manager", "admin"])
            
            if st.button("Register"):
                if register_user(u, p, r):
                    st.success("✅ Registered! Await admin approval")
                else:
                    st.error("❌ Username exists")
        
        else:
            st.subheader("Login to System")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            
            if st.button("Login"):
                user = login_user(u, p)
                
                if user == "PENDING":
                    st.warning("⏳ Await admin approval")
                elif user:
                    st.session_state.user = user
                    st.success("✅ Login successful")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

    st.stop()

# =========================
# AFTER LOGIN
# =========================
user = st.session_state.user
role = user["role"]

with st.sidebar:
    st.markdown(f"### 👤 {user['username'].upper()}")
    st.markdown(f"**Role:** {role}")
    st.divider()
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.trained_models = None
        st.rerun()

with st.sidebar:
    st.markdown("### 📊 NAVIGATION")
    menu = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Objectives",
            "Risk Analysis",
            "Model Building",
            "Evaluation",
            "Loan Decisions",
            "Rejected Applications",
            "Forecasting",
            "Integration",
            "Settings",
            "Audit Logs"
        ]
    )

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.markdown("# 🏦 Dashboard")
    st.divider()
    
    try:
        loans = get_loans()
        borrowers = get_borrowers()

        approved = len(loans[loans["status"] == "approved"]) if len(loans) > 0 else 0
        rejected = len(loans[loans["status"] == "rejected"]) if len(loans) > 0 else 0
        total_loans = len(loans)
        total_exposure = safe_float(loans[loans["status"] == "approved"]['amount'].sum()) if len(loans) > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("👥 Total Borrowers", len(borrowers))
        with col2:
            st.metric("📋 Total Loans", total_loans)
        with col3:
            st.metric("✅ Approved", approved)
        with col4:
            st.metric("❌ Rejected", rejected)
        with col5:
            st.metric("💰 Total Exposure", f"${total_exposure:,.0f}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if len(loans) > 0:
                try:
                    status_counts = loans["status"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="📊 Loan Status Distribution",
                        color_discrete_map={"approved": "#2ecc71", "rejected": "#e74c3c", "pending_review": "#f39c12"}
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Status chart error: {str(e)}")
            else:
                st.info("No loans to display")

        with col2:
            if len(loans) > 0:
                try:
                    risk_data = loans['risk_score'].dropna()
                    if len(risk_data) > 0:
                        risk_binned = pd.cut(risk_data, bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
                        risk_counts = risk_binned.value_counts().sort_index()
                        
                        fig = px.bar(
                            x=risk_counts.index.astype(str),
                            y=risk_counts.values,
                            title="📊 Risk Score Distribution",
                            labels={"x": "Risk Level", "y": "Count"},
                            color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"]
                        )
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Risk chart error: {str(e)}")
            else:
                st.info("No risk data to display")

    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")

# =========================
# OBJECTIVES
# =========================
elif menu == "Objectives":
    st.markdown("# 🎯 System Objectives & Quick Predictor")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### System Objectives")
        
        objectives = [
            ("1. Assessment Analysis", "Examine existing loan assessment & credit risk management practices", "#e74c3c"),
            ("2. Risk Factors Analysis", "Identify and analyze key factors influencing loan defaults", "#f39c12"),
            ("3. Model Development", "Build predictive models using multiple ML algorithms", "#27ae60"),
            ("4. Performance Evaluation", "Evaluate model reliability and effectiveness", "#2ecc71"),
            ("5. Integration Framework", "Propose framework for practical adoption", "#e67e22")
        ]
        
        for title, desc, color in objectives:
            st.markdown(f"""
            <div style="background: rgba(45, 122, 153, 0.3); border-left: 5px solid {color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <b>{title}</b><br>
                {desc}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔮 Quick Predictor")
        
        applicant_income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
        applicant_age = st.number_input("Age", 18, 80, 35)
        applicant_repayment = st.number_input("Repayment History (%)", 0, 100, 75)
        applicant_prev_loans = st.number_input("Previous Loans", 0, 20, 2)
        applicant_transaction = st.number_input("Transaction Frequency", 0, 100, 10)
        
        if st.button("Predict Default Risk", use_container_width=True, type="primary"):
            borrowers = get_borrowers()
            if len(borrowers) >= 20:
                results, _, _ = train_models(borrowers)
                if results:
                    best_model_name = max(results.items(), key=lambda x: x[1]["ROC-AUC"])[0]
                    best_model_info = results[best_model_name]
                    
                    features = ["income", "age", "repayment_history", "previous_loans", "transaction_freq"]
                    data = [applicant_income, applicant_age, applicant_repayment, applicant_prev_loans, applicant_transaction]
                    
                    risk = predict_risk(best_model_info["model"], best_model_info["scaler"], data, best_model_name, features)
                    
                    st.markdown(f"### ⚡ Result: {risk:.2f}% Default Risk")
                    
                    if risk < 30:
                        st.success("🟢 LOW RISK - Recommended for approval")
                    elif risk < 60:
                        st.warning("🟡 MEDIUM RISK - Requires manual review")
                    else:
                        st.error("🔴 HIGH RISK - Recommend rejection or collateral")
                else:
                    st.error("Model training failed")
            else:
                st.warning(f"Need 20+ borrowers. Current: {len(borrowers)}")

# =========================
# RISK ANALYSIS
# =========================
elif menu == "Risk Analysis":
    st.markdown("# 📊 Risk Analysis")
    st.divider()

    borrowers = get_borrowers()

    if len(borrowers) >= 5:
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_risk = borrowers['defaults'].mean()
            st.metric("📉 Average Default Rate", f"{avg_risk:.2%}")

        with col2:
            avg_income = borrowers['income'].mean()
            st.metric("💰 Average Income", f"${avg_income:,.0f}")

        with col3:
            avg_age = borrowers['age'].mean()
            st.metric("👤 Average Age", f"{avg_age:.0f} years")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            try:
                age_bins = pd.cut(borrowers['age'], bins=5)
                age_counts = age_bins.value_counts().sort_index()

                fig = px.bar(
                    x=age_counts.index.astype(str),
                    y=age_counts.values,
                    title="📊 Age Distribution",
                    labels={"x": "Age Range", "y": "Number of Borrowers"},
                    color_discrete_sequence=["#3498db"]
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Age chart error: {str(e)}")

        with col2:
            try:
                income_bins = pd.cut(borrowers['income'], bins=5)
                income_counts = income_bins.value_counts().sort_index()

                fig = px.bar(
                    x=income_counts.index.astype(str),
                    y=income_counts.values,
                    title="💰 Income Distribution",
                    labels={"x": "Income Range", "y": "Number of Borrowers"},
                    color_discrete_sequence=["#2ecc71"]
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Income chart error: {str(e)}")

        st.divider()

        try:
            borrowers_display = borrowers[['name', 'income', 'repayment_history']].copy()
            borrowers_display = borrowers_display.sort_values('income', ascending=False).head(15)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=borrowers_display['name'],
                y=borrowers_display['income'],
                name='Income ($)',
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Bar(
                x=borrowers_display['name'],
                y=borrowers_display['repayment_history'],
                name='Repayment History (%)',
                marker_color='#2ecc71'
            ))
            
            fig.update_layout(
                title="💰 Income vs Repayment History (Top 15 Borrowers)",
                xaxis_title="Borrower Name",
                yaxis_title="Values",
                barmode='group',
                template="plotly_dark",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Risk vs Income chart error: {str(e)}")

        st.markdown("### 👥 Borrower Data")
        st.dataframe(borrowers, use_container_width=True)

    else:
        st.warning(f"Need at least 5 borrowers. Current: {len(borrowers)}")

# =========================
# MODEL BUILDING
# =========================
elif menu == "Model Building":
    st.markdown("# 🤖 Model Building")
    st.divider()
    
    borrowers = get_borrowers()
    
    if len(borrowers) >= 20:
        if st.button("🚀 Train All Models", use_container_width=True, type="primary"):
            with st.spinner("Training models..."):
                results, X_test, y_test = train_models(borrowers)
                st.session_state.trained_models = results
                st.success("Models trained successfully!")
        
        if st.session_state.trained_models:
            results = st.session_state.trained_models
            
            st.markdown("### 📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                model_name: {
                    "Accuracy": info["Accuracy"],
                    "Precision": info["Precision"],
                    "Recall": info["Recall"],
                    "F1 Score": info["F1 Score"],
                    "ROC-AUC": info["ROC-AUC"],
                    "CV Mean": info["CV Mean"]
                }
                for model_name, info in results.items()
            }).T
            
            st.dataframe(metrics_df.round(4), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
                    metrics_melted = metrics_df[metrics_to_plot].reset_index()
                    metrics_melted = metrics_melted.melt(id_vars=['index'], var_name='Metric', value_name='Score')
                    
                    fig = px.bar(
                        metrics_melted,
                        x="index",
                        y="Score",
                        color="Metric",
                        title="📊 Performance Metrics Comparison (All Metrics)",
                        labels={"index": "Model", "Score": "Score"},
                        barmode="group",
                        color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
                    )
                    fig.update_layout(template="plotly_dark", height=500)
                    fig.update_yaxes(range=[0, 1.05])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Metrics chart error: {str(e)}")
            
            with col2:
                try:
                    roc_auc_data = pd.DataFrame({
                        "Model": list(results.keys()),
                        "ROC-AUC": [results[model_name]["ROC-AUC"] for model_name in results.keys()]
                    }).sort_values("ROC-AUC", ascending=True)
                    
                    fig = px.bar(
                        roc_auc_data,
                        y="Model",
                        x="ROC-AUC",
                        title="🏆 ROC-AUC Ranking (Actual Values)",
                        labels={"ROC-AUC": "AUC Score"},
                        color="ROC-AUC",
                        color_continuous_scale="Greens",
                        orientation="h",
                        text="ROC-AUC"
                    )
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig.update_layout(template="plotly_dark", height=500)
                    fig.update_xaxes(range=[0, 1.05])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"ROC-AUC chart error: {str(e)}")
    else:
        st.warning(f"Need at least 20 borrowers. Current: {len(borrowers)}")

# =========================
# EVALUATION
# =========================
elif menu == "Evaluation":
    st.markdown("# 📈 Model Evaluation")
    st.divider()
    
    borrowers = get_borrowers()
    
    if len(borrowers) >= 20:
        if not st.session_state.trained_models:
            st.info("Train models in 'Model Building' first")
        else:
            results = st.session_state.trained_models
            
            st.markdown("### 🎯 Best Model Performance")
            
            col1, col2, col3 = st.columns(3)
            best_model_name = max(results.items(), key=lambda x: x[1]["ROC-AUC"])[0]
            best_model_info = results[best_model_name]
            
            with col1:
                st.metric("Best Model", best_model_name)
            with col2:
                st.metric("ROC-AUC", f"{best_model_info['ROC-AUC']:.4f}")
            with col3:
                st.metric("Accuracy", f"{best_model_info['Accuracy']:.4f}")
            
            st.divider()
            
            st.markdown("### 🔄 ROC-AUC Curves Comparison")
            try:
                fig = go.Figure()
                
                for model_name, info in results.items():
                    y_test_actual = info["y_test"]
                    y_prob_actual = info["y_prob"]
                    
                    fpr, tpr, _ = roc_curve(y_test_actual, y_prob_actual)
                    auc_score = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, 
                        y=tpr, 
                        mode='lines',
                        name=f"{model_name} (AUC={auc_score:.4f})",
                        line=dict(width=3)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1], 
                    y=[0, 1], 
                    mode='lines', 
                    name='Random Classifier (AUC=0.5)',
                    line=dict(dash='dash', color='gray', width=2)
                ))
                
                fig.update_xaxes(title="False Positive Rate")
                fig.update_yaxes(title="True Positive Rate")
                fig.update_layout(
                    title="ROC Curves Comparison (Actual Model Predictions)",
                    height=600,
                    template="plotly_dark",
                    hovermode="closest",
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"ROC curve error: {str(e)}")
            
            st.markdown("### 🎲 Confusion Matrices (Actual Predictions)")
            
            cols = st.columns(3)
            
            for idx, (model_name, info) in enumerate(results.items()):
                with cols[idx]:
                    try:
                        y_test_actual = info["y_test"]
                        y_pred_actual = info["y_pred"]
                        
                        cm = confusion_matrix(y_test_actual, y_pred_actual)
                        
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=["No Default (0)", "Default (1)"],
                            y=["No Default (0)", "Default (1)"],
                            text=cm,
                            texttemplate="%{text}",
                            textfont={"size": 14},
                            colorscale="Blues"
                        ))
                        fig.update_layout(
                            title=f"{model_name}",
                            height=450,
                            template="plotly_dark",
                            xaxis_title="Predicted",
                            yaxis_title="Actual"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"""
                        **Confusion Matrix Details:**
                        - True Negatives (TN): {tn}
                        - False Positives (FP): {fp}
                        - False Negatives (FN): {fn}
                        - True Positives (TP): {tp}
                        - Sensitivity (Recall): {sensitivity:.4f}
                        - Specificity: {specificity:.4f}
                        """)
                    except Exception as e:
                        st.error(f"Matrix error for {model_name}: {str(e)}")
    else:
        st.warning(f"Need at least 20 borrowers. Current: {len(borrowers)}")

# =========================
# LOAN DECISIONS
# =========================
elif menu == "Loan Decisions":
    st.markdown("# ✅ Loan Decisions Overview")
    st.divider()

    decisions = get_loan_decisions()

    if len(decisions) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        approved = decisions[decisions['status'] == 'approved']
        rejected = decisions[decisions['status'] == 'rejected']
        
        with col1:
            st.metric("📋 Total Decisions", len(decisions))
        with col2:
            st.metric("✅ Approved", len(approved))
        with col3:
            st.metric("❌ Rejected", len(rejected))
        with col4:
            approval_rate = (len(approved) / len(decisions) * 100) if len(decisions) > 0 else 0
            st.metric("📊 Approval Rate", f"{approval_rate:.1f}%")

        st.divider()

        col1, col2 = st.columns(2)
        
        with col1:
            if len(approved) > 0:
                avg_amount = approved['amount'].mean()
                st.metric("💰 Avg Approved Amount", f"${avg_amount:,.0f}")
            else:
                st.metric("💰 Avg Approved Amount", "$0")

        with col2:
            if len(approved) > 0:
                avg_risk = approved['risk_score'].mean()
                st.metric("📊 Avg Risk (Approved)", f"{avg_risk:.2f}%")
            else:
                st.metric("📊 Avg Risk (Approved)", "0%")

        st.divider()

        try:
            approved_display = approved.sort_values('amount', ascending=False).head(15)
            
            if len(approved_display) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=approved_display['borrower_name'],
                    y=approved_display['amount'],
                    name='Loan Amount ($)',
                    marker_color='#3498db'
                ))
                
                fig.add_trace(go.Bar(
                    x=approved_display['borrower_name'],
                    y=approved_display['risk_score'],
                    name='Risk Score (%)',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title="💰 Approved Loan Amount & Risk Score by Borrower (Top 15)",
                    xaxis_title="Borrower Name",
                    yaxis_title="Values",
                    barmode='group',
                    template="plotly_dark",
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Loan amount chart error: {str(e)}")

        st.markdown("### 📋 All Loan Decisions")
        st.dataframe(
            decisions[['borrower_name', 'amount', 'duration', 'risk_score', 'status', 'decision_reason']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No loan decisions yet")

# =========================
# REJECTED APPLICATIONS
# =========================
elif menu == "Rejected Applications":
    st.markdown("# ❌ Rejected Applications")
    st.divider()

    decisions = get_loan_decisions()
    rejected = decisions[decisions['status'] == 'rejected']

    if len(rejected) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        total_loans = len(decisions)
        rejection_rate = (len(rejected) / total_loans * 100) if total_loans > 0 else 0
        avg_risk_rejected = rejected['risk_score'].mean()
        
        with col1:
            st.metric("❌ Total Rejected", len(rejected))
        with col2:
            st.metric("📊 Rejection Rate", f"{rejection_rate:.1f}%")
        with col3:
            st.metric("⚠️ Avg Risk Score", f"{avg_risk_rejected:.2f}%")
        with col4:
            st.metric("👥 Total Applications", total_loans)

        st.divider()

        try:
            rejection_reasons = rejected['decision_reason'].value_counts().head(10)
            fig = px.bar(
                x=rejection_reasons.index,
                y=rejection_reasons.values,
                title="❌ Top Rejection Reasons",
                labels={"x": "Reason", "y": "Count"},
                color_discrete_sequence=["#e74c3c"]
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Rejection reasons chart error: {str(e)}")

        try:
            fig = px.scatter(
                rejected,
                x="income",
                y="risk_score",
                size="repayment_history",
                color="defaults",
                hover_name="borrower_name",
                title="📊 Risk vs Income for Rejected Applicants",
                labels={"income": "Income ($)", "risk_score": "Risk Score (%)", "defaults": "Defaults"},
                color_continuous_scale="Reds"
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Scatter chart error: {str(e)}")

        st.markdown("### 📋 Rejected Applicants - Detailed View")
        
        for idx, (_, applicant) in enumerate(rejected.iterrows()):
            with st.expander(f"📌 {applicant['borrower_name']} - Risk: {applicant['risk_score']:.1f}%"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Income:** ${applicant['income']:,.0f}")
                    st.write(f"**Age:** {applicant['age']:.0f} years")
                    st.write(f"**Repayment History:** {applicant['repayment_history']:.1f}%")
                    st.write(f"**Previous Loans:** {applicant['previous_loans']:.0f}")
                
                with col2:
                    st.write(f"**Defaults:** {applicant['defaults']:.0f}")
                    st.write(f"**Transaction Frequency:** {applicant['transaction_freq']:.1f}")
                    st.write(f"**Risk Score:** {applicant['risk_score']:.2f}%")
                    st.write(f"**Model Used:** {applicant['model_name']}")
                
                st.divider()
                st.write(f"**Rejection Reason:** {applicant['decision_reason']}")
                
                st.markdown("### 💡 How to Improve & Get Loan Approved")
                
                advice_list = []
                
                if applicant['defaults'] > 0:
                    advice_list.append(f"🔴 **Pay Off Previous Defaults:** You have {applicant['defaults']:.0f} previous default(s). Work with creditors to settle these obligations immediately. This is critical as defaults significantly impact creditworthiness.")
                
                if applicant['repayment_history'] < 60:
                    advice_list.append(f"📉 **Improve Repayment History:** Your repayment history is only {applicant['repayment_history']:.1f}%. Try to maintain 100% on-time payments for next 6-12 months. Set up automatic payments to ensure no missed deadlines.")
                elif applicant['repayment_history'] < 80:
                    advice_list.append(f"📈 **Strengthen Payment Record:** Your repayment history is {applicant['repayment_history']:.1f}%. Work towards 90%+ by making all payments on time. Even a few missed payments can impact your score.")
                
                if applicant['transaction_freq'] < 5:
                    advice_list.append(f"💼 **Increase Financial Activity:** Your transaction frequency is low ({applicant['transaction_freq']:.1f}). Show more regular financial engagement with at least 10+ transactions/month. This demonstrates active account usage.")
                
                if applicant['income'] < 30000:
                    advice_list.append(f"💰 **Increase Income:** Your income (${applicant['income']:,.0f}) is below average. Try to increase earnings through employment growth, side income, or consider a co-guarantor with higher income.")
                
                if applicant['previous_loans'] == 0:
                    advice_list.append("📚 **Build Credit History:** You have no previous loans. Consider starting with a smaller loan or secured loan to build positive credit history.")
                
                for advice in advice_list:
                    st.markdown(f"- {advice}")

    else:
        st.info("✅ No rejected applications! All applicants have been approved.")

# =========================
# FORECASTING (ENHANCED - LEARNS FROM DATABASE)
# =========================
elif menu == "Forecasting":
    st.markdown("# 🔮 Financial Forecasting (AI-Powered)")
    st.divider()

    # Get database statistics
    db_stats = get_database_stats()
    
    if db_stats is None:
        st.error("Unable to connect to database")
        st.stop()
    
    # Display database overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Borrowers", db_stats['borrowers_count'])
    with col2:
        st.metric("📋 Total Loans", db_stats['loans_count'])
    with col3:
        st.metric("📝 Total Predictions", db_stats['predictions_count'])
    with col4:
        if db_stats['loans_count'] > 0:
            approval_rate = (len(db_stats['loans_df'][db_stats['loans_df']['status'] == 'approved']) / db_stats['loans_count'] * 100)
            st.metric("✅ Approval Rate", f"{approval_rate:.1f}%")
        else:
            st.metric("✅ Approval Rate", "0%")

    st.divider()

    st.markdown("### 🎯 Forecast Models")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Default Risk Forecast**\nAnalyzes risk score trends from loans table")
    with col2:
        st.info("**Approval Rate Forecast**\nTracks approved vs rejected decisions")
    with col3:
        st.info("**Loan Amount Forecast**\nPredicts average loan disbursement trends")

    st.divider()

    col1, col2 = st.columns(2)

    # Default Risk Forecast
    with col1:
        try:
            forecast_data = forecast_default_risk(90)
            if forecast_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['historical_dates'],
                    y=forecast_data['historical_values'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_dates'],
                    y=forecast_data['forecast_values'],
                    mode='lines+markers',
                    name='Forecast (90 days)',
                    line=dict(color='#e74c3c', dash='dash', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"📊 Default Risk Forecast (R² = {forecast_data['r_squared']:.3f})",
                    xaxis_title="Date",
                    yaxis_title="Risk Score (%)",
                    template="plotly_dark",
                    height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display insights
                with st.expander("📈 Risk Analysis Insights"):
                    st.write(f"**Data Points Analyzed:** {forecast_data['data_points']} hourly records")
                    st.write(f"**Total Loans Analyzed:** {forecast_data['total_loans_analyzed']} loans")
                    st.write(f"**Current Average Risk:** {forecast_data['current_avg_risk']:.2f}%")
                    st.write(f"**90-Day Forecast Average:** {forecast_data['forecast_avg_risk']:.2f}%")
                    st.write(f"**Trend:** {forecast_data['trend']}")
                    if forecast_data['trend'] == "Increasing":
                        st.warning("⚠️ Risk is increasing - recommend tighter credit policies")
                    else:
                        st.success("✅ Risk is decreasing - positive indicator")
            else:
                st.warning("⚠️ Insufficient Data")
                st.info("Need at least 2 loan records with risk scores to generate forecast.")
                if db_stats['loans_count'] == 0:
                    st.write("📌 **Action:** Go to Integration → Create Loans to generate forecasting data")
        except Exception as e:
            st.error(f"Default risk forecast error: {str(e)}")

    # Approval Rate Forecast
    with col2:
        try:
            forecast_data = forecast_approval_rate(30)
            if forecast_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['historical_dates'],
                    y=forecast_data['historical_values'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_dates'],
                    y=forecast_data['forecast_values'],
                    mode='lines+markers',
                    name='Forecast (30 days)',
                    line=dict(color='#f39c12', dash='dash', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"✅ Approval Rate Forecast (R² = {forecast_data['r_squared']:.3f})",
                    xaxis_title="Date",
                    yaxis_title="Approval Rate (%)",
                    template="plotly_dark",
                    height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display insights
                with st.expander("📊 Approval Analysis Insights"):
                    st.write(f"**Data Points Analyzed:** {forecast_data['data_points']} hourly records")
                    st.write(f"**Total Decisions:** {forecast_data['total_decisions']}")
                    st.write(f"**Approved:** {forecast_data['approved']}")
                    st.write(f"**Rejected:** {forecast_data['rejected']}")
                    st.write(f"**Current Approval Rate:** {forecast_data['current_approval_rate']:.2f}%")
                    st.write(f"**30-Day Forecast Average:** {forecast_data['forecast_approval_rate']:.2f}%")
            else:
                st.warning("⚠️ Insufficient Data")
                st.info("Need at least 2 loan records to generate forecast.")
                if db_stats['loans_count'] == 0:
                    st.write("📌 **Action:** Go to Integration → Create Loans to generate forecasting data")
        except Exception as e:
            st.error(f"Approval rate forecast error: {str(e)}")

    col1, col2 = st.columns(2)

    # Loan Amount Forecast
    with col1:
        try:
            forecast_data = forecast_loan_amount(30)
            if forecast_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['historical_dates'],
                    y=forecast_data['historical_values'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#9b59b6', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast_dates'],
                    y=forecast_data['forecast_values'],
                    mode='lines+markers',
                    name='Forecast (30 days)',
                    line=dict(color='#1abc9c', dash='dash', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"💰 Avg Loan Amount Forecast (R² = {forecast_data['r_squared']:.3f})",
                    xaxis_title="Date",
                    yaxis_title="Average Loan Amount ($)",
                    template="plotly_dark",
                    height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display insights
                with st.expander("💵 Amount Analysis Insights"):
                    st.write(f"**Data Points Analyzed:** {forecast_data['data_points']} hourly records")
                    st.write(f"**Total Approved Loans:** {forecast_data['total_approved_loans']}")
                    st.write(f"**Current Avg Amount:** ${forecast_data['current_avg_amount']:,.2f}")
                    st.write(f"**30-Day Forecast Average:** ${forecast_data['forecast_avg_amount']:,.2f}")
                    st.write(f"**Total Approved Value:** ${forecast_data['total_approved_value']:,.2f}")
                    st.write(f"**Trend:** {forecast_data['trend']}")
            else:
                st.warning("⚠️ Insufficient Data")
                st.info("Need at least 2 approved loan records with amounts to generate forecast.")
                if db_stats['loans_count'] == 0:
                    st.write("📌 **Action:** Go to Integration → Create Loans to generate forecasting data")
        except Exception as e:
            st.error(f"Loan amount forecast error: {str(e)}")

    # Summary Dashboard
    with col2:
        st.markdown("### 📊 System Forecast Summary")
        
        st.markdown("""
        #### 🎯 Forecasting Engine Features:
        - ✅ **Database Learning:** Analyzes borrowers, loans, and prediction logs
        - ✅ **Real-Time Analysis:** Updates as new data is added
        - ✅ **Trend Detection:** Identifies patterns in risk, approval, and amounts
        - ✅ **Predictive Models:** Uses polynomial regression for forecasting
        - ✅ **Quality Metrics:** R² score shows forecast reliability
        
        #### 📌 Data Sources:
        - **Borrowers Table:** Demographic and financial profiles
        - **Loans Table:** Decision history and amounts
        - **Prediction Logs:** Model predictions and outcomes
        
        #### 🔄 How It Works:
        1. System fetches all records from database
        2. Groups data by hour for trend analysis
        3. Applies polynomial regression models
        4. Generates 30-90 day forecasts
        5. Calculates accuracy (R² score)
        6. Displays actionable insights
        """)
        
        st.divider()
        
        st.markdown("### 📌 Data Availability")
        if db_stats['borrowers_count'] == 0:
            st.warning("❌ No borrowers in system")
        else:
            st.success(f"✅ {db_stats['borrowers_count']} borrowers")
        
        if db_stats['loans_count'] == 0:
            st.warning("❌ No loans in system - forecasting requires loan data")
        else:
            st.success(f"✅ {db_stats['loans_count']} loans")
        
        if db_stats['predictions_count'] == 0:
            st.warning("❌ No predictions logged yet")
        else:
            st.success(f"✅ {db_stats['predictions_count']} predictions")

# =========================
# INTEGRATION
# =========================
elif menu == "Integration":

    st.markdown("# 🔗 Integration & Loan Management")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📥 Borrowers", "🚀 Create Loans", "📊 View Loans"])

    with tab1:
        st.markdown("### Add Borrowers")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Single Borrower**")

            name = st.text_input("Name")
            age = st.number_input("Age", 18, 80)
            income = st.number_input("Annual Income ($)", 0, 1000000)

            if st.button("💾 Save Borrower"):
                if name and income > 0:
                    if add_borrower((name, age, income, 80, 0, 0, 5)):
                        st.success("✅ Borrower added!")
                        st.rerun()
                    else:
                        st.error("❌ Error adding borrower")
                else:
                    st.warning("Enter valid name and income")

        with col2:
            st.markdown("**Bulk Upload**")

            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])

            if uploaded_file:
                try:
                    uploaded_file.seek(0)

                    if uploaded_file.name.endswith('.csv'):
                        preview_df = pd.read_csv(uploaded_file)
                    else:
                        preview_df = pd.read_excel(uploaded_file)

                    st.dataframe(preview_df.head(), use_container_width=True)

                except Exception as e:
                    st.error(f"Preview error: {str(e)}")

            if uploaded_file and st.button("📤 Upload"):
                try:
                    uploaded_file.seek(0)

                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    if df.empty or len(df.columns) == 0:
                        st.error("❌ Uploaded file is empty or invalid")
                    else:
                        inserted, errors = bulk_upload_borrowers(df)

                        st.success(f"✅ Imported {inserted} borrowers")

                        if errors:
                            st.warning(f"⚠️ {len(errors)} errors")
                            with st.expander("View errors"):
                                for err in errors[:10]:
                                    st.text(err)

                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("### Auto-Generate Loans")

        borrowers = get_borrowers()
        st.info(f"📊 {len(borrowers)} borrowers ready")

        if st.button("🚀 Auto-Generate Loans for All", use_container_width=True, type="primary"):
            auto_generate_loans()

    with tab3:
        st.markdown("### Loan Decisions")

        decisions = get_loan_decisions()

        if len(decisions) > 0:
            st.dataframe(
                decisions[
                    [
                        'borrower_name',
                        'amount',
                        'duration',
                        'risk_score',
                        'status',
                        'decision_reason'
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No loans yet. Generate loans first!")

# =========================
# SETTINGS (ADMIN ONLY)
# =========================
elif menu == "Settings":
    if role != "admin":
        st.error("❌ Admin access required")
    else:
        st.markdown("# ⚙️ Admin Settings")
        st.divider()
        
        tab1, tab2 = st.tabs(["User Approval", "Risk Configuration"])
        
        with tab1:
            st.markdown("### Pending User Approvals")
            
            users = get_pending_users()
            
            if len(users) > 0:
                st.dataframe(users[['id', 'username', 'role', 'created_at']], use_container_width=True)
                
                uid = st.selectbox("Select User", users["id"])
                if st.button("✅ Approve User"):
                    if approve_user(uid):
                        st.success("User approved!")
                        st.rerun()
            else:
                st.info("No pending users")
        
        with tab2:
            st.markdown("### Risk Thresholds")
            
            col1, col2 = st.columns(2)
            with col1:
                low_threshold = st.number_input("Low Risk (%)", 0, 50, 30)
            with col2:
                high_threshold = st.number_input("High Risk (%)", 50, 100, 60)
            
            st.info(f"🟢 0-{low_threshold}% | 🟡 {low_threshold}-{high_threshold}% | 🔴 {high_threshold}%+")

# =========================
# AUDIT LOGS
# =========================
elif menu == "Audit Logs":
    st.markdown("# 📋 Audit Logs")
    st.divider()
    
    logs = get_prediction_logs(30)
    
    if len(logs) > 0:
        st.dataframe(logs, use_container_width=True, hide_index=True)
        st.info(f"Total predictions: {len(logs)}")
    else:
        st.info("No prediction logs yet")