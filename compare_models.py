import pandas as pd
import numpy as np
import random
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#գեներացնել սինթետիկ դատա
n_samples = 100000
loan_purposes = ['hypothec','consumer','agriculture','business']
marital_statuses = ['single','married','divorced']
region_codes = ['Aragatsotn','Ararat','Armavir','Gegharkunik','Kotayk',
                'Lori','Shirak','Syunik','Tavush','Vayots Dzor','Yerevan']

data=[]
for _ in range(n_samples):
    if random.random() < 0.70:
        credit_score = random.randint(650,850)
        monthly_income = round(random.uniform(500000,1000000),2)
        monthly_expenses = round(random.uniform(1000, monthly_income*0.4),2)
        late_payments = random.randint(0,1)
    else:
        credit_score = random.randint(300,599)
        monthly_income = round(random.uniform(60000,300000),2)
        monthly_expenses = round(random.uniform(monthly_income*0.8, monthly_income*1.5),2)
        late_payments = random.randint(2,20)

    row={
        'age': random.randint(18,70),
        'employment_years': round(random.uniform(0,60),1),
        'monthly_income': monthly_income,
        'credit_score': credit_score,
        'loan_amount': round(random.uniform(50000,77000000),2),
        'loan_term_months': random.randint(1,36),
        'monthly_expenses': monthly_expenses,
        'other_loans_total': round(random.uniform(0,1000000),2),
        'is_guaranteed': random.choice([0,1]),
        'loan_purpose': random.choice(loan_purposes),
        'marital_status': random.choice(marital_statuses),
        'region_code': random.choice(region_codes),
        'late_payments_last_1m': late_payments
    }
    data.append(row)

df = pd.DataFrame(data)

def loan_approval_rule(row):
    if row['credit_score'] < 450:
        return 0
    elif row['credit_score'] < 600 and row['late_payments_last_1m'] > 2:
        return 0
    elif row['monthly_income'] < row['monthly_expenses'] * 1.2:
        return 0
    elif row['loan_amount'] > row['monthly_income'] * 15 and row['employment_years'] < 2:
        return 0
    elif row['loan_purpose'] == 2 and row['credit_score'] < 650: 
        return 0
    else:
        return 0 if np.random.rand() < 0.1 else 1

df['target'] = df.apply(loan_approval_rule, axis=1)

categorical_cols = ['loan_purpose','marital_status','region_code']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
joblib.dump(le_dict, 'le_dict.pkl')

#վերջնական արդյունք
print(df['target'].value_counts())
df.to_csv('synthetic_loan_data.csv', index=False)
print("Dataset saved as 'synthetic_loan_data.csv'")

X = df.drop('target', axis=1)
y = df['target']  
#վարկի բազա
df = pd.read_csv('synthetic_loan_data.csv')

#Features / Target
X = df.drop('target', axis=1)
y = df['target']

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:,1]
threshold = 0.4 
y_pred_rf = (y_proba_rf >= threshold).astype(int)

print("=== Random Forest Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

#Isolation Forest
iso_model = IsolationForest(contamination=0.05, random_state=42) 
iso_model.fit(X_train)
y_pred_iso = iso_model.predict(X_test)
y_pred_iso_mapped = np.where(y_pred_iso==1, 1, 0)
y_scores_iso = iso_model.decision_function(X_test)

print("\n=== Isolation Forest Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_iso_mapped))
print("Precision:", precision_score(y_test, y_pred_iso_mapped))
print("Recall:", recall_score(y_test, y_pred_iso_mapped))
print("F1 Score:", f1_score(y_test, y_pred_iso_mapped))
print("ROC AUC:", roc_auc_score(y_test, y_scores_iso))



# ---------------- Logistic Regression ----------------
log_reg = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:,1]

print("\n=== Logistic Regression Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_proba_log))


# ---------------- Gradient Boosting ----------------
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)[:,1]

print("\n=== Gradient Boosting Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Precision:", precision_score(y_test, y_pred_gb))
print("Recall:", recall_score(y_test, y_pred_gb))
print("F1 Score:", f1_score(y_test, y_pred_gb))
print("ROC AUC:", roc_auc_score(y_test, y_proba_gb))

# ---------------- K-Nearest Neighbors ----------------
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_proba_knn = knn_model.predict_proba(X_test)[:,1]

print("\n=== KNN Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print("F1 Score:", f1_score(y_test, y_pred_knn))
print("ROC AUC:", roc_auc_score(y_test, y_proba_knn))

# --- ROC Curve Comparison with All Models ---
# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# Isolation Forest
fpr_iso, tpr_iso, _ = roc_curve(y_test, y_scores_iso)

# Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)

# Gradient Boosting
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
# KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

# --- Plot All ROC Curves ---
plt.figure(figsize=(12,12))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC={:.3f})".format(roc_auc_score(y_test, y_proba_rf)))
plt.plot(fpr_iso, tpr_iso, label="Isolation Forest (AUC={:.3f})".format(roc_auc_score(y_test, y_scores_iso)))
plt.plot(fpr_log, tpr_log, label="Logistic Regression (AUC={:.3f})".format(roc_auc_score(y_test, y_proba_log)))
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting (AUC={:.3f})".format(roc_auc_score(y_test, y_proba_gb)))
plt.plot(fpr_knn, tpr_knn, label="KNN (AUC={:.3f})".format(roc_auc_score(y_test, y_proba_knn)))

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Loan Approval Models")
plt.legend()
plt.grid()
plt.show()

# --- Random Forest Confusion Matrix ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0,1])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# --- Isolation Forest Confusion Matrix ---
cm_iso = confusion_matrix(y_test, y_pred_iso_mapped)
disp_iso = ConfusionMatrixDisplay(confusion_matrix=cm_iso, display_labels=[0,1])
disp_iso.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Isolation Forest")
plt.show()

# --- Logistic Regression Confusion Matrix ---
cm_log = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=[0,1])
disp_log.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# --- Gradient Boosting Confusion Matrix ---
cm_gb = confusion_matrix(y_test, y_pred_gb)
disp_gb = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=[0,1])
disp_gb.plot(cmap=plt.cm.Purples)
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()

# --- KNN Confusion Matrix ---
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=[0,1])
disp_knn.plot(cmap=plt.cm.Greys)
plt.title("Confusion Matrix - KNN")
plt.show()

def print_confusion_words(cm, model_name):
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== {model_name} Confusion Matrix ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Total: {tn+fp+fn+tp}")

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
print_confusion_words(cm_rf, "Random Forest")

# Isolation Forest
cm_iso = confusion_matrix(y_test, y_pred_iso_mapped)
print_confusion_words(cm_iso, "Isolation Forest")

# Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
print_confusion_words(cm_log, "Logistic Regression")

# Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
print_confusion_words(cm_gb, "Gradient Boosting")

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
print_confusion_words(cm_knn, "KNN")


# ----------------- Save Random Forest model -----------------
joblib.dump(rf_model, 'rf_loan_model.pkl')
joblib.dump(iso_model, 'iso_loan_model.pkl')
joblib.dump(log_reg, "log_reg_loan_model.pkl")
joblib.dump(gb_model, "gb_loan_model.pkl")
joblib.dump(knn_model, "knn_loan_model.pkl")
print("Extra models saved")