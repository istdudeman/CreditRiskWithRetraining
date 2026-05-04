import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('/FinalProject/ModelTraining/synthetic_credit_risk_data.csv')
cols_to_drop = [c for c in ['customer_id', 'name_masked', 'loan_id', 'issue_date', 'expected_loss', 'LGD', 'EAD'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

X = df.drop(columns=['missed_payment_flag'])
y = df['missed_payment_flag']

# Cat features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols].fillna('Missing'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
print('Raw XGBoost AUC:', roc_auc_score(y_test, y_pred_proba))
