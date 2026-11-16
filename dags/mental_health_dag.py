from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# --- Tâche 1 : Chargement et nettoyage des données ---
def prepare_data():
    df = pd.read_csv("/opt/airflow/data/student_mental_health.csv")
    df = df.drop_duplicates()
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])
    df.to_pickle("/opt/airflow/data/clean_data.pkl")
    print(f"Données préparées : {df.shape[0]} lignes sauvegardées.")

# --- Tâche 2 : Prétraitement et entraînement du modèle ---
def train_model():
    df = pd.read_pickle("/opt/airflow/data/clean_data.pkl")
    target_col = "Mental_Health_Risk"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encodage de la cible si nécessaire
    if y.dtype == "object":
        y = y.map({"Low": 0, "Medium": 1, "High": 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(C=100, solver="lbfgs", max_iter=1000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    joblib.dump((pipe, X_test, y_test), "/opt/airflow/models/temp_pipeline.joblib")
    print("Modèle Logistic Regression entraîné avec succès.")

# --- Tâche 3 : Évaluation qualité et export final ---
def evaluate_and_save():
    pipe, X_test, y_test = joblib.load("/opt/airflow/models/temp_pipeline.joblib")
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"F1-Score sur le jeu de test : {f1:.4f}")
    if f1 >= 0.70:
        joblib.dump(pipe, "/opt/airflow/models/mental_health_pipeline_airflow.joblib")
        print("Validation réussie : pipeline sauvegardé en production !")
    else:
        raise ValueError(f"Performance insuffisante (F1={f1:.4f} < 0.70). Déploiement annulé.")

# --- Définition du DAG ---
default_args = {
    "owner": "safiy_youcode",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="student_mental_health_pipeline",
    default_args=default_args,
    description="Pipeline ML automatisé de prédiction du risque de santé mentale étudiant",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["machine_learning", "mental_health"],
) as dag:

    t1 = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model)
    t3 = PythonOperator(task_id="evaluate_and_save", python_callable=evaluate_and_save)

    t1 >> t2 >> t3