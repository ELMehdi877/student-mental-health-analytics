# 🧠 Prédiction du Risque de Santé Mentale Étudiante (Pipeline MLOps)

Ce projet met en place une solution complète de **Machine Learning et MLOps** permettant d'analyser et de prédire le niveau de risque de santé mentale chez les étudiants (`Low`, `Medium`, `High`). 

L'entraînement du modèle est entièrement automatisé et orchestré avec **Apache Airflow** (via Docker), et le modèle est déployé dans une interface interactive développée avec **Streamlit**.

---

## 📋 Table des matières
- [Architecture du Projet](#-architecture-du-projet)
- [Structure du Dépôt](#-structure-du-dépôt)
- [Performances du Modèle](#-performances-du-modèle)
- [Installation et Lancement](#-installation-et-lancement)
  - [1. Orchestration avec Apache Airflow](#1-orchestration-avec-apache-airflow)
  - [2. Application Web Streamlit](#2-application-web-streamlit)
- [Technologies Utilisées](#-technologies-utilisées)

---

## 🏗️ Architecture du Projet

1. **Données (`Data/student_mental_health.csv`)** : Jeu de données contenant 16 variables socio-académiques, comportementales et environnementales (sommeil, GPA, stress financier, temps d'écran, soutien familial, etc.) avec la cible `Mental_Health_Risk`.
2. **Pipeline ML & Orchestration (`Apache Airflow`)** :
   - **DAG** : `student_mental_health_pipeline` (planifié automatiquement chaque semaine via `@weekly`).
   - **Tâche 1 (`prepare_data`)** : Nettoyage des données, gestion des valeurs manquantes et encodage de la cible (`Low`: 0, `Medium`: 1, `High`: 2).
   - **Tâche 2 (`train_model`)** : Prétraitement (`StandardScaler` + `OneHotEncoder`) et entraînement d'un modèle de **Régression Logistique**.
   - **Tâche 3 (`evaluate_and_save`)** : Évaluation sur le jeu de test (20 %) et sauvegarde du pipeline complet au format `.joblib`.
3. **Déploiement (`Streamlit`)** :
   - **Onglet Prédiction** : Formulaire interactif permettant d'estimer en temps réel le risque d'un étudiant avec ses probabilités associées.
   - **Onglet Performances** : Tableau de bord affichant les métriques globales, la matrice de confusion et la distribution comparative *Réel vs Prédit*.

---

## 📁 Structure du Dépôt

```text
student-mental-health-analytics/
├── dags/
│   └── mental_health_dag.py                  # DAG Apache Airflow (ETL, Entraînement, Évaluation)
├── Data/
│   └── student_mental_health.csv             # Dataset source
├── models/
│   └── mental_health_pipeline_airflow.joblib # Modèle sérialisé généré par Airflow
├── app.py                                    # Application interactive Streamlit
├── docker-compose.yaml                       # Configuration Docker pour Apache Airflow
├── requirements.txt                          # Dépendances Python
└── README.md                                 # Documentation du projet
```

---

## 📊 Performances du Modèle

Évalué sur le jeu de test (20 % des données, soit 322 étudiants), le modèle obtient les résultats suivants :

| Métrique | Score | Interprétation |
| :--- | :---: | :--- |
| **Accuracy (Exactitude)** | **76.71 %** | Taux de réussite global sur l'ensemble des prédictions. |
| **Precision (Weighted)** | **77.51 %** | Fiabilité moyenne des prédictions sur les 3 classes. |
| **Recall (Weighted)** | **76.71 %** | Capacité globale à retrouver les vrais cas de chaque classe. |
| **F1-Score (Weighted)** | **76.98 %** | Équilibre global entre Precision et Recall. |
| **Recall (Classe `High`)** | **83.04 %** | Détection de **93 étudiants sur 112** réellement en situation critique. |

---

## 🚀 Installation et Lancement

### Prérequis
- **Python 3.10+**
- **Docker** et **Docker Compose** (pour exécuter Apache Airflow)

---

### 1. Orchestration avec Apache Airflow

1. **Démarrer les conteneurs Airflow :**
   ```bash
   docker compose up -d
   ```
2. **Accéder à l'interface Web Airflow :**
   - Ouvrez votre navigateur sur [http://localhost:8080](http://localhost:8080)
   - Identifiants par défaut : `airflow` / `airflow`
3. **Exécuter le pipeline :**
   - Activez le DAG **`student_mental_health_pipeline`** et déclenchez-le manuellement (*Trigger DAG*) pour générer ou mettre à jour le fichier `models/mental_health_pipeline_airflow.joblib`.

---

### 2. Application Web Streamlit

1. **Installer les dépendances Python :**
   ```bash
   pip install streamlit pandas scikit-learn joblib matplotlib seaborn dill
   ```
2. **Lancer l'application Streamlit :**
   ```bash
   streamlit run app.py
   ```
   *(Sous Windows, si la commande directe n'est pas reconnue : `py -m streamlit run app.py`)*
3. **Accéder à l'application :**
   - Ouvrez votre navigateur sur [http://localhost:8501](http://localhost:8501).

---

## 🛠️ Technologies Utilisées

- **Langage :** Python
- **Machine Learning & Data :** Scikit-Learn, Pandas, NumPy, Joblib
- **Orchestration MLOps :** Apache Airflow, Docker & Docker Compose
- **Visualisation & Déploiement :** Streamlit, Matplotlib, Seaborn