import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Santé Mentale Étudiante",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/mental_health_pipeline_airflow.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("Data/student_mental_health.csv")

model = load_model()
df = load_data()

# En-tête
st.title("🧠 MLOps — Prédiction du Risque de Santé Mentale Étudiante")
st.markdown(
    "Pipeline automatisé avec **Apache Airflow** & interface interactive **Streamlit**."
)

# Navigation par onglets
tab1, tab2 = st.tabs(["🔮 Prédiction sur nouvelles données", "📈 Performances du modèle"])

labels_fr = {0: "Faible (Low)", 1: "Modéré (Medium)", 2: "Élevé (High)"}

# ==============================================================================
# ONGLET 1 : PRÉDICTION INTERACTIVE
# ==============================================================================
with tab1:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Profil & Académique")
            age = st.number_input("Âge (Age)", min_value=16, max_value=60, value=21)
            gender = st.selectbox("Genre (Gender)", sorted(df["Gender"].dropna().unique()))
            year = st.number_input("Année d'étude (Year_of_Study)", min_value=1, max_value=8, value=2)
            gpa = st.slider("Moyenne (GPA)", 0.0, 4.0, 3.0, step=0.1)
            attendance = st.slider("Taux de présence % (Attendance_Rate)", 0.0, 100.0, 85.0, step=1.0)

        with col2:
            st.subheader("⏱️ Habitudes & Écrans")
            sleep = st.slider("Heures de sommeil / nuit (Sleep_Hours)", 2.0, 14.0, 7.0, step=0.5)
            study_hours = st.slider("Heures d'étude / semaine (Study_Hours_Per_Week)", 0.0, 80.0, 20.0, step=1.0)
            social_media = st.slider("Réseaux sociaux h/jour (Social_Media_Hours)", 0.0, 15.0, 3.0, step=0.5)
            screen_time = st.slider("Temps d'écran total h/jour (Screen_Time_Hours)", 0.0, 18.0, 6.0, step=0.5)
            physical_activity = st.selectbox("Activité physique (Physical_Activity)", sorted(df["Physical_Activity"].dropna().unique()))

        with col3:
            st.subheader("🏡 Environnement & Santé")
            diet_quality = st.selectbox("Qualité de l'alimentation (Diet_Quality)", sorted(df["Diet_Quality"].dropna().unique()))
            part_time_job = st.selectbox("Travail à temps partiel (Part_Time_Job)", sorted(df["Part_Time_Job"].dropna().unique()))
            financial_stress = st.slider("Stress financier (Financial_Stress)", 1.0, 10.0, 5.0, step=1.0)
            family_support = st.slider("Soutien familial (Family_Support)", 1.0, 10.0, 7.0, step=1.0)
            prev_history = st.selectbox(
                "Antécédents de santé mentale (Previous_Mental_Health_History)",
                sorted(df["Previous_Mental_Health_History"].dropna().unique())
            )

        submitted = st.form_submit_button("🔍 Analyser le risque", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Age": int(age),
            "Gender": gender,
            "Year_of_Study": int(year),
            "Sleep_Hours": float(sleep),
            "Study_Hours_Per_Week": float(study_hours),
            "GPA": float(gpa),
            "Social_Media_Hours": float(social_media),
            "Screen_Time_Hours": float(screen_time),
            "Physical_Activity": physical_activity,
            "Financial_Stress": float(financial_stress),
            "Family_Support": float(family_support),
            "Attendance_Rate": float(attendance),
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": prev_history,
        }])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        pred_label = labels_fr.get(prediction, str(prediction))

        st.divider()
        st.subheader("📊 Résultat de l'analyse")

        if prediction == 0:
            st.success(f"**Niveau de risque prédit : {pred_label}** — Profil globalement équilibré.")
        elif prediction == 1:
            st.warning(f"**Niveau de risque prédit : {pred_label}** — Vigilance recommandée sur le sommeil et le stress.")
        else:
            st.error(f"**Niveau de risque prédit : {pred_label}** — Accompagnement et suivi fortement conseillés.")

        st.markdown("**Probabilités estimées par le modèle :**")
        prob_df = pd.DataFrame({
            "Niveau de risque": [labels_fr[int(c)] for c in model.classes_],
            "Probabilité (%)": [round(p * 100, 2) for p in probabilities]
        })
        st.bar_chart(prob_df.set_index("Niveau de risque"))

# ==============================================================================
# ONGLET 2 : PERFORMANCES DU MODÈLE (MÉTRIQUES, GRAPHIQUES & MATRICE)
# ==============================================================================
with tab2:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report

    st.subheader("🎯 Métriques globales (Jeu de test 20 %)")

    df_eval = df.drop(columns=["Student_ID"], errors="ignore").dropna()
    target_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df_eval["Mental_Health_Risk"].map(target_map).astype(int)
    X = df_eval.drop(columns=["Mental_Health_Risk"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision (Weighted)", f"{prec:.2%}")
    m3.metric("Recall (Weighted)", f"{rec:.2%}")
    m4.metric("F1-Score (Weighted)", f"{f1:.2%}")

    st.divider()
    g1, g2 = st.columns(2)
    class_names_fr = ["Faible (Low)", "Modéré (Medium)", "Élevé (High)"]

    # 1. Heatmap de la Matrice de Confusion
    with g1:
        st.subheader("🧩 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Faible", "Modéré", "Élevé"],
            yticklabels=["Faible", "Modéré", "Élevé"],
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax_cm
        )
        ax_cm.set_xlabel("Classe Prédite", fontsize=11, fontweight="bold")
        ax_cm.set_ylabel("Classe Réelle", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)

    # 2. Graphique comparatif côte à côte (Réel vs Prédit)
    with g2:
        st.subheader("📊 Comparaison Réel vs Prédit (Test)")
        comp_df = pd.DataFrame({
            "Niveau de risque": ["Faible", "Modéré", "Élevé"] * 2,
            "Nombre d'étudiants": [int((y_test == i).sum()) for i in [0, 1, 2]] +
                                  [int((y_pred == i).sum()) for i in [0, 1, 2]],
            "Type": ["Valeurs Réelles"] * 3 + ["Prédictions Modèle"] * 3
        })
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
        sns.barplot(
            data=comp_df,
            x="Niveau de risque",
            y="Nombre d'étudiants",
            hue="Type",
            palette=["#4C78A8", "#F58518"],
            ax=ax_bar
        )
        for container in ax_bar.containers:
            ax_bar.bar_label(container, fontsize=10, padding=3)
        ax_bar.set_ylim(0, max(comp_df["Nombre d'étudiants"]) + 25)
        ax_bar.set_xlabel("")
        ax_bar.legend(title="")
        plt.tight_layout()
        st.pyplot(fig_bar)

    st.divider()

    # 3. Détail des scores par classe (Precision, Recall, F1-Score)
    st.subheader("📋 Détail des performances par classe")
    report = classification_report(y_test, y_pred, target_names=class_names_fr, output_dict=True)
    report_df = pd.DataFrame(report).T.iloc[:3, :3] * 100
    report_df.columns = ["Precision (%)", "Recall (%)", "F1-Score (%)"]
    st.dataframe(report_df.style.format("{:.2f} %"), use_container_width=True)