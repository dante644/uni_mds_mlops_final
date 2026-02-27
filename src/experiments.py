import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Modelos
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Configuración de rutas
DATA_PATH = "data/training/processed_credit_card_data.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Configurar el experimento
mlflow.set_experiment("Credit_Card_Classification_Project")

def train_and_evaluate():
    # 2. Carga y Limpieza de datos
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(exclude=['object']) 
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics_list = []

    # 3. Definición de modelos
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes": GaussianNB()
    }

    # 4. Entrenamiento y Registro detallado en MLflow
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f">>> Entrenando y Logueando métricas: {name}")
            
            model.fit(X_train, y_train)
            
            # Persistencia local (.pkl)
            model_path = os.path.join(MODEL_DIR, f"{name.lower()}.pkl")
            joblib.dump(model, model_path)
            
            y_pred = model.predict(X_test)
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # --- REGISTRO EN MLFLOW ---
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Matriz de Confusión
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            temp_plot_path = os.path.join(REPORT_DIR, f"cm_{name}.png")
            plt.savefig(temp_plot_path)
            mlflow.log_artifact(temp_plot_path)
            plt.close()

            # --- REGISTRO DEL MODELO PARA SERVING ---
            # La firma (signature) es CRUCIAL para que el servidor sepa qué datos recibe
            signature = infer_signature(X_test, y_pred)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=name,
                signature=signature
            )
            
            # Guardamos la Run ID para facilitar el despliegue posterior
            run_id = run.info.run_id
            metrics_list.append({
                "Modelo": name, 
                "F1-Score": f1, 
                "Accuracy": acc, 
                "Run_ID": run_id
            })

    # 5. Reporte Comparativo Final
    df_metrics = pd.DataFrame(metrics_list).sort_values(by="F1-Score", ascending=False)
    df_metrics.to_csv(os.path.join(REPORT_DIR, "final_metrics.csv"), index=False)
    
    print("\n" + "="*60)
    print("PROCESO FINALIZADO EXITOSAMENTE")
    print("Módulos listos para ser servidos:")
    print(df_metrics[["Modelo", "Run_ID"]])
    print("\nPara servir un modelo, copia su Run_ID y ejecuta en tu terminal:")
    print("mlflow models serve -m runs:/<RUN_ID>/model --port 5001")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate()