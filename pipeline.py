import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def ejecutar_pipeline(path_excel, hojas):
    # Leer dataset inicial
    df_base = pd.read_excel(path_excel, sheet_name=hojas[0])
    X_base = df_base[["tiempo_espera", "tono_voz", "quejas"]].astype(float)
    y_base = df_base["satisfaccion"]

    # Modelo inicial
    X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.3, random_state=42)
    modelo_actual = LogisticRegression().fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, modelo_actual.predict(X_test))

    print(f"🔹 Baseline ({hojas[0]}) -> Accuracy: {baseline_acc:.2f}")

    # Iterar sobre las demás hojas
    for hoja in hojas[1:]:
        df_new = pd.read_excel(path_excel, sheet_name=hoja)
        X_new = df_new[["tiempo_espera", "tono_voz", "quejas"]].astype(float)
        y_new = df_new["satisfaccion"]

        # Unir datasets
        X_total = pd.concat([X_base, X_new])
        y_total = pd.concat([y_base, y_new])

        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=42)
        modelo_nuevo = LogisticRegression().fit(X_train, y_train)
        new_acc = accuracy_score(y_test, modelo_nuevo.predict(X_test))

        # Log en MLflow
        mlflow.set_experiment("/Users/<TU-EMAIL>/mlops-callcenter1")
        with mlflow.start_run():
            mlflow.log_param("dataset", hoja)
            mlflow.log_metric("baseline_accuracy", baseline_acc)
            mlflow.log_metric("new_accuracy", new_acc)

            if new_acc > baseline_acc:
                mlflow.sklearn.log_model(modelo_nuevo, "modelo_mejorado")
                print(f"✅ Nuevo modelo publicado con {hoja}, accuracy={new_acc:.2f}")
                modelo_actual = modelo_nuevo
                baseline_acc = new_acc
            else:
                print(f"❌ Se mantiene modelo anterior (no mejora con {hoja})")
