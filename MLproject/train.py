import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
import os

# 1. Kunci koneksi ke tracking server lokal (Port 5000)
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Iris_Production")

# 2. Muat dataset Iris standar
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚡ Memulai pelatihan model RandomForest...")
with mlflow.start_run() as run:
    # 3. Latih model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Ambil skema input secara otomatis (Mendeklarasikan Kontrak Kolom)
    # Ini menjawab pertanyaanmu tadi: kapan dan di mana kolom dideklarasikan!
    input_schema = [
        "sepal length (cm)", 
        "sepal width (cm)", 
        "petal length (cm)", 
        "petal width (cm)"
    ]
    # Konversi data sampel ke format yang melampirkan nama kolom asli
    import pandas as pd
    X_sample = pd.DataFrame(X_train[:1], columns=input_schema)
    signature = infer_signature(X_sample, clf.predict(X_sample))
    
    # 5. Simpan model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="iris_model",
        signature=signature
    )
    
    print("\n=======================================================")
    print("✅ MODEL BERHASIL DILATIH & DICATAT DI MLFLOW!")
    print(f"🆔 RUN ID KAMU: {run.info.run_id}")
    print("=======================================================")