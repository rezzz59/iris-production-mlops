import requests
import json

# URL mengarah ke port luar Docker yang baru saja kita run (5004)
url = "http://127.0.0.1:5004/invocations"
headers = {"Content-Type": "application/json"}

# Contoh data sampel (4 fitur: sepal length, sepal width, petal length, petal width)
data = {
    "dataframe_split": {
        "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Sampel data 1
            [6.7, 3.0, 5.2, 2.3], # Sampel data 2
            [2.7, 3.5, 105.2, 1.3]    
        ]
    }
}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print("========================================")
    print("🤖 RESPONS PREDIKSI DARI DOCKER KONTANER:")
    print(f"Status Code: {response.status_code}")
    print(f"Hasil Prediksi Kelas: {response.json()}")
    print("========================================")
except Exception as e:
    print(f"Gagal terhubung ke API: {e}")