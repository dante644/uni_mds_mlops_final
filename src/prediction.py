import requests
import json
import os

# Configuración de la URL y los headers
url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

# Estructura de los datos
payload = {
    "dataframe_split": {
        "columns": [
            "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", 
            "PAY_5", "PAY_6", "BILL_AMT2", "BILL_AMT5", "PAY_AMT1", 
            "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ],
        "data": [
            [20000, 24, 2, 2, -1, -1, -2, -2, 3102, 0, 0, 689, 0, 0, 0, 0]
        ]
    }
}

# Definir la ruta de salida
# Esto asegura que se guarde en 'reports' respecto a la raíz de tu proyecto
output_path = os.path.join("reports", "prediction_result.json")

try:
    # Realizar la petición POST
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Verificar el resultado
    if response.status_code == 200:
        result = response.json()
        print("✅ Éxito: Guardando resultado en la carpeta reports...")
        
        # Guardar el JSON en la carpeta reports
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"📄 Resultado guardado en: {output_path}")
    else:
        error_msg = f"Error {response.status_code}: {response.text}"
        print(f"❌ {error_msg}")
        
        # Opcional: Guardar también el error para registro
        with open(os.path.join("reports", "error_log.txt"), "w") as f:
            f.write(error_msg)

except requests.exceptions.ConnectionError:
    print("❌ Error: No se pudo conectar al servidor. ¿Está encendido el modelo en el puerto 5001?")