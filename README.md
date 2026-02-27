# Modelo Avanzado de Credit Scoring - Banco de Taiwán (BOT)



## 1. Definición del Caso de Uso (IA/ML/MLOps)
**Problema Específico:** Predicción de la probabilidad de incumplimiento de pago (*default*) en clientes de tarjetas de crédito para optimizar la toma de decisiones financieras.

El Banco de Taiwán busca transformar su gestión de riesgos mediante un modelo de **Machine Learning de Clasificación Binaria**. El objetivo es identificar patrones de comportamiento en el historial crediticio y factores demográficos que preceden a un impago, permitiendo a la institución actuar de manera preventiva.

---

## 2. Descripción del Proyecto

### Contexto del Problema
El sector financiero en Taiwán requiere herramientas de alta precisión para competir en un mercado volátil. Contamos con un conjunto de datos histórico (abril 2005 - septiembre 2005) que detalla:
* **Factores Demográficos:** Género, educación, estado civil y edad.
* **Historial de Crédito:** Monto del crédito otorgado.
* **Historial de Pagos:** Registros de puntualidad en los últimos 6 meses.
* **Estado de Cuenta:** Montos de facturación y pagos realizados.



### Limitaciones
* **Ventana Temporal:** Los datos representan un periodo específico de 2005; el modelo debe ser validado frente a cambios en el comportamiento del consumidor moderno.
* **Desbalance de Datos:** Generalmente, los casos de incumplimiento son significativamente menores a los de cumplimiento, lo que requiere técnicas de remuestreo (SMOTE) o ajuste de pesos en el algoritmo.
* **Interpretabilidad:** Debido a regulaciones bancarias, el modelo no solo debe ser preciso, sino también explicable (identificar por qué se rechaza un crédito).

### Objetivos
* **Desarrollar** un modelo predictivo robusto basado en algoritmos avanzados (ej. XGBoost, Random Forest o Redes Neuronales).
* **Mejorar** la precisión en la evaluación del riesgo crediticio en comparación con métodos estadísticos tradicionales.
* **Fortalecer** la competitividad del Banco de Taiwán en el mercado mediante una gestión de cartera más inteligente.

### Beneficios y Resultados Esperados
1.  **Reducción de la Tasa de Incumplimiento:** Disminución directa de las pérdidas por cuentas incobrables.
2.  **Optimización de la Cartera:** Capacidad de ofrecer mejores condiciones a clientes de bajo riesgo, aumentando la retención.
3.  **Automatización:** Reducción de tiempos en el análisis manual de solicitudes de crédito.
4.  **Dashboard de Riesgo:** Visualización clara de las variables que más impactan en el riesgo crediticio actual.

---

## 3. Métricas de Éxito (KPIs de ML)
Para que la solución sea considerada exitosa, el modelo debe alcanzar o superar los siguientes umbrales:

| Métrica | Objetivo | Justificación |
| :--- | :--- | :--- |
| **AUC-ROC** | $> 0.78$ | Capacidad del modelo para distinguir entre un cliente que pagará y uno que no. |
| **F1-Score** | Balanceado | Asegura un equilibrio entre precisión y sensibilidad ante datos desbalanceados. |

---

## 4. Tecnologías Sugeridas
* **Lenguaje:** Python 3.x
* **Librerías:** Pandas, Scikit-learn, XGBoost/LightGBM, Matplotlib/Seaborn para visualización.
* **Entorno:** Visual Code / Git Hub.

---


## 5. Adquisición de datos

| Variable | Descripción |
| :--- | :--- | 
| ID | ID del cliente (índice)  |
| LIMIT_BAL | Monto del crédito otorgado (NT dollar)  |
| SEX | Género (masculino; femenino)|
| EDUCATION | Nivel educativo (escuela de posgrado; universidad; escuela secundaria; otros)   |
| MARRIAGE | Estado civil (casado; soltero; otros)  |
| AGE | Edad en años. |
| PAY_0 | Estado de pago en septiembre de 2005 (-2 = sin consumo; -1 = pago puntual; 0 = uso de crédito renovable; 1 = retraso en el pago de un mes; 2 = retraso en el pago de dos meses; y así sucesivamente)  |
| PAY_2 | Estado de pago en agosto de 2005 (misma escala que la anterior)  |
| PAY_3 | Estado de pago en julio de 2005 (misma escala que la anterior) |
| PAY_4 | Estado de pago en junio de 2005 (misma escala que la anterior)  |
| PAY_5 | Estado de pago en mayo de 2005 (misma escala que la anterior) arriba)  |
| PAY_6 | Estado de pago en abril de 2005 (misma escala que arriba)  |
| BILL_AMT1 | Importe del estado de cuenta en septiembre de 2005 (dólar NT)  |
| BILL_AMT2 | Importe del estado de cuenta en agosto de 2005 (dólar NT)  |
| BILL_AMT3 | Importe del estado de cuenta en julio de 2005 (dólar NT)  |
| BILL_AMT4 | Importe del estado de cuenta en junio de 2005 (dólar NT) |
| BILL_AMT5 | Importe del estado de cuenta en mayo de 2005 (dólar NT)  |
| BILL_AMT6 | Importe del estado de cuenta en abril de 2005 (dólar NT)  |
| PAY_AMT1 | Importe pagado en septiembre de 2005 (dólar NT)  |
| PAY_AMT2 | Importe pagado en agosto de 2005 (dólar NT)  |
| PAY_AMT3 | Importe pagado en julio de 2005 (dólar NT)  |
| PAY_AMT4 | Monto pagado en junio de 2005 (dólar NT)  |
| PAY_AMT5 | Monto pagado en mayo de 2005 (dólar NT)  |
| PAY_AMT6 | Monto pagado en abril de 2005 (dólar NT)  |
| default_payment_next_month | Pago predeterminado (1 = sí; 0 = no) |

---


## 6. Estructura del Proyecto

```
Credit Scoring/
├── data/
│   ├── raw/                    # Dataset original
│   │   └── cleaned_credit_card_data.csv
│   └── training/               # Datos procesados
│        └── processed_credit_card_data.csv
├── imagen/                     # Imagenes de analisis 
│   ├── boxplot_variables.png
│   ├── correlacion_final.png
│   └── correlacion_inicial.png
├── mlruns/                     # Run ID del MLflow
├── models/                     # Modelos entrenados
│   ├── adaboost.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── reports/                     # Resultados
│   ├── cm_AdaBoost.png
│   ├── cm_Gradient_Boosting.png
│   ├── cm_KNN.png
│   ├── cm_Logistic_Regression.png
│   ├── cm_Naive_Bayes.png
│   ├── cm_Random_Forest.png
│   ├── cm_SVM.png
│   ├── final_metrics.csv
│   └── prediction_result.json
├── src/                        # Código modular
│   ├── data_preparation.py
│   ├── experiments.py
│   └── prediction.py
├── requirements.txt
├── README.md
├── terminal.txt                # Ejecuciones que se hicieron
└── .gitignore
```

---

## 7. Inicio Rápido

### Instalación de librerias

```bash
# Ejecutar en el terminal de Powershell
pip install -r requirements.txt

```

### Ejecutar los scripts

```bash
# Ejecutar en el terminal de Powershell

# 1ro (Tratamiento de dato)
py src/data_preparation.py

# 2do (Modelamiento - Implementación - Evaluación)
py src/experiments.py
```

```bash
# Ejecutar en el terminal de Powershell - dejarlo abierto
mlflow ui

```

Abres para ver el entorno de MLflow: **http://127.0.0.1:5000**

---

## 8. Métricas

El mejor modelo es Gradient_Boosting (Evaluación)

| Modelo | F1-Score | Accuracy | Run_ID |
|--------|----------|----------|--------|
| **Gradient_Boosting** | **0.7995453627211427** | **0.8208333333333333** | **62d737358b96407b82b8bdd76aef4b98** |
| Random_Forest | 0.7949377236570891 | 0.8141666666666667 | 668f238435ab44d9a08cd0f03b45f1dd |
| AdaBoost | 0.7933341293532338 | 0.8186666666666667 | b93fe74931344e919068fb1d7e6207b2 |
| KNN | 0.7174260659393438 | 0.7545 | b71a745034dd47dea49d88d56f585429 |
| Logistic_Regression | 0.6859041959551291 | 0.7813333333333333 | 201938fa783a4a0ca160556983ca7a99 |
| SVM | 0.6851928823180812 | 0.7811666666666667 | fabf51ecb11d47abb1774c6ea0dece7d |
| Naive_Bayes | 0.6793083546301553 | 0.6548333333333334 | 5762cccf20c644d69b9eb06e547919d7 |



---

## 9. API REST 

```bash
# En el terminal del Git Bash - dejarlo abierto
mlflow ui --port 5000
```

```bash
# En otro terminal del Git Bash - dejarlo abierto
mlflow models serve -m runs:/668f238435ab44d9a08cd0f03b45f1dd/model --port 5001 --no-conda
```

Abre: **http://127.0.0.1:5001**

```bash
# Ejecutar en otro terminal de Powershell
py src/prediction.py

```
**O**

```bash
# En el terminal del Git Bash
curl -X POST -H "Content-Type: application/json" \
  --data '{
    "dataframe_split": {
      "columns": [
      "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT2", "BILL_AMT5", "PAY_AMT1", "PAY AMT2", "PAY_AMT3", "PAY AMT4", "PAY_ AMT5", "PAY AMT6"
      ],
    "data": [
    [20000, 24, 2, 2, -1, -1, -2, -2, 3102, 0, 0, 689, 0, 0, 0, 0]
      ]
     }
   }' \
http://127.0.0.1:5001/invocations

```

***Resultado de la predición: 1***
---

