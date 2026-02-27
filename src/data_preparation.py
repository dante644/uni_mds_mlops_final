#TRATAMIENTO DE DATOS

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de rutas
# Obtenemos la ruta raíz y definimos la carpeta de imágenes
ruta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_carpeta_imagenes = os.path.join(ruta_raiz, "imagen")
ruta_training = os.path.join(ruta_raiz, "data", "training")

# Crear carpetas si no existen
os.makedirs(ruta_carpeta_imagenes, exist_ok=True)
os.makedirs(ruta_training, exist_ok=True)


# 2. Carga de datos
ruta_csv = os.path.join(ruta_raiz, "data", "raw", "cleaned_credit_card_data.csv")
df = pd.read_csv(ruta_csv)
print("✅ Datos cargados correctamente.")

# 3. Verificación de valores nulos
# Se realiza la suma de nulos para confirmar que el dataset está limpio
nulos_totales = df.isna().sum().sum()
if nulos_totales == 0:
        print("✅ El dataset no presenta valores nulos.")
else:
        print(f"⚠️ Atención: Se encontraron {nulos_totales} valores nulos.")

# 4. Procesamiento inicial: Eliminar ID
# Se elimina la variable ID ya que no aporta al análisis
if 'ID' in df.columns:
    id_data = df['ID']
    df.drop(columns=['ID'], inplace=True)
    print("✅ ID procesado y eliminado.")

# 5. Cambio de tipos de datos
# Conversión a categóricas y enteros según corresponda
cols_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE', 'default_payment_next_month']
df[cols_categoricas] = df[cols_categoricas].astype('category')

cols_pay = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df[cols_pay] = df[cols_pay].astype('int64')
print("✅ Tipos de datos actualizados.")

# 6. Análisis de Correlación Inicial
cols_corr_completa = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
  'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
  'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
  'default_payment_next_month']

corr_inicial = df[cols_corr_completa].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_inicial, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación Inicial")
plt.savefig(os.path.join(ruta_carpeta_imagenes, "correlacion_inicial.png"))

# 7. Eliminación de variables por alta correlación
# Se eliminan BILL_AMT1, 3, 4 y 6 por su alto grado de correlación
vars_redundantes = ["BILL_AMT1", "BILL_AMT3", "BILL_AMT4", "BILL_AMT6"]
df.drop(columns=vars_redundantes, inplace=True)
print(f"✅ Variables eliminadas por alta correlación: {vars_redundantes}")

# 8. Segunda Matriz de Correlación (Variables filtradas)
# Solo las variables que quedaron tras la limpieza
cols_finales = ['LIMIT_BAL', 'AGE', 'BILL_AMT2', 'BILL_AMT5', 'PAY_AMT1', 
'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
'default_payment_next_month']

corr_final = df[cols_finales].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_final, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación Post-Limpieza")
plt.savefig(os.path.join(ruta_carpeta_imagenes, "correlacion_final.png"))

# 9. Tratamiento de outliers con boxplot
# Seleccionamos las variables numéricas que quedaron en el DataFrame
cols_numericas = df.select_dtypes(include=['number']).columns

# Creamos un gráfico de cajas conjunto
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[cols_numericas])
plt.xticks(rotation=45)
plt.title("Boxplot de Variables Numéricas")

# Guardar y mostrar
plt.savefig(os.path.join(ruta_carpeta_imagenes, "boxplot_variables.png"))
print("📊 Boxplot conjunto generado y guardado.")

# Función de ajuste para Outliers (Capping)
def outlier_capping(x):
    """Aplica un tope al valor del cuantil 0.90 para reducir el impacto de outliers."""
    return x.clip(upper=x.quantile(0.90))

# Columnas identificadas para tratamiento según tus imágenes
cols_outliers = ['BILL_AMT2', 'BILL_AMT5', 'PAY_AMT1', 'PAY_AMT2', 
 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Aplicamos la función de ajuste
df[cols_outliers] = df[cols_outliers].apply(outlier_capping)
print("✅ Tratamiento de outliers completado (Capping al cuantil 0.90).")

print("\n--- Estructura final del DataFrame ---")
print(df.info())


# 10. Guardar dataset final como CSV
ruta_final_csv = os.path.join(ruta_training, "processed_credit_card_data.csv")
df.to_csv(ruta_final_csv, index=False)
print(f"🚀 Dataset final guardado en: {ruta_final_csv}")