
# 🔍 Proyecto de Detección de Fraudes con Machine Learning

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de detectar transacciones fraudulentas en un entorno bancario, utilizando datos simulados de tarjetas, usuarios y transacciones. Fue diseñado para incorporarse a un portafolio profesional orientado a roles en ciencia de datos.

## 🚀 Demo en vivo

Puedes probar la app en línea:

🌐 [Streamlit App →](https://proyectofraude-gmmuhnecsqpb6zbkyatapp.streamlit.app)

---

## 📦 Estructura del Proyecto

```
📁 Proyecto_Fraude/
│
├── app.py                       # App principal en Streamlit
├── modelo_fraude_mejorado_3.pkl # Modelo RandomForest entrenado
├── imputer_model.pkl           # Objeto de imputación entrenado
├── scaler_model.pkl            # Escalador estándar entrenado
├── numeric_cols.json           # Lista de columnas numéricas usadas
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

---

## ⚙️ Stack Tecnológico

- Python 3.11
- pandas, numpy, scikit-learn, imbalanced-learn
- Streamlit para interfaz interactiva
- RandomForest con tuning + SMOTE
- Google Colab para entrenamiento
- GitHub + Streamlit Cloud para despliegue

---

## 🧠 Descripción del Modelo

El modelo fue entrenado con un dataset sintético enriquecido, utilizando una **heurística ponderada** para definir la variable objetivo `is_fraud`:

```
is_fraud = 
  0.5 * (amount > 3000) +
  0.3 * card_on_dark_web +
  0.2 * has_error
```

Se utilizó:
- SMOTE para balancear clases
- RandomizedSearchCV para optimizar hiperparámetros
- Ingeniería de features temporal y categórica

---

## 📊 Ejemplo de Predicciones

| Caso                         | Resultado                                    | Probabilidad de Fraude |
|------------------------------|----------------------------------------------|-------------------------|
| Transacción Sospechosa       | ⚠️ Sospechosa de fraude                      | 89.91%                  |
| Transacción Legítima         | ✅ No sospechosa                             | 3.74%                   |

---

## 🧪 Cómo Ejecutar Localmente

1. Clona este repositorio:

```bash
git clone https://github.com/Pedro-Rubio/Proyecto_Fraude.git
cd Proyecto_Fraude
```

2. Crea un entorno y activa:

```bash
python -m venv env
source env/bin/activate  # en Linux/Mac
env\Scripts\activate     # en Windows
```

3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta la app:

```bash
streamlit run app.py
```

---

## 📁 Dataset

El dataset `dataset_fraude_final.csv` fue generado a partir de tres fuentes simuladas: `transactions`, `cards` y `users`, unidas y enriquecidas. No se incluye por razones de tamaño, pero puedes solicitarlo vía [contacto](mailto:analistas@datatrust.ai).

---

## 📈 Métricas del Modelo

- **Accuracy:** 1.00 (en datos balanceados)
- **ROC AUC:** 1.00
- **Recall (fraude):** 1.00
- **Precision (fraude):** 1.00

---

## 🤝 Créditos

Este proyecto fue desarrollado por **Pedro Rubio** como parte de su portafolio profesional de Ciencia de Datos, combinando técnicas de Machine Learning, ingeniería de datos y BI.

---

## 📬 Contacto

📧 analistas@datatrust.ai  
🔗 [LinkedIn](https://www.linkedin.com/in/pedro-rubio) *(reemplaza si aplica)*

---

> “La mejor manera de predecir el futuro es creándolo.” – Peter Drucker
