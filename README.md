***Predicción de Abandono Estudiantil con Árboles de Decisión***

Modelo predictivo desarrollado en Python para identificar el riesgo de deserción académica y determinar los factores críticos que influyen en la continuidad de los estudiantes.

🚀 **Introducción al Proyecto**

El objetivo de este trabajo es automatizar el análisis de datos educativos mediante un enfoque basado en Machine Learning. El script procesa un dataset histórico de entrenamiento, evalúa la precisión del modelo mediante un conjunto de prueba, extrae las variables con mayor peso analítico y genera predicciones sobre nuevos registros para exportarlas a un reporte final (Resultado_Final.csv).

🌲 **¿Qué es un Árbol de Decisión?**

Es un algoritmo de aprendizaje supervisado que clasifica datos imitando un diagrama de flujo jerárquico:

Entropía: Mide el grado de desorden o mezcla de clases (continúa/abandona) en un grupo de datos.

Ganancia de Información: Evalúa cuánta reducción de entropía aporta cada variable al realizar una división. La variable con mayor ganancia se ubica en los nodos principales del árbol.

Hojas: Representan la decisión final o categoría estimada (continúa o abandona).

💻 **Implementación en el Código**

Carga y Codificación: Lee el archivo CSV de entrenamiento (sep=';') y utiliza pd.get_dummies() para transformar las variables categóricas en formatos binarios numéricos.

Entrenamiento y Evaluación: Instancia un DecisionTreeClassifier(criterion='entropy'), realiza la partición con train_test_split y calcula el porcentaje de aciertos del modelo.

Análisis de Importancia: Extrae los pesos mediante el atributo .feature_importances_ y genera un gráfico de barras horizontal ordenado para visualizar las 4 variables más influyentes.

Predicción: Aplica las reglas del modelo entrenado sobre el archivo de nuevos alumnos y exporta el resultado final.
