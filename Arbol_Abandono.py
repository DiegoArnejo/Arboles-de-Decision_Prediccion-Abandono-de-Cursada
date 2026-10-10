import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df_train = pd.read_csv('TablaPrediccionAbandono-Entrenamiento.csv', sep=';', encoding='latin1')

# Separa las caracteristicas (lo que usamos para adivinar, como notas y asistencia)
X = df_train.drop(columns=['EstadoFinal'])
y = df_train['EstadoFinal'] 

# Palabras a numeros
X_numerico = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_numerico, y, test_size=0.33, random_state=42)

print("\n--- EJEMPLO DE DIVISION ---")
print("Cantidad para entrenar:", len(X_train))
print("Cantidad para probar:", len(X_test))

# Entrenamiento del arbol (Usando Entropia)

# Creamos el arbol de decision configurado para usar entropia
arbol = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Entrenamos el modelo con los datos de entrenamiento
arbol.fit(X_train, y_train)

# Que tan bueno es el modelo, evaluacion del modelo con el conjunto de prueba
aciertos = arbol.score(X_test, y_test)
print(f"\nPrecision del modelo en las pruebas: {aciertos * 100:.2f}%")

# Que variables tienen mas ganancia de informacion
importancias = arbol.feature_importances_
nombres_columnas = X_numerico.columns

print("\nVariables mas importantes para decidir el abandono (Ganancia de Informacion):")
# Une la columna con su nivel de importancia y ordena de mayor a menor
ranking = sorted(zip(nombres_columnas, importancias), key=lambda x: x[1], reverse=True)

for columna, valor in ranking[:4]:
    print(f" - {columna.replace('_97,1', '')}: {valor:.4f}")


# Grafico
# Las 4 variables mas importantes
top_variables = ranking[:4]
variables_nombres = [item[0].replace('_97,1', '') for item in top_variables]
variables_valores = [item[1] for item in top_variables]

plt.figure(figsize=(8, 4))
plt.barh(variables_nombres[::-1], variables_valores[::-1], color='blue')
plt.xlabel('Nivel de Importancia (Ganancia de Informacion)')
plt.ylabel('Variables del Estudiante')
plt.title('Variables mas importantes para predecir el Abandono')
plt.tight_layout()
plt.show() 


# Prediccion Final
# Carga el archivo de los alumnos nuevos (Prediccion)
df_pred = pd.read_csv('TablaPrediccionAbandono-Prediccion.csv', sep=';', encoding='latin1')
df_pred_numerico = pd.get_dummies(df_pred).reindex(columns=X_numerico.columns, fill_value=0)

# El arbol predice si abandona o continua
predicciones = arbol.predict(df_pred_numerico)

df_pred['Resultado_Prediccion'] = predicciones
df_pred.to_csv('Resultado_Final.csv', index=False, sep=';', encoding='utf-8-sig')
