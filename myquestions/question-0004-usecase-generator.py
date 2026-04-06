import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
def generar_caso_de_uso_pipeline_imputacion_gradient_boosting():
    """
    Genera un caso de uso aleatorio para
    pipeline_imputacion_gradient_boosting(df, target_col, test_size).

    Retorna:
        input  (dict): {"df": DataFrame, "target_col": str, "test_size": float}
        output (dict): resultado esperado de la función
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(150, 400))
    n_features = int(rng.integers(4, 10))
    n_classes = int(rng.integers(2, 4))
    test_size = round(float(rng.choice([0.15, 0.2, 0.25, 0.3])), 2)

    feature_cols = [f"feat_{i+1}" for i in range(n_features)]
    target_col = "clase"

    X_data = rng.standard_normal(size=(n_samples, n_features))

    # Introducir NaN (~12% de celdas)
    mask = rng.random(size=X_data.shape) < 0.12
    X_data = X_data.astype(float)
    X_data[mask] = np.nan

    # Generar etiquetas con cierta separabilidad
    coef = rng.standard_normal(size=n_features)
    scores = np.nansum(X_data * coef, axis=1)
    cuts = np.nanpercentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1])
    y_data = np.digitize(scores, bins=cuts)

    df = pd.DataFrame(X_data, columns=feature_cols)
    df[target_col] = y_data

    # --- Calcular output esperado ---
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    resultado = {
        "accuracy": round(float((y_pred == y_test).mean()), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    input_dict = {
        "df": df.copy(),
        "target_col": target_col,
        "test_size": test_size,
    }
    return input_dict, resultado
    # --- Ejemplo de uso ---
if __name__ == "__main__":
    
    # 1. Generar un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_pipeline_imputacion_gradient_boosting()
    
    # 2. Mostrar INPUT
    print("=== INPUT ===")
    
    print("\n📊 DataFrame (primeras 5 filas):")
    print(entrada["df"].head())
    
    print("\n🎯 Columna objetivo:")
    print(entrada["target_col"])
    
    print("\n⚙️ Test size:")
    print(entrada["test_size"])
    
    # 3. Mostrar OUTPUT esperado
    print("\n=== OUTPUT ESPERADO ===")
    print("Métricas del modelo:")
    print(salida_esperada)
    
    # 4. Información útil
    print("\n=== INFO ===")
    print(f"Número total de muestras: {len(entrada['df'])}")
    print(f"Número de features: {entrada['df'].shape[1] - 1}")
    print(f"Número de clases: {entrada['df'][entrada['target_col']].nunique()}")
    print(f"Tamaño entrenamiento: {salida_esperada['n_train']}")
    print(f"Tamaño prueba: {salida_esperada['n_test']}")