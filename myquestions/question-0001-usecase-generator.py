import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
def generar_caso_de_uso_consolidar_pedidos_por_proveedor():
    """
    Genera un caso de uso aleatorio para
    consolidar_pedidos_por_proveedor(df).

    Retorna:
        input  (dict): {"df": DataFrame con datos de pedidos}
        output (pd.DataFrame): resultado esperado de la función
    """
    rng = np.random.default_rng()

    proveedores_base = rng.choice(
        ["TechSupply", "GlobalParts", "QuickDeliver", "MegaStore", "FastGoods"],
        size=rng.integers(3, 6),
        replace=False,
    ).tolist()

    categorias = rng.choice(
        ["electronica", "ropa", "alimentos", "herramientas", "papeleria"],
        size=rng.integers(2, 4),
        replace=False,
    ).tolist()

    n_filas = int(rng.integers(15, 40))

    def ensuciar(nombre):
        variantes = [
            nombre,
            nombre.upper(),
            nombre.lower(),
            f"  {nombre}  ",
            nombre.capitalize(),
        ]
        return str(rng.choice(variantes))

    proveedores_col = [ensuciar(str(rng.choice(proveedores_base))) for _ in range(n_filas)]
    categorias_col = [str(rng.choice(categorias)) for _ in range(n_filas)]
    cantidades = rng.integers(1, 500, size=n_filas).tolist()
    precios = np.round(rng.uniform(0.5, 300.0, size=n_filas), 2).tolist()

    df = pd.DataFrame({
        "proveedor": proveedores_col,
        "categoria": categorias_col,
        "cantidad": cantidades,
        "precio_unit": precios,
    })

    # --- Calcular output esperado ---
    df_work = df.copy()
    df_work["proveedor"] = df_work["proveedor"].str.strip().str.lower()
    df_work["valor_total"] = df_work["cantidad"] * df_work["precio_unit"]

    resultado = (
        df_work.groupby(["proveedor", "categoria"], as_index=False)
        .agg(
            total_unidades=("cantidad", "sum"),
            total_valor=("valor_total", "sum"),
            num_pedidos=("cantidad", "count"),
        )
        .sort_values(["proveedor", "total_valor"], ascending=[True, False])
        .reset_index(drop=True)
    )

    input_dict = {"df": df.copy()}
    return input_dict, resultado
    # --- Ejemplo de uso ---
if __name__ == "__main__":
    
    # 1. Generar un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_consolidar_pedidos_por_proveedor()
    
    # 2. Mostrar INPUT
    print("=== INPUT (Diccionario) ===")
    print("DataFrame de pedidos (primeras 5 filas):")
    print(entrada["df"].head())
    
    # 3. Mostrar OUTPUT esperado
    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame consolidado:")
    print(salida_esperada.head())
    
    # 4. Información adicional útil
    print("\n=== INFO ===")
    print(f"Número de filas de entrada: {len(entrada['df'])}")
    print(f"Número de filas consolidadas: {len(salida_esperada)}")
    print("Columnas de salida:", list(salida_esperada.columns))