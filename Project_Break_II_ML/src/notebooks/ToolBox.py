import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr


def cardinalidad(df_in, umbral_categoria, umbral_continua = 30):
    '''
    Esta función obtiene la cardinalidad de cada una de las variables y en función de dicha cardinalidad sugiere un tipo de variable.
    Los tipos posibles son: binaria, categórica, numérica discreta y numérica continua.
    Nota: los tipos mostrados en el DataFrame que retorna la función son solo una sugerencia, es mejor guardar el DataFrame en una variable y modificar los tipos si es necesario.

    
    Argumentos:
    
    df_in (pd.DataFrame): dataset en formato DataFrame para clasificar sus variables.

    umbral_categoría (int): número máximo de valores únicos para clasificar una variable como categórica.

    umbral_continua (int): porcentaje mínimo de valores únicos que debe tener una variable para clasificarla como numérica continua.

    
    Retorna:

    pd.DataFrame: DataFrame con 4 columnas: nombre de la variable, cardinalidad, porcentaje de valores únicos y tipo sugerido
    '''


    cardinalidad = [df_in[col].nunique() for col in df_in.columns]
    cardinalidad_por = [df_in[col].nunique()/len(df_in[col]) for col in df_in.columns]
    dict_df = {"nombre_variable": df_in.columns, "valores_unicos": cardinalidad, "cardinalidad": cardinalidad_por}
    nuevo_df = pd.DataFrame(dict_df)
    nuevo_df["tipo_sugerido"] = "Categórica"
    nuevo_df.loc[nuevo_df["valores_unicos"] == 2, "tipo_sugerido"] = "Binaria"
    nuevo_df.loc[nuevo_df["valores_unicos"] >= umbral_categoria, "tipo_sugerido"] = "Numerica Discreta"
    nuevo_df.loc[nuevo_df["cardinalidad"] >= umbral_continua, "tipo_sugerido"] = "Numerica Continua"
    return nuevo_df


def describe_df(df_in):

    '''
    Esta función obtiene para las variables de un dataset el tipo de datos que contienen, su porcentaje de nulos, sus valores únicos y su cardinalidad.

    
    Argumentos:
    
    df_in (pd.DataFrame): dataset en formato DataFrame para obtener los datos mencionados.

    
    Retorna:

    pd.DataFrame: un DataFrame cuyas columnas son las mismas que las del DataFrame que se pasa como argumento y cuyas filas son los datos que se proporciona sobre las variables, a saber: tipo de datos que contienen, porcentaje de nulos, valores únicos y cardinalidad.
    '''

    data_type = [df_in[col].dtype for col in df_in.columns]
    missings = [df_in[col].isna().value_counts(normalize = True)[True] if True in df_in[col].isna().value_counts().index else 0.0 for col in df_in.columns ]
    cardinalidad = [df_in[col].nunique() for col in df_in.columns]
    cardinalidad_por = [df_in[col].nunique()/len(df_in[col]) for col in df_in.columns]
    dict_df = {"DATA_TYPE": data_type, "MISSINGS (%)": missings, "UNIQUE_VALUES": cardinalidad, "CARDIN (%)": cardinalidad_por}
    nuevo_df = pd.DataFrame(dict_df, df_in.columns)
    return nuevo_df.T

def get_features_cat_regression(df, target_col, p_value = 0.05, umbral_cat = 10):
    '''
    Esta función filtra las variables categóricas de un dataset introducido para entrenar un modelo de regresión lineal. 
    Verifica el tipo de variable llamando a la función "cardinalidad".
    Si la variable es binaria aplica el test Mann-Whitney U, si es categórica pero no binaria aplica el test ANOVA para comprobar su relación con la variable target.
    Si el valor p de los tests está por debajo del umbral especificado en "p_value" puede descartarse la hipótesis nula (la variable target y la categórica no estan relacionadas) con confianza estadística y se añade la variable a la lista de features categóricas para el modelo.


    
    Argumentos:

    df(pd.DataFrame): DataFrame cuyas variables categóricas se desea filtrar.

    target_col(string): nombre de la columna target que se pretende predecir con el modelo.

    p_value(float): umbral de valor p por debajo del cual debe estar el valor p del test aplicado para determinar la relación entre una variable y el target para añadir dicha variable a la lista de features categóricas.
    
    umbral_cat(int): controla el número máximo de valores que puede tener una columna para ser considerada categórica.

    

    Retorna:

    list: Lista de features categóricas para entrenar un modelo de regresión lineal con el dataset dado.
    '''
    if not is_numeric_dtype(df[target_col]):
        target_col = input("Tu variable objetivo no es de tipo numérico, introduce una nueva variable target o la palabra 'parar' para dejar de ejecutar la función.")
        if target_col == "parar":
            return "La función no se ha ejecutado porque has decidido pararla"
    if len(df.loc[df[target_col].isna()]) > 0:
        raise Exception(f"La variable '{target_col}' tiene valores nulos, introduce una variable target sin nulos")    
    from scipy.stats import mannwhitneyu
    from scipy.stats import f_oneway
    lista_cat = []
    for col in df:
        if len(df.loc[df[col].isna()]) > 0:
            raise Exception(f"La variable '{col}' tiene valores nulos, introduce un DataFrame sin nulos")
        tipo_col = cardinalidad(df[[col]], umbral_categoria = umbral_cat).tipo_sugerido[0]
        if tipo_col == "Binaria":
            value_1 = df[col].unique()[0]
            value_2 = df[col].unique()[1]
            group_a = df.loc[df[col] == value_1, target_col]
            group_b = df.loc[df[col] == value_2, target_col]
            _, p_val = mannwhitneyu(group_a, group_b)
            print(f"Para '{target_col}' y '{col}' el p-value es: {p_val} (Test realizado: Mann-Whitney U)")
            print(p_val)
            if p_val < p_value:
                lista_cat.append(col)
        elif tipo_col == "Categórica":
            groups = df[col].unique()
            target_values_x_group = [df.loc[df[col] == group, target_col] for group in groups]
            _, p_val = f_oneway(*target_values_x_group)
            print(f"Para '{target_col}' y '{col}' el p-value es: {p_val} (Test aplicado: ANOVA)")
            if p_val < p_value:
                lista_cat.append(col)
    return lista_cat




def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False, umbral_cat=10):
    
    """
    Pinta histogramas agrupados de columnas categóricas que muestran relación estadísticamente significativa
    con una variable target numérica. Usa Mann-Whitney U para 2 grupos y ANOVA para más.

    Argumentos:
    df (pd.DataFrame): El DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo (debe ser numérica continua o discreta con alta cardinalidad).
    columns (list of str): Lista de nombres de columnas categóricas a analizar. Si está vacía, se seleccionan automáticamente.
    pvalue (float): Nivel de significación estadística para los test (por defecto 0.05).
    with_individual_plot (bool): Si es True, se muestran gráficos individuales por variable.

    Devuelve:
    list of str | None: Lista de variables categóricas con relación estadísticamente significativa, o None si hay error.
    """

    # Comprobamos: Si es un DataFrame
    
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame de pandas.")
        return None

    # Comprobamos: Si target_col es una columna en el DataFrame y si no es un string
    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: 'target_col' debe ser una columna válida del DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser numérico.")
        return None

    if df[target_col].isna().sum() > 0:
        print(f"Error: La variable '{target_col}' tiene valores nulos.")
        return None

    if df[target_col].nunique() < 10:
        print("Error: 'target_col' debe tener alta cardinalidad (mínimo 10 valores únicos).")
        return None

    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        print("Error: 'columns' debe ser una lista de strings.")
        return None

    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1.")
        return None

    if not isinstance(with_individual_plot, bool):
        print("Error: 'with_individual_plot' debe ser booleano.")
        return None


      # Si no se especifican columnas,  las obtenemos automáticamente
    if not columns:
        columns = get_features_cat_regression(df, target_col, p_value=pvalue, umbral_cat=umbral_cat)
    else:
        # Validar columnas pasadas y filtrar por significancia
        valid_columns = [col for col in columns if col in df.columns]
        columns = [col for col in valid_columns
                   if col in get_features_cat_regression(df[[col, target_col]], target_col, p_value=pvalue, umbral_cat=umbral_cat)]

    if not columns:
        print("No se encontraron variables categóricas con relación significativa con el target.")
        return []

    # Graficar si se solicita
    for col in columns:
        if with_individual_plot:
            plt.figure(figsize=(8, 4))
            sns.histplot(data=df, x=target_col, hue=col, multiple="stack", kde=True)
            plt.title(f"{target_col} por {col}")
            plt.tight_layout()
            plt.show()

    return columns


import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr


def get_features_num_regression(df, target_col, umbral_corr, p_value=None):

    """
    Esta funcion se encarga de comprarar la columna target con el resto y calcular la correlacion, sea igual o mayor al umbral pasado

    Valores:
    df: DataFrame a revisar.
    target_col: Nombre de la columna target. 
    umbral_corr: Umbral de correlación (valor entre 0 y 1).
    p_value (float, opcional): Valor p para test de hipótesis. 

    Devuelve:
    Lista de nombres de columnas numéricas con la correlacion entre ellas.
    """

    #Validaciones. 

    #Validamos que df = un DATAFRAME. Si no arroja error. if not isinstance(df, pd.DataFrame): verifica si df NO es un df de Pandas 

    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un valor DataFrame")
        return None

    #Con el is istance validamos que umbral_corr sea un float o integer y despues validamos que este entre 0 y 1. 
    if not isinstance(umbral_corr, (float, int)):
        print("Error: umbral_corr debe ser un número (float o int)")
        return None

    if umbral_corr < 0 or umbral_corr > 1:
        print("Errorr: umbral_corr debe estar entre 0 y 1")
        return None 
    
    #Validamos que se coloque el nombre de la target_col bien
    if target_col not in df.columns:
        print("Error: target_col tiene otro nombre o no esta en este dataframe, verifica los datos")
        return None
    
    #Validamos que sea una columna numerica (en este caso no usamos is instance porque nos dio error en pruebas, ya que al hacer pruebas el tipo de dato era Serie y el is instance no lo tomaba porque buscaba un valor diferente un float)
    if not is_numeric_dtype(df[target_col]):
        print("Error: target_col tiene que ser una columna numerica")
        return None
    
    #P-value inicia como None pero si se llega a pasar su valor debemos comprobar lo mismo. Repetimos el is instance porque es un valor unico y no una serie como arriba
    if p_value is not None:
        if not isinstance(p_value, (float, int)):
            print("Error: p_value tiene que ser un numero")
            return None
        
        #Copiar y pegar comporbacion de umbral corr 
        if p_value < 0 or p_value > 1:
            print("Errorr: p_value tiene que estar entre 0 y 1")
            return None    

    # Si todo ok, print
    print("Features numericas Ok Validadas")
    
    # Identificar columnas numéricas
    columnas_numericas = []  # aquí guardaremos solo las columnas que sean numerics

    for columna in df.columns:
        if is_numeric_dtype(df[columna]): #Usamos la misma is_numeric_dtype
            columnas_numericas.append(columna) #La añadimos a nuestra lista vacia 
    
    print(f"Columnas numéricas en df: {columnas_numericas}")
   
    #Calcular correlaciones con el target_col que sea superior
    columnas_filtradas = []  # aquí guardaremos las columnas que tienen alta correlación con el target

    for columna in columnas_numericas:
        # Evitamos calcular la correlación de target_col consigo mismo porque dara 1=1
        if columna != target_col:
            correlacion = df[columna].corr(df[target_col])  # usamos .corr() de pandas
            print(f"Correlación entre '{columna}' y '{target_col}': {correlacion:}")

            # Si la correlación es mayor al umbral, guardamos la columna
            if abs(correlacion) > umbral_corr: #con el abs evitamos que nos de error por que solo nos daba relacion fuerte positiva.
                columnas_filtradas.append(columna)
                print(f"{columna} supera el umbral de {umbral_corr}")
            else:
                print(f"{columna} no supera el umbral de {umbral_corr}")

    # p_value = none, devolvemos ya la lista
    if p_value is None:
        return columnas_filtradas
    
    # si p_value = valor entre 0 y 1, mediremos que tan probable es que haya ocurrido el evento por csualidad con pearson

    columnas_significativas = []
    #Recorremos las columnas que yapasaron el filtro de correlaciony eliminamos las que tengan nulos
    for columna in columnas_filtradas:
        datos = df[[columna, target_col]].dropna()
        if len(datos) < 3:
            print(f"Error: No se puede aplicar Pearson en '{columna}' por pocos datos.")
            continue
        r, p = pearsonr(datos[columna], datos[target_col])
        if p <= (1 - p_value):
            columnas_significativas.append(columna)
            print(f" {columna} pasa el test de p-value (p={p:.4f})")
        else:
            print(f"{columna} NO pasa el test de p-value (p={p:.4f})")
            
    return columnas_significativas
    



def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    
    """
    Pinta un gráfico de dispersión comparando la variable independiente con las variables dependientes que cumplan con los criterios de p-value y mayor que el umbral de correlación
    Usa la función get_features_num_regression para obtener las columnas significativas y se hagan las comprobaciones necesarias.

    Argumentos:
    df (pd.DataFrame): El DataFrame con los datos.
    target_col (str): Nombre de la columna objetivo o dependiente (debe ser numérica continua o discreta con alta cardinalidad).
    columns (list of str): Lista de nombres de columnas categóricas a analizar. Si está vacía, se seleccionan automáticamente.
    umbral_corr (float): Umbral de correlación minimo para filtrar las variables independientes en valores absolutos (valor entre 0 y 1).
    pvalue (float): Nivel de significación estadística para los test (por defecto None, normalmente se usa 0.05).
    

    Devuelve:
    list of str | None: Lista de variables númericas con relación estadísticamente significativa, o None si hay error.
    """
    
    #hacemos una llamda a la funcion get_features_num_regression para obtener las columnas significativas y se hagan las comprobaciones necesarias
    columnas_significativas = get_features_num_regression(df = df, target_col = target_col, umbral_corr = umbral_corr, p_value= pvalue)
    # Si columns es igual a cero utilizamos todas las columnas del dataframe 
    if len(columns) == 0:
        columns = df.columns[:]
    
    # Hacemos una intersección entre las columnas del dataframe y las columnas significativas
    columnas_validadas = [col for col in columns if col in columnas_significativas]
    #En caso de no haber columnas validas, mostramos un mensaje y salimos de la funcion
    if not columnas_validadas:
        print("No hay columnas que cumplan con los criterios de correlación y p-value.")
        return
    else:
        ini = 0
        # Utilizamos un bucle para que grafique de 5 en 5
        while ini <= len(columnas_validadas):
            columnas_validadas1 = columnas_validadas[ini:ini+5]
            # Grafico pairplot con target col como variable dependiente representada en el eje Y
            sns.pairplot(df, x_vars=columnas_validadas1, y_vars=[target_col])
            # Añadimo titulo al gráfico
            plt.suptitle(f"Gráficos de dispersión para {target_col} y columnas significativas", y=1.04)
            # Mostramos el gráfico
            plt.show()
            #sumamos 5 al indice para que se muestren las siguientes 5 columnas
            ini += 5
