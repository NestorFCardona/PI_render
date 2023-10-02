# Bibliotecas necesarias y creación de la aplicación FastAPI:
'''
'''
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast
from typing import List, Union


app = FastAPI(title="Proyecto MLOps By Néstor Cardona")
# @app.get("/")

'''
 Debes crear las siguientes funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).

3)    def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

4)    def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]



5)    def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}



6)    def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Si es un sistema de recomendación user-item:

6)    def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

'''

'''
1) PLAY TIME GENRE
def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.

Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
'''

games = pd.read_csv('games_final.csv')
reviews = pd.read_csv('reviews_final.csv')
items_hora = pd.read_csv('itemsPorHorasJugadas.csv')

# Crear el Enum de géneros de acuerdo a los mas importantes 99% de uso
class Genre(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Casual = "Casual"
    Free_to_Play = "Free to Play"
    Indie = "Indie"
    RPG = "RPG"
    Simulation = "Simulation"
    Strategy = "Strategy"
    Singleplayer = "Single Player"
    Multiplayer = 'Multiplayer'
@app.get("/PlayTimeGenre/{genero}")
async def PlayTimeGenre(genero: Genre = None):
    # Paso 1: Unir los DataFrames en función de la columna 'id'
    merged_df = pd.merge(games, items_hora, on='id')

    # Paso 2: Filtrar por género
    genre_df = merged_df[merged_df[genero.value] == 1]

    if genre_df.empty:
        return {"message": f"No hay datos para el género {genero.value}"}

    # Paso 3: Agrupar por año y sumar las horas jugadas
    genre_hours_per_year = genre_df.groupby(['year'])['playtime_forever'].sum().reset_index()

    # Paso 4: Encontrar el año con la suma máxima de horas jugadas
    max_hours_year = genre_hours_per_year.loc[genre_hours_per_year['playtime_forever'].idxmax()]

    return {"El Año de lanzamiento": max_hours_year['year'],"fue con más horas jugadas para el Género": genero.value,
    "con un acumulado de": max_hours_year['playtime_forever']}


'''
2) USER FOR GENRE
def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
'''
@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: Genre = None):
    '''
    USER FOR GENRE
    Debe devolver el usuario que acumula más horas jugadas para el género dado y 
    una lista de la acumulación de horas jugadas por año.

    '''    
    year = genero
    return {"Usuario con más horas jugadas para Género": year}


'''
3) USERS RECOMMEND 
def UsersRecommend( año: int ) : Devuelve el top 3 de juegos MÁS 
recomendados por usuarios para el año dado. 
(reviews.recommend = True y comentarios positivos/neutrales)
'''
@app.get("/UsersRecommend/{year}")
async def UsersRecommend(year: int):
    '''
    USERS RECOMMEND
    Esta funcion Devuelve el top 3 de juegos MÁS recomendados 
    por usuarios para el año dado. 
    (reviews.recommend = True y comentarios positivos/neutrales)
    '''
    # Filtra las reseñas para el año dado, donde 'recommend' es True y el sentimiento es positivo o neutral
    filtered_reviews = reviews[(reviews['year'] == year) & (reviews['recommend'] == True) & (reviews['sentimiento'].isin(['Positivo', 'Neutral']))]

    # Agrupa las reseñas por juego ('id') y cuenta cuántas reseñas tiene cada juego
    game_counts = filtered_reviews['id'].value_counts().reset_index()
    game_counts.columns = ['id', 'count']

    # Ordena los juegos por la cantidad de reseñas en orden descendente
    top_games = game_counts.sort_values(by='count', ascending=False).head(3)

     # Cambia la columna 'id' a tipo str en ambos DataFrames
    game_counts['id'] = game_counts['id'].astype(str)
    games['id'] = games['id'].astype(str)
    top_games['id'] = top_games['id'].astype(str)

    # Fusiona (merge) con el DataFrame 'games' para obtener los nombres de los juegos
    top_games_with_names = pd.merge(top_games, games[['id', 'app_name']], left_on='id', right_on='id', how='left')

    # Crea los tres resultados independientes con el nombre del juego
    results = [{"Puesto {}: ".format(i + 1): {"Juego": game_data['app_name'], "Cantidad de reseñas": game_data['count']}} for i, (_, game_data) in enumerate(top_games_with_names.iterrows())]

    # Formatea los resultados en el formato deseado
    formatted_result = [{"Puesto {}: ".format(i + 1) + result_key: result_value for result_key, result_value in result.items()} for i, result in enumerate(results)]

    return formatted_result

'''
4) USERS NOT RECOMMEND
def UsersNotRecommend( año: int ) : Devuelve el top 3 de juegos MENOS recomendados
por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
'''

@app.get("/UsersNotRecommend/{year}")
async def UsersNotRecommend(year: int):
    '''
    USERS RECOMMEND
    Esta funcion Devuelve el top 3 de juegos MENOS recomendados 
    por usuarios para el año dado. (reviews.recommend = False y comentarios Negativos)
    '''
    # Filtra las reseñas para el año dado, donde 'recommend' es True y el sentimiento es Negativo
    #---filtered_reviews = reviews[(reviews['year'] == year) & (reviews['recommend'] == False) & (reviews['sentimiento'].isin(['Negativo']))]
    #filtered_reviews = reviews[(reviews['year'] == year) & (reviews['recommend'] == True) & (reviews['sentimiento'].isin(['Positivo', 'Neutral']))]
    filtered_reviews = reviews[(reviews['year'] == year) & ~((reviews['recommend'] == True) & (reviews['sentimiento'].isin(['Positivo', 'Neutral'])))]


    # Agrupa las reseñas por juego ('id') y cuenta cuántas reseñas tiene cada juego
    game_counts = filtered_reviews['id'].value_counts().reset_index()
    game_counts.columns = ['id', 'count']

    # Ordena los juegos por la cantidad de reseñas en orden descendente
    top_games = game_counts.sort_values(by='count', ascending=False).head(3)

    # Cambia la columna 'id' a tipo str en ambos DataFrames
    game_counts['id'] = game_counts['id'].astype(str)
    games['id'] = games['id'].astype(str)
    top_games['id'] = top_games['id'].astype(str)

    # Fusiona (merge) con el DataFrame 'games' para obtener los nombres de los juegos
    top_games_with_names = pd.merge(top_games, games[['id', 'app_name']], left_on='id', right_on='id', how='left')

    # Crea los tres resultados independientes con el nombre del juego
    results = [{"Puesto {}: ".format(i + 1): {"Juego": game_data['app_name'], "Cantidad de reseñas": game_data['count']}} for i, (_, game_data) in enumerate(top_games_with_names.iterrows())]

    # Formatea los resultados en el formato deseado
    result_formateado = [{"Puesto {}: ".format(i + 1) + result_key: result_value for result_key, result_value in result.items()} for i, result in enumerate(results)]

    return result_formateado


'''
5) ANALISIS DE SENTIMIENTO
def sentiment_analysis( año: int ) : Según el año de lanzamiento, se devuelve 
una lista con la cantidad de registros de reseñas de usuarios 
que se encuentran categorizados con un análisis de sentimiento.
'''
@app.get("/sentiment_analysis/{año}")
async def sentiment_analysis( año: int ):
    '''
    ANALISIS DE SENTIMIENTO
    Según el año de lanzamiento, se devuelve 
    una lista con la cantidad de registros de reseñas de usuarios 
    que se encuentran categorizados con un análisis de sentimiento.
    '''
    # Filtrar reseñas para el año especificado y contar la cantidad de cada sentimiento
    sentiment_counts = reviews[reviews['year'] == año]['sentimiento'].value_counts()

    # Devolver el resultado como un diccionario
    return {
        'Negativo': sentiment_counts.get('Negativo', 0),
        'Neutral': sentiment_counts.get('Neutral', 0),
        'Positivo': sentiment_counts.get('Positivo', 0)
    }

'''
6) RECOMENDACION JUEGO
Ingresando el id de producto, deberíamos recibir una lista con 5 juegos 
recomendados similares al ingresado.
'''
@app.get("/recomendacion_juego/{producto}")
async def recomendacion_juego(producto: int):
    '''
    RECOMENDACION JUEGO
    Ingresando el id de producto, debes recibir 5 juegos recomendados 
    similares al ingresado.
    '''

    # Cargar el dataframe 'reviews' desde tu fuente de datos
    df = pd.read_csv('reviews_ML.csv')  # Reemplaza 'reviews.csv' con tu fuente de datos

    # Tomar una muestra aleatoria del 20% de los datos
    df_sample = df.sample(frac=0.2, random_state=1)

    # Reindexar df_sample
    df_sample.reset_index(drop=True, inplace=True)

    #df_sample.to_csv('reviews_sampleML.csv') # PARA REVISAR DATOS

    # Crear un vectorizador TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Calcular la matriz TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_sample['review'])

    # Obtener el índice del producto ingresado por el usuario
    producto_ingresado = producto  # Reemplaza 'ID_DEL_PRODUCTO' con el ID proporcionado por el usuario

    # Buscar directamente el índice del producto ingresado en la muestra
    indice_producto_ingresado = df_sample[df_sample['id'] == producto_ingresado].index

    # Verificar si se encontró el producto ingresado
    if not indice_producto_ingresado.empty:
        indice_producto_ingresado = indice_producto_ingresado[0]  # Tomar el primer índice encontrado

        # Calcular la similitud del coseno entre los juegos
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Obtener las puntuaciones de similitud del coseno para el producto ingresado
        similarity_scores = cosine_sim[indice_producto_ingresado]

        # Enumerar las puntuaciones de similitud para ordenarlas
        similarity_scores = list(enumerate(similarity_scores))

        # Ordenar los juegos por similitud del coseno en orden descendente
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Obtener los 5 juegos recomendados (excluyendo el juego ingresado)
        top_recommendations = [df_sample.iloc[score[0]] for score in sorted_scores[1:6]]

        # Crear una lista para almacenar los nombres de los juegos recomendados
        resultados = []

        # Iterar sobre los juegos recomendados y agregar sus nombres a la lista de resultados
        for i, juego in enumerate(top_recommendations, start=1):
            juego_nombre = juego['name']
            resultado = f"Juego #{i}: {juego_nombre} - Similitud del coseno: {sorted_scores[i-1][1]:.4f}"
            resultados.append(resultado)

        # Agregar el nombre del producto ingresado a los resultados
        nombre_producto = df_sample.loc[indice_producto_ingresado, 'name']
        resultados.insert(0, f"Producto ingresado: {nombre_producto}")

        # Devolver los nombres de los 5 juegos recomendados como respuestas separadas
        return resultados
    else:
        return [f"Error: El producto ingresado ({producto_ingresado}) no se encuentra en la muestra."]

