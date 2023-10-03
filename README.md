# PI_render
Proyecto Individual 01 / HENRY
# Proyecto Individual Machine Learning Operations (MLOps)

## Autor
Néstor Fabio Cardona C.

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar el rol de `Data Scientist` y `Data Engineer`, aplicando técnicas de extracción, transformación y carga de datos (`ETL`), análisis exploratorio de datos (`EDA`) e implementar una solución de MLOps, para la gestión de 6 consultas (query) para una compañía de juegos de steam.

## Módulos del Proyecto
El proyecto se divide en los siguientes módulos: 

1. **Extracción de Datos**:
Este módulo se encarga de la adquisición de los datos necesarios para las consultas que se piden y dejarlos listos para el tratamiento de los mismos.

PASO 1: Se descomprimen y se cargan los 3 archivos json y los convertimos a dataframes de pandas (steam_games.json, users_items.json y user_reviews.json) y finalmente se convierten a archivos CSV (steam_games_orig.csv, users_items_orig.csv y user_reviews_orig.csv)

PASO 2: Se depuran los 3 archivos csv extraídos en la 1ra parte, se les borra los datos NaN y vacíos a los campos relevantes para los Endpoints y para los join entre los diferentes DF.
steam_games_orig.csv, users_items_orig.csv y user_reviews_orig.csv

2. **Transformación de Datos**: 
PASO 3: Transformaciones en  steam_games
Crear campo 'year' sacado del campo 'release_date',  luego se crean campos con los generos mas usados (99.3%), sacados del campo 'tags' y se eliminan las columnas no necesarias para los endpoints y se crea el archivo final csv 'games_final.csv'.

PASO 4: Transformaciones en user_reviews 
Se crea el campo 'year' sacado del campo 'posted' y se crea igualmente el campo 'sentimiento' con la def  analizar_sentimiento sacado del campo 'review' y finalmente se eliminan las columnas no necesarias para los endpoints y se crea el archivo final csv 'reviews_final.csv', adicional se crea otro archivo csv ('reviews_final.csv') con la información necesaria para el endpoint de ML  recomendacion_juego.

PASO 5: Transformaciones en user_items 
Se renombra campo 'item_id' a 'id' para mejorar joins y se eliminan las columnas no necesarias para los endpoints y se crea el archivo final csv 'items_final.csv'

3. **Carga de Datos**:

PASO 6: Se cargan, los archivos csv ('games_final.csv', 'item s_final.csv' y 'reviews_final.csv'),  creados en el paso anterior, para ser leidos en los endpoints en la API de fastapi. 

4. **API en fastapi**: Aquí se encuentran los scripts y recursos para entrenar diferentes modelos de Machine Learning utilizando los datos adquiridos en el módulo anterior.
PASO 7: Se hacen los endpoints con la información mínima necesaria.

5. **Machine Learning**:
PASO 8: Se hace el endopoint “analisis de sentimiento” @app.get("/sentiment_analysis/{year}") con la información creada en el campo sentimiento, que se creo importando la librería TextBlob pero solo para el idioma inglés.

PASO 9: Se hace el endpoint de “recomendación de 5 juegos”, este es punto de Machine Learning @app.get("/recomendacion_juego/{producto}"), para que la información se pudiera ejecutar en render se hizo una función que saca una muestra aleatoria de datos del 20% ,  “df_sample = df.sample(frac=0.2, random_state=1)” y se obtuvieron las 5 mayores puntuaciones de los juegos steam, usando la estrategia de similitud del coseno para el producto ingresado.

3. **Deploy en render**: 

PASO 10: Se hizo el deploy en render, desde el repositorio de github https://github.com/NestorFCardona/PI_render en el cual esta toda la documentación del proyecto.

## Ejecución
Para ejecutar este proyecto en tu entorno local, ingresa el siguientes link
https://pi-render.onrender.com/docs#/

## Información completa del proyecto
Para acceder al repositorio en tu máquina local, usa el siguiente link
https://github.com/NestorFCardona/PI_render.git
