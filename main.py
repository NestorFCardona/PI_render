# Bibliotecas necesarias y creación de la aplicación FastAPI:
'''
'''
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from enum import Enum
from typing import Union

import ast
app = FastAPI(title="Proyecto MLOps By Néstor Cardona")
# @app.get("/")

'''
1) USER DATA
def userdata( User_id: str ) : Debe devolver cantidadde dinero 
gastado por el usuario, el porcentajede recomendación en base 
a reviews.recommend y cantidad de items.
'''
@app.get("/userdata/{User_id}")
async def userdata( User_id: str ) :
    # Creamos el DataFrame   (Año: str)  
    return {"User Data":User_id}


'''
2) COUNT REVIEWS
def countreviews( YYYY-MM-DDy YYYY-MM-DD: str ) : Cantidad de usuarios que 
realizó reviews entre las fechas dadas y, el porcentajede recomendación 
de los mismos en base a reviews.recommend.
'''
@app.get("/countreviews/{year}/{year2}")
async def countreviews(year: str, year2: str):
    # Tu código aquí
    year = year2
    return {"Count Reviews": year}



# Crear el Enum de géneros
class Genre(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Casual = "Casual"
    Early_Access = "Early Access"
    Free_to_Play = "Free to Play"
    Indie = "Indie"
    Massively_Multiplayer = "Massively Multiplayer"
    RPG = "RPG"
    Racing = "Racing"
    Simulation = "Simulation"
    Sports = "Sports"
    Strategy = "Strategy"
    Video_Production = "Video Production"

'''
3) GÉNERO
def género(género: str ) : Devuelve el puestoen el que se encuentra 
un género sobre el ranking de los mismos analizados bajo 
la columna PlayTimeForever.
'''
@app.get("/genero/{genero}")
async def genero( genero: Genre = None ):
    # Creamos el DataFrame  (Año: str)
    return {"Género:":genero}

'''
4) USER FOR GENRE
def userforgenre( género: str ): Top 5 de usuarios con más horas de juego 
en el género dado, con su URL (del usuario) y user_id.
'''
@app.get("/userforgenre/{genero}")
async def userforgenre( genero: Genre = None ):
    return{"Usuario por Género:":genero}


'''
5) DESARROLLADOR
def desarrollador( desarrollador: str ) : Cantidadde items y 
porcentaje de contenido Free por año según empresa desarrolladora. 
'''
@app.get("/desarrollador/{desarrollador}")
async def desarrollador( desarrollador: str):
    return{"Desarrollador:":desarrollador}

'''
6) ANALISIS DE SENTIMIENTO
def sentiment_analysis( año: int ) : Según el año de lanzamiento, se devuelve 
una lista con la cantidad de registros de reseñas de usuarios 
que se encuentran categorizados con un análisis de sentimiento.
'''
@app.get("/sentiment_analysis/{año}")
async def sentiment_analysis( año: int ):

    return año


