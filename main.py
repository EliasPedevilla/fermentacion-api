import os
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import google.generativeai as genai
from typing import Optional

app = FastAPI()

# CONFIGURACIÓN SEGURA POR VARIABLES DE ENTORNO
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

# CONEXIÓN OFICIAL A TU MONGODB ATLAS
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client["pizzeria_db"]
historico_col = db["historico_fermentacion"]

LATITUD = -32.41
LONGITUD = -63.24

class FeedbackSchema(BaseModel):
    lote_id: str
    resultado: int  # Del 1 al 5
    gramos_usados: Optional[float] = None

def obtener_clima_rango(inicio: datetime, fin: datetime):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUD}&longitude={LONGITUD}&hourly=temperature_2m,relative_humidity_2m&timezone=auto"
        res = requests.get(url).json()
        horas = res["hourly"]["time"]
        temps = res["hourly"]["temperature_2m"]
        hums = res["hourly"]["relative_humidity_2m"]
        
        temps_periodo = []
        hums_periodo = []
        for i, hora_str in enumerate(horas):
            hora_dt = datetime.fromisoformat(hora_str)
            if inicio <= hora_dt <= fin:
                temps_periodo.append(temps[i])
                hums_periodo.append(hums[i])
                
        if not temps_periodo:
            raise HTTPException(status_code=400, detail="No se encontraron datos climáticos.")
            
        return {
            "temp_promedio": round(sum(temps_periodo) / len(temps_periodo), 1),
            "temp_max": max(temps_periodo),
            "temp_min": min(temps_periodo),
            "hum_promedio": round(sum(hums_periodo) / len(hums_periodo), 1),
            "horas_totales": len(temps_periodo)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clima: {str(e)}")

@app.get("/recomendar-levadura")
def recomendar_levadura(
    inicio: str = Query(..., description="YYYY-MM-DD HH:MM"),
    fin: str = Query(..., description="YYYY-MM-DD HH:MM")
):
    try:
        inicio_dt = datetime.strptime(inicio, "%Y-%m-%d %H:%M")
        fin_dt = datetime.strptime(fin, "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato inválido. Usar 'YYYY-MM-DD HH:MM'")

    # Formato ISO 8601 limpio para el lote id
    lote_id = inicio_dt.strftime("%Y-%m-%dT%H:%M")
    clima = obtener_clima_rango(inicio_dt, fin_dt)
    
    # 🔍 LOG 1: Ver qué nos devolvió Open-Meteo
    print(f"\n================ [LOG CLIMA - {lote_id}] ================")
    print(f"Clima obtenido: {clima}")
    
    registros = list(historico_col.find({"estado": "completado"}, {"_id": 0}).sort("_id", 1))
    
    historial_texto = ""
    if registros:
        for r in registros:
            c = r.get("clima_info", {})
            historial_texto += f"Duracion:{c.get('horas_totales')}h|T_Prom:{c.get('temp_promedio')}°C->{r['gramos_usados']}g->Resultado:{r['resultado']}\n"
    else:
        historial_texto = "No hay datos históricos aún."

    # 🔍 LOG 2: Ver el historial exacto que le mandamos a la IA
    print(f"\n================ [LOG HISTORIAL ENVIADO] ================\n{historial_texto}")

    prompt = f"""
    Eres un maestro pizzero experto en fermentación. Calculamos en gramos de LEVADURA SECA INSTANTÁNEA por cada 1kg de harina.
    
    Historial:
    {historial_texto}
    
    Hoy:
    - Duración: {clima['horas_totales']} horas
    - Temp Promedio: {clima['temp_promedio']}°C
    - Temp Máxima: {clima['temp_max']}°C
    - Temp Mínima: {clima['temp_min']}°C
    
    Escala numérica de resultados (1 al 5):
    - 3 ("Genial"): Punto perfecto.
    - 4 o 5 ("Sobrefermentada"): Reduce la levadura seca para climas similares.
    - 1 o 2 ("Poca fermentación"): Sube la levadura seca para climas similares.
    
    Responde ÚNICAMENTE con el número de gramos de levadura seca (ej: 0.8 o 1.5). Sin texto.
    """

    # 🔍 LOG 3: Ver qué responde Gemini exactamente antes de intentar convertirlo a número
    try:
        response = model.generate_content(prompt)
        texto_ia = response.text.strip()
        print(f"\n================ [LOG RESPUESTA GEMINI] ================")
        print(f"Texto crudo de la IA: '{texto_ia}'")
        
        gramos_sugeridos = float(texto_ia)
    except Exception as e:
        print(f"\n❌ [ERROR EN EL BLOQUE GEMINI]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en Gemini o formato: {str(e)}")

    borrador = {
        "_id": lote_id,
        "inicio_fermentacion": inicio,
        "fin_fermentacion": fin,
        "clima_info": clima,
        "gramos_sugeridos_ia": gramos_sugeridos,
        "estado": "pendiente"
    }
    historico_col.replace_one({"_id": lote_id}, borrador, upsert=True)

    return {
        "lote_id": lote_id,
        "levadura_sugerida_gramos_seca": gramos_sugeridos,
        "resumen_clima_periodo": clima
    }

@app.post("/feedback")
def guardar_feedback(data: FeedbackSchema):
    borrador = historico_col.find_one({"_id": data.lote_id})
    if not borrador:
        raise HTTPException(status_code=404, detail="No se encontró el lote_id.")
    
    gramos_finales = data.gramos_usados if data.gramos_usados is not None else borrador["gramos_sugeridos_ia"]
    
    historico_col.update_one(
        {"_id": data.lote_id},
        {
            "$set": {
                "gramos_usados": gramos_finales,
                "resultado": data.resultado,
                "estado": "completado"
            }
        }
    )
    return {"status": "success", "message": f"Lote {data.lote_id} guardado con {gramos_finales}g."}

@app.get("/historial")
def obtener_todo_el_historial():
    try:
        registros = list(historico_col.find({}, {"_id": 0}).sort("inicio_fermentacion", 1))
        
        print(f"\n================ [LOG HISTORIAL GET ALL] ================")
        print(f"Se solicitaron todos los registros. Total encontrados: {len(registros)}")
        
        return {
            "total_registros": len(registros),
            "data": registros
        }
    except Exception as e:
        print(f"\n❌ [ERROR EN GET HISTORIAL]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al leer la base de datos: {str(e)}")