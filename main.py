import os
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import google.generativeai as genai
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = FastAPI()

# CONFIGURACIÓN SEGURA POR VARIABLES DE ENTORNO
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# CONEXIÓN OFICIAL A TU MONGODB ATLAS
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client["pizzeria_db"]
historico_col = db["historico_fermentacion"]

LATITUD = -32.4162156
LONGITUD = -63.2418001

class FeedbackSchema(BaseModel):
    lote_id: str
    resultado: int  # Del 1 al 5
    gramos_usados: Optional[float] = None

def obtener_clima_rango(inicio: datetime, fin: datetime):
    try:
        # Pedimos el pronóstico por hora
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUD}&longitude={LONGITUD}&hourly=temperature_2m,relative_humidity_2m&timezone=auto"
        res = requests.get(url, timeout=10).json()
        
        horas = res["hourly"]["time"]
        temps = res["hourly"]["temperature_2m"]
        hums = res["hourly"]["relative_humidity_2m"]
        
        temps_periodo = []
        hums_periodo = []
        
        for i, hora_str in enumerate(horas):
            # Limpiamos cualquier desfase de formato string
            hora_dt = datetime.fromisoformat(hora_str.replace("Z", ""))
            if inicio <= hora_dt <= fin:
                temps_periodo.append(temps[i])
                hums_periodo.append(hums[i])
                
        # PARACAÍDAS: Si el rango falla o es muy a futuro, usamos los datos globales disponibles para no colgar el sistema
        if not temps_periodo:
            print("⚠️ Rango específico no hallado, aplicando fallback de datos generales.")
            temps_periodo = temps[:24]
            hums_periodo = hums[:24]
            
        duracion_horas = int((fin - inicio).total_seconds() / 3600)
        if duracion_horas <= 0:
            duracion_horas = len(temps_periodo)
            
        return {
            "temp_promedio": round(sum(temps_periodo) / len(temps_periodo), 1),
            "temp_max": max(temps_periodo),
            "temp_min": min(temps_periodo),
            "hum_promedio": round(sum(hums_periodo) / len(hums_periodo), 1),
            "horas_totales": duracion_horas
        }
    except Exception as e:
        print(f"❌ Error interno en procesamiento de clima: {str(e)}")
        # Segundo paracaídas: Datos duros de invierno promedio por si cae la API externa
        return {
            "temp_promedio": 12.0,
            "temp_max": 16.0,
            "temp_min": 7.0,
            "hum_promedio": 70.0,
            "horas_totales": int((fin - inicio).total_seconds() / 3600) or 4
        }

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

    lote_id = inicio_dt.strftime("%Y-%m-%dT%H:%M")
    clima = obtener_clima_rango(inicio_dt, fin_dt)
    
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

    print(f"\n================ [LOG HISTORIAL ENVIADO] ================\n{historial_texto}")

    # PROMPT RIGIDO Y MATEMÁTICO
    prompt = f"""
    Eres un motor de cálculo matemático para fermentación de masa de pizza (1kg de harina, LEVADURA SECA INSTANTÁNEA).
    Tu objetivo es calcular la cantidad exacta de gramos basándote estrictamente en el historial proporcionado, actuando como una función determinista. No inventes variaciones.

    HISTORIAL DE REFERENCIA DE 21 HORAS:
    - Temp Promedio: 9.3°C -> 0.625g -> Resultado: 1 (Insuficiente)
    - Temp Promedio: 8.9°C -> 0.625g -> Resultado: 1 (Insuficiente)
    - Temp Promedio: 10.1°C -> 1.0g -> Resultado: 3 (PERFECTO)

    DATO ACTUAL A CALCULAR:
    - Duración: {clima['horas_totales']} horas
    - Temp Promedio: {clima['temp_promedio']}°C

    REGLA DE CÁLCULO LOGÍSTICO:
    1. Si la duración es de 21 horas y la Temp Promedio está cerca de los 10°C, el punto base de éxito absoluto comprobado es 1.0g.
    2. Si hace más frío que 10.1°C, debes ajustar ligeramente hacia ARRIBA de 1.0g (ej. 1.05g o 1.1g si baja mucho), ya que 0.625g fracasó por completo.
    3. Si hace más calor que 10.1°C, debes ajustar proporcionalmente hacia ABAJO de 1.0g.
    4. Para tiempos más cortos (ej. 3-4 horas), la escala sube exponencialmente hacia 1.2g - 2.4g según el frío.

    Devuelve SIEMPRE el mismo resultado para los mismos datos de entrada.
    Responde ÚNICAMENTE con este formato JSON: {{"gramos": X.XX}}
    """

    try:
        # ACA PASAMOS LA TEMPERATURA A 0 PARA EVITAR VARIACIONES CREATIVAS
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,  # <-- ESTO CLAVA LA RESPUESTA EN MODO DETERMINISTA
                "top_p": 1.0
            }
        )
        texto_ia = response.text.strip()
        print(f"\n================ [LOG RESPUESTA GEMINI] ================")
        print(f"Texto crudo de la IA: '{texto_ia}'")
        
        import json
        datos_ia = json.loads(texto_ia)
        gramos_sugeridos = float(datos_ia["gramos"])
        
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
        return {
            "total_registros": len(registros),
            "data": registros
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer la base de datos: {str(e)}")
    
@app.get("/grafica", response_class=HTMLResponse)    
def mostrar_grafica():
    # Traemos todos los registros de la base de datos igual que en /historial
    registros = list(historico_col.find({}, {"_id": 0}).sort("inicio_fermentacion", 1))
    
    # Procesamos los datos para dárselos masticados a Chart.js
    puntos_grafica = []
    for r in registros:
        clima = r.get("clima_info", {})
        horas = clima.get("horas_totales", 0)
        temp = clima.get("temp_promedio", 0)
        gramos = r.get("gramos_sugeridos_ia", 0)
        id_lote = r.get("_id", "Desconocido")
        
        # Guardamos la estructura que Chart.js necesita para un gráfico de dispersión (Scatter)
        puntos_grafica.append({
            "x": horas,
            "y": gramos,
            "temp": temp,
            "lote": id_lote
        })

    import json
    puntos_json = json.dumps(puntos_grafica)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Panel de Fermentación - Gráfica IA</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                width: 90%;
                max-width: 800px;
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }}
            h1 {{
                color: #ff9800;
                margin-bottom: 5px;
            }}
            p {{
                color: #aaaaaa;
                margin-bottom: 25px;
            }}
        </style>
    </head>
    <body>
        <h1>📊 Curva de Aprendizaje - IA Pizzera</h1>
        <p>Eje X: Horas totales de leudado | Eje Y: Gramos de levadura seca (por 1kg de harina)</p>
        
        <div class="container">
            <canvas id="pizzaChart"></canvas>
        </div>

        <script>
            // Inyectamos los datos reales traídos de MongoDB desde Python
            const datosBackend = {puntos_json};

            const ctx = document.getElementById('pizzaChart').getContext('2d');
            new Chart(ctx, {{
                type: 'scatter',
                data: {{
                    datasets: [{{
                        label: 'Lotes sugeridos por Gemini',
                        data: datosBackend,
                        backgroundColor: '#ff9800',
                        borderColor: '#f57c00',
                        pointRadius: 8,
                        pointHoverRadius: 12
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Horas Totales de Fermentación', color: '#fff' }},
                            grid: {{ color: '#333' }},
                            ticks: {{ color: '#aaa' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Gramos de Levadura Seca', color: '#fff' }},
                            grid: {{ color: '#333' }},
                            ticks: {{ color: '#aaa' }}
                        }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const raw = context.raw;
                                    return [
                                        `Lote: ${{raw.lote}}`,
                                        `⏱️ Duración: ${{raw.x}}h`,
                                        `🍞 Levadura: ${{raw.y}}g`,
                                        `🌡️ Temp Promedio: ${{raw.temp}}°C`
                                    ];
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/recomendar-levadura-ml")
def recomendar_levadura_ml(inicio: str, fin: str):
    try:
        # 1. Calcular las horas del lote solicitado
        formato = "%Y-%m-%d %H:%M"
        dt_inicio = datetime.strptime(inicio, formato)
        dt_fin = datetime.strptime(fin, formato)
        horas_consulta = int((dt_fin - dt_inicio).total_seconds() / 3600)
        
        # 2. Consultar el pronóstico en Open-Meteo
        clima_futuro = obtener_clima_rango(dt_inicio, dt_fin)
        temp_consulta = clima_futuro["temp_promedio"]
        
        # 3. Traer el historial de lotes reales de la DB
        registros = list(historico_col.find({"estado": "completado"}, {"_id": 0}))
        
        X_list = []
        y_list = []
        
        for r in registros:
            clima = r.get("clima_info", {})
            t = clima.get("temp_promedio")
            gramos = r.get("gramos_usados")
            resultado = r.get("resultado")
            
            # Ahora solo requerimos temperatura, gramos y resultado
            if t is None or gramos is None or resultado is None:
                continue
            if resultado == 0:
                continue
                
            # Arbitraje de precisión de 5 niveles (Alineado a tus catas reales)
            if resultado == 1:
                gramos = gramos * 1.50
            elif resultado == 2:
                gramos = gramos * 1.15
            elif resultado == 3:
                pass
            elif resultado == 4:
                gramos = gramos * 0.85
            elif resultado == 5:
                gramos = gramos * 0.50
                
            # X ahora es SOLO la temperatura promedio
            X_list.append([t])
            y_list.append(gramos)
            
        if len(X_list) < 2:  # Para una línea recta sola necesitamos mínimo 2 puntos
            raise ValueError(f"No hay suficientes lotes puntuados para entrenar. Encontrados: {len(X_list)}")

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        # 4. Entrenar la Regresión Lineal de una sola variable (Temperatura)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # 5. Ejecutar la predicción usando solo la temperatura pronosticada
        gramos_predichos = float(modelo.predict(np.array([[temp_consulta]]))[0])
        gramos_finales = max(0.1, min(3.5, round(gramos_predichos, 3)))
        
        # --- EXTRACCIÓN DE LA FÓRMULA INDESTRUCTIBLE ---
        coefs = modelo.coef_
        
        formula_secreta = {
            "constante_base_b0": round(float(modelo.intercept_), 4),
            "peso_temperatura_b1": round(float(coefs[0]), 4),
            "ecuacion_texto": f"Gramos = {round(modelo.intercept_, 3)} + ({round(coefs[0], 3)} * Temp)"
        }
        
        return {
            "motor": "Machine Learning Nativo (Scikit-Learn) v2.1",
            "estado_consulta": "exitoso",
            "lotes_usados_para_entrenar": len(X_train),
            "input_horas": horas_consulta,
            "input_temp_promedio": temp_consulta,
            "gramos_sugeridos": gramos_finales,
            "matematica_viva": formula_secreta
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor ML: {str(e)}")