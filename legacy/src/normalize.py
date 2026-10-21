"""
Utilidades de normalización para AgroDashRD.

Este módulo contiene funciones para normalizar y estandarizar
nombres de rubros y productos agrícolas.
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Diccionarios para normalización
RUBRO_MAPPING = {
    # Avícolas - simplificado como solicitado
    "Avicolas": "Avícolas",
    "Avicolass": "Avícolas",
    "Avícolas": "Avícolas",
    "avicolas": "Avícolas",
    "AVICOLAS": "Avícolas",
    "Aves": "Avícolas",
    "Pollos": "Avícolas",
    "Huevos": "Avícolas",

    # Lácteos - simplificado como solicitado
    "Lacteos": "Lácteos",
    "Lácteos": "Lácteos",
    "Lacteo": "Lácteos",
    "Lácteo": "Lácteos",
    "lacteo": "Lácteos",
    "Lateo": "Lácteos",
    "lateo": "Lácteos",
    "Leche": "Lácteos",
    "Queso": "Lácteos",
    "LACTEOS": "Lácteos",

    # Legumbres - simplificado como solicitado, usando solo "Legumbres"
    "Legumbres": "Legumbres",
    "Legumbres Y Hortalizas": "Legumbres",
    "Legumbres-Hortalizas": "Legumbres",
    "Legumbres y Hortalizas": "Legumbres",
    "Legumbres-secas": "Legumbres",
    "Legumbres secas": "Legumbres",
    "Legumbres-Secas": "Legumbres",
    "Leguminosas": "Legumbres",
    "Leguminosas Secas": "Legumbres",
    "Leguminosas secas": "Legumbres",
    "Habichuela": "Legumbres",
    "Habichuelas": "Legumbres",
    "Guandules": "Legumbres",
    "Guandul": "Legumbres",
    "LEGUMBRES": "Legumbres",

    # Musáceas
    "Musáceas": "Musáceas",
    "Musaceas": "Musáceas",
    "musaceas": "Musáceas",
    "MUSACEAS": "Musáceas",
    "Plátano": "Musáceas",
    "Platano": "Musáceas",
    "Guineo": "Musáceas",
    "Banana": "Musáceas",

    # Oleaginosas
    "Oleaginosas": "Oleaginosas",
    "Oleginosas": "Oleaginosas",
    "oleaginosas": "Oleaginosas",
    "OLEAGINOSAS": "Oleaginosas",
    "Aceite": "Oleaginosas",

    # Pecuario - simplificado como solicitado
    "Pecuario": "Pecuario",
    "Pecuarios": "Pecuario",
    "pecuario": "Pecuario",
    "pecuarios": "Pecuario",
    "PECUARIO": "Pecuario",
    "PECUARIOS": "Pecuario",
    "Pescuario": "Pecuario",
    "Carne": "Pecuario",
    "Carnes": "Pecuario",

    # Raíces y Tubérculos
    "Raices-Tuberculos": "Raíces-Tubérculos",
    "Raices Tuberculos": "Raíces-Tubérculos",
    "Raices Y Tuberculos": "Raíces-Tubérculos",
    "Raices-tuberculos": "Raíces-Tubérculos",
    "Raices y tuberculos": "Raíces-Tubérculos",
    "Raíces y Tubérculos": "Raíces-Tubérculos",
    "Tubérculos": "Raíces-Tubérculos",
    "Tuberculos": "Raíces-Tubérculos",
    "Raíz": "Raíces-Tubérculos",
    "Raiz": "Raíces-Tubérculos",
    "Yuca": "Raíces-Tubérculos",
    "Batata": "Raíces-Tubérculos",
    "Papa": "Raíces-Tubérculos",
    "TUBERCULOS": "Raíces-Tubérculos",
    "RAICES": "Raíces-Tubérculos",

    # Cereales - corregido el error tipográfico "Celeares"
    "Cereales": "Cereales",
    "Cereal": "Cereales",
    "Cerealess": "Cereales",
    "Celeares": "Cereales",
    "cereal": "Cereales",
    "cereales": "Cereales",
    "CEREALES": "Cereales",
    "Arroz": "Cereales",
    "Maíz": "Cereales",
    "Maiz": "Cereales",

    # Frutas
    "Frutas": "Frutas",
    "Frutass": "Frutas",
    "Frutales": "Frutas",
    "Fruta": "Frutas",
    "frutas": "Frutas",
    "FRUTAS": "Frutas",
    "Cítricos": "Frutas",
    "Citricos": "Frutas",

    # Hortalizas
    "Hortalizas": "Hortalizas",
    "Hortaliza": "Hortalizas",
    "hortalizas": "Hortalizas",
    "HORTALIZAS": "Hortalizas",
    "Vegetales": "Hortalizas",
    "Vegetal": "Hortalizas",
    "Verduras": "Hortalizas",
    "Verdura": "Hortalizas",
    "Ajíes": "Hortalizas",
    "Ají": "Hortalizas",
    "Ajies": "Hortalizas",
    "Aji": "Hortalizas",
    "Tomate": "Hortalizas",
    "Cebolla": "Hortalizas",
    "Zanahoria": "Hortalizas",
}

# Mapeo de productos para normalizar variedades similares
PRODUCTO_MAPPING = {
    # Papas/Patatas/Batatas
    r"papa.*": "Papa",
    r"batata.*": "Batata",
    r"patata.*": "Papa",

    # Ajíes/Pimientos
    r"aj[ií].*cubanela.*": "Ají Cubanela",
    r"aj[ií].*morrón.*": "Ají Morrón",
    r"aj[ií].*picante.*": "Ají Picante",
    r"aj[ií].*cachucha.*": "Ají Cachucha",
    r"pimiento.*rojo.*": "Ají Morrón Rojo",
    r"pimiento.*verde.*": "Ají Morrón Verde",
    r"pimiento.*amarillo.*": "Ají Morrón Amarillo",

    # Tomates
    r"tomate.*ensalada.*": "Tomate Ensalada",
    r"tomate.*barceló.*": "Tomate Barceló",
    r"tomate.*pasta.*": "Tomate Pasta",
    r"tomate.*americano.*": "Tomate Americano",
    r"tomate.*cherry.*": "Tomate Cherry",

    # Plátanos/Guineos
    r"pl[áa]tano.*barahon.*": "Plátano Barahonero",
    r"pl[áa]tano.*cibaeño.*": "Plátano Cibaeño",
    r"pl[áa]tano.*verde.*": "Plátano Verde",
    r"pl[áa]tano.*maduro.*": "Plátano Maduro",
    r"guineo.*verde.*": "Guineo Verde",
    r"guineo.*maduro.*": "Guineo Maduro",
    r"guineo.*manzano.*": "Guineo Manzano",
    r"banana.*": "Guineo",

    # Habichuelas/Frijoles
    r"habichuela.*negra.*": "Habichuela Negra",
    r"habichuela.*roja.*": "Habichuela Roja",
    r"habichuela.*blanca.*": "Habichuela Blanca",
    r"habichuela.*pinta.*": "Habichuela Pinta",
    r"fr[ií]jol.*negro.*": "Habichuela Negra",
    r"fr[ií]jol.*rojo.*": "Habichuela Roja",
    r"fr[ií]jol.*blanco.*": "Habichuela Blanca",
    r"fr[ií]jol.*pinto.*": "Habichuela Pinta",
    r"poroto.*": "Habichuela",

    # Cebollas
    r"cebolla.*roja.*": "Cebolla Roja",
    r"cebolla.*blanca.*": "Cebolla Blanca",
    r"cebolla.*amarilla.*": "Cebolla Amarilla",
    r"cebolla.*morada.*": "Cebolla Morada",

    # Arroz
    r"arroz.*selecto.*": "Arroz Selecto",
    r"arroz.*primera.*": "Arroz de Primera",
    r"arroz.*segunda.*": "Arroz de Segunda",
    r"arroz.*integral.*": "Arroz Integral",
    r"arroz.*popular.*": "Arroz Popular",

    # Frutas
    r"limón.*": "Limón",
    r"limon.*": "Limón",
    r"naranja.*dulce.*": "Naranja Dulce",
    r"naranja.*agria.*": "Naranja Agria",
    r"pi[ñn]a.*": "Piña",
    r"guayaba.*": "Guayaba",
    r"mango.*": "Mango",
    r"chinola.*": "Chinola",
    r"parchita.*": "Chinola",
    r"maracuy[áa].*": "Chinola",

    # Otros vegetales
    r"lechuga.*": "Lechuga",
    r"zanahoria.*": "Zanahoria",
    r"remolacha.*": "Remolacha",
    r"berenjena.*": "Berenjena",
    r"repollo.*": "Repollo",
    r"apio.*": "Apio",
    r"pepino.*": "Pepino",
    r"calabac[ií]n.*": "Calabacín",
    r"calabaza.*": "Calabaza",
    r"auyama.*": "Auyama",
    r"tayota.*": "Tayota",
}

# Diccionario para normalización de mercados
MERCADO_MAPPING = {
    # Mercado Nuevo (normalizar variaciones)
    "Mercado_Nuevo": "Mercado Nuevo",
    "Mercado Nuevo": "Mercado Nuevo",
    "mercado nuevo": "Mercado Nuevo",
    "mercado_nuevo": "Mercado Nuevo",
    "Mdo. Nuevo": "Mercado Nuevo",
    "Mdo Nuevo": "Mercado Nuevo",
    "MercadoNuevo": "Mercado Nuevo",

    # Otros mercados (normalizar para consistencia)
    "Mercado_Conaprope": "Mercado Conaprope",
    "Mercado Conaprope": "Mercado Conaprope",

    "Mercado_Los_Mina": "Mercado Los Mina",
    "Mercado Los Mina": "Mercado Los Mina",
    "Mercado los mina": "Mercado Los Mina",

    "Mercado_Villa_Consuelo": "Mercado Villa Consuelo",
    "Mercado Villa Consuelo": "Mercado Villa Consuelo",

    "Mercado_Cristo_Rey": "Mercado Cristo Rey",
    "Mercado Cristo Rey": "Mercado Cristo Rey",

    "Mercadom": "Mercadom",

    "Supermercado": "Supermercado",
}

# Cache para aumentar el rendimiento de funciones de normalización
_RUBRO_CACHE = {}
_PRODUCTO_CACHE = {}
_MERCADO_CACHE = {}

def clean_text(text: str) -> str:
    """
    Limpia y estandariza un texto para normalización.

    Args:
        text: Texto original

    Returns:
        Texto limpio y normalizado
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Convertir a minúsculas
    text = text.lower().strip()

    # Eliminar caracteres especiales y múltiples espacios
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text

@lru_cache(maxsize=1000)
def normalize_rubro(rubro: str) -> str:
    """
    Normaliza nombres de rubros usando un mapeo y lógica de similitud.

    Args:
        rubro: Nombre del rubro a normalizar

    Returns:
        Nombre del rubro normalizado
    """
    if pd.isna(rubro) or not isinstance(rubro, str) or rubro.strip() == "":
        # Verificamos si hay una categoría base para usar como predeterminada
        logger.warning(f"Valor de rubro clasificado como 'Frutas': {rubro}")
        print(f"VALOR CLASIFICADO COMO FRUTAS: {rubro} (Tipo: {type(rubro)})")
        return "Frutas"  # Categoría predeterminada en lugar de "Otros"

    # Convertir a string, limpiar y verificar caché
    rubro_str = str(rubro).strip()

    if rubro_str in _RUBRO_CACHE:
        return _RUBRO_CACHE[rubro_str]

    # Buscar coincidencia exacta en el diccionario
    if rubro_str in RUBRO_MAPPING:
        result = RUBRO_MAPPING[rubro_str]
        _RUBRO_CACHE[rubro_str] = result
        return result

    # Buscar coincidencias sin distinguir mayúsculas/minúsculas
    rubro_lower = rubro_str.lower()
    for key, value in RUBRO_MAPPING.items():
        if key.lower() == rubro_lower:
            _RUBRO_CACHE[rubro_str] = value
            return value

    # Buscar coincidencias parciales priorizando palabras clave específicas
    for key, value in RUBRO_MAPPING.items():
        # Verificamos si alguna palabra clave está en el texto
        keyword_lower = key.lower()
        if keyword_lower in rubro_lower or rubro_lower in keyword_lower:
            _RUBRO_CACHE[rubro_str] = value
            logger.info(f"Normalizado rubro por coincidencia parcial: '{rubro_str}' -> '{value}'")
            return value

    # Verificamos por palabras clave específicas que ayuden a clasificar
    rubro_words = rubro_lower.split()

    # Mapeo de palabras clave a rubros
    keyword_mapping = {
        "pollo": "Avícolas", "huevo": "Avícolas", "gallina": "Avícolas", "ave": "Avícolas",
        "leche": "Lácteos", "queso": "Lácteos", "yogurt": "Lácteos", "mantequilla": "Lácteos",
        "habichuela": "Legumbres", "guandul": "Legumbres", "lenteja": "Legumbres", "garbanzo": "Legumbres",
        "platano": "Musáceas", "plátano": "Musáceas", "guineo": "Musáceas", "banana": "Musáceas",
        "aceite": "Oleaginosas", "manteca": "Oleaginosas", "oliva": "Oleaginosas",
        "carne": "Pecuario", "res": "Pecuario", "cerdo": "Pecuario", "chivo": "Pecuario", "cordero": "Pecuario",
        "yuca": "Raíces-Tubérculos", "papa": "Raíces-Tubérculos", "batata": "Raíces-Tubérculos", "ñame": "Raíces-Tubérculos",
        "arroz": "Cereales", "maiz": "Cereales", "maíz": "Cereales", "trigo": "Cereales", "avena": "Cereales",
        "naranja": "Frutas", "limón": "Frutas", "piña": "Frutas", "mango": "Frutas", "manzana": "Frutas",
        "tomate": "Hortalizas", "cebolla": "Hortalizas", "ají": "Hortalizas", "zanahoria": "Hortalizas", "lechuga": "Hortalizas"
    }

    # Revisamos cada palabra por separado
    for word in rubro_words:
        if word in keyword_mapping:
            result = keyword_mapping[word]
            _RUBRO_CACHE[rubro_str] = result
            logger.info(f"Normalizado rubro por palabra clave '{word}': '{rubro_str}' -> '{result}'")
            return result

    # Si no se encontró ninguna coincidencia, usamos el rubro original con formato mejorado
    # (primera letra en mayúscula) en lugar de usar "Sin clasificar"
    result = rubro_str.capitalize()
    _RUBRO_CACHE[rubro_str] = result
    logger.warning(f"No se pudo normalizar el rubro: '{rubro_str}', se mantiene como '{result}'")
    return result

@lru_cache(maxsize=1000)
def normalize_producto(producto: str) -> str:
    """
    Normaliza nombres de productos usando expresiones regulares y mapeo.

    Args:
        producto: Nombre del producto a normalizar

    Returns:
        Nombre del producto normalizado
    """
    if pd.isna(producto) or not isinstance(producto, str) or producto.strip() == "":
        return "Producto no especificado"  # En lugar de "Sin especificar"

    # Convertir a string, limpiar y verificar caché
    producto_str = str(producto).strip()

    if producto_str in _PRODUCTO_CACHE:
        return _PRODUCTO_CACHE[producto_str]

    # Limpiar el texto para la coincidencia
    producto_clean = clean_text(producto_str)

    # Buscar coincidencias en patrones regex
    for pattern, normalized in PRODUCTO_MAPPING.items():
        if re.match(pattern, producto_clean, re.IGNORECASE):
            _PRODUCTO_CACHE[producto_str] = normalized
            logger.info(f"Normalizado producto por regex: '{producto_str}' -> '{normalized}'")
            return normalized

    # Buscar coincidencias parciales con productos base
    base_productos = {
        "ají": ["aji", "pimiento", "chili", "picante"],
        "arroz": ["rice", "blanco", "selecto"],
        "plátano": ["platano", "guineo", "banano", "banana"],
        "habichuela": ["frijol", "poroto", "judía", "alubia"],
        "papa": ["patata", "potato"],
        "yuca": ["casava", "mandioca"],
        "cebolla": ["onion", "cebollín", "chalote"],
        "tomate": ["tomato", "jitomate"],
        "lechuga": ["lettuce", "escarola"],
        "zanahoria": ["carrot"],
        "naranja": ["orange", "china"],
        "limón": ["limon", "lima", "lime", "lemon"]
    }

    for base_producto, variantes in base_productos.items():
        if any(variante in producto_clean for variante in variantes) or base_producto in producto_clean:
            result = base_producto.capitalize()
            _PRODUCTO_CACHE[producto_str] = result
            logger.info(f"Normalizado producto por coincidencia parcial: '{producto_str}' -> '{result}'")
            return result

    # Si no hay coincidencias, devolver el original con formato mejorado
    # (primera letra de cada palabra en mayúscula)
    words = producto_str.split()
    result = ' '.join(word.capitalize() for word in words)
    _PRODUCTO_CACHE[producto_str] = result
    return result

@lru_cache(maxsize=1000)
def normalize_mercado(mercado: str) -> str:
    """
    Normaliza nombres de mercados usando un diccionario de mapeo.

    Args:
        mercado: Nombre del mercado a normalizar

    Returns:
        Nombre del mercado normalizado
    """
    if pd.isna(mercado) or not isinstance(mercado, str) or mercado.strip() == "":
        return "Mercado no especificado"  # En lugar de "Sin especificar"

    # Convertir a string y limpiar
    mercado_str = str(mercado).strip()

    # Verificar caché
    if mercado_str in _MERCADO_CACHE:
        return _MERCADO_CACHE[mercado_str]

    # Buscar coincidencia exacta
    if mercado_str in MERCADO_MAPPING:
        result = MERCADO_MAPPING[mercado_str]
        _MERCADO_CACHE[mercado_str] = result
        return result

    # Buscar coincidencias sin distinguir mayúsculas/minúsculas
    mercado_lower = mercado_str.lower()
    for key, value in MERCADO_MAPPING.items():
        if key.lower() == mercado_lower:
            _MERCADO_CACHE[mercado_str] = value
            return value

    # Buscar coincidencia parcial más agresiva
    for key, value in MERCADO_MAPPING.items():
        # Primera verificación: nombres que contienen al otro
        key_lower = key.lower()
        if (key_lower in mercado_lower or mercado_lower in key_lower) and len(mercado_lower) > 3:
            _MERCADO_CACHE[mercado_str] = value
            logger.info(f"Normalizado mercado por inclusión: '{mercado_str}' -> '{value}'")
            return value

        # Segunda verificación: comparar palabras individuales
        mercado_words = mercado_lower.split()
        key_words = key_lower.split()
        common_words = set(mercado_words).intersection(set(key_words))

        # Si hay al menos una palabra en común (que no sea una preposición o artículo)
        meaningful_words = [word for word in common_words 
                          if word not in ['de', 'la', 'el', 'los', 'las', 'del', 'y', 'en', 'con']]

        if len(meaningful_words) > 0:
            _MERCADO_CACHE[mercado_str] = value
            logger.info(f"Normalizado mercado por palabras comunes: '{mercado_str}' -> '{value}'")
            return value

    # Si el nombre incluye una palabra clave específica de mercado, usar esa clasificación
    market_keywords = {
        "nuevo": "Mercado Nuevo",
        "mina": "Mercado Los Mina",
        "consuelo": "Mercado Villa Consuelo",
        "cristo": "Mercado Cristo Rey",
        "conaprope": "Mercado Conaprope"
    }

    for keyword, market_name in market_keywords.items():
        if keyword in mercado_lower:
            _MERCADO_CACHE[mercado_str] = market_name
            logger.info(f"Normalizado mercado por palabra clave '{keyword}': '{mercado_str}' -> '{market_name}'")
            return market_name

    # Si no se encuentra, devolver el original con formato mejorado
    words = mercado_str.split()
    result = ' '.join(word.capitalize() for word in words)
    _MERCADO_CACHE[mercado_str] = result
    logger.warning(f"No se pudo normalizar el mercado: '{mercado_str}', se mantiene como '{result}'")
    return result

def normalize_dataframe_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las columnas de Rubro, Producto y Mercado en un DataFrame.

    Args:
        df: DataFrame a normalizar

    Returns:
        DataFrame con columnas normalizadas
    """
    df_copy = df.copy()

    # Normalizar columna de Rubro si existe
    rubro_col = None
    for col in df.columns:
        if col.lower() in ['rubro', 'categoria', 'categoría', 'category']:
            rubro_col = col
            break

    if rubro_col is not None:
        logger.info(f"Normalizando columna de rubro: {rubro_col}")
        df_copy[rubro_col] = df_copy[rubro_col].apply(normalize_rubro)

    # Normalizar columna de Producto si existe
    producto_col = None
    for col in df.columns:
        if col.lower() in ['producto', 'productos', 'product', 'item']:
            producto_col = col
            break

    if producto_col is not None:
        logger.info(f"Normalizando columna de producto: {producto_col}")
        df_copy[producto_col] = df_copy[producto_col].apply(normalize_producto)

    # Normalizar columna de Mercado si existe
    mercado_col = None
    for col in df.columns:
        if col.lower() in ['mercado', 'market', 'plaza']:
            mercado_col = col
            break

    if mercado_col is not None:
        logger.info(f"Normalizando columna de mercado: {mercado_col}")
        df_copy[mercado_col] = df_copy[mercado_col].apply(normalize_mercado)

    return df_copy

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """
    Obtiene los valores únicos de una columna, filtrando valores nulos.

    Args:
        df: DataFrame de origen
        column: Nombre de la columna

    Returns:
        Lista de valores únicos no nulos
    """
    if column not in df.columns:
        return []

    values = df[column].dropna().unique().tolist()
    return sorted([str(v) for v in values if v and not pd.isna(v)])