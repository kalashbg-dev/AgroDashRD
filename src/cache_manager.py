"""
Módulo para manejar el caché de datos preprocesados.
"""
import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
import hashlib
import logging
from typing import Optional, Tuple, Dict, Any

# --- NUEVO: Función para hash de archivos fuente ---
def get_file_hash(file_path: str) -> str:
    """
    Calcula el hash MD5 de un archivo.
    """
    h = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return ''

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"
CACHE_EXPIRY_DAYS = 1  # Número de días antes de que el caché expire

def get_cache_path(prefix: str = "data") -> str:
    """
    Obtiene la ruta del archivo de caché.
    
    Args:
        prefix: Prefijo para el nombre del archivo de caché
        
    Returns:
        Ruta completa del archivo de caché
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{prefix}_cache.pkl")

def load_cached_data(prefix: str) -> Optional[Dict[str, Any]]:
    """
    Carga datos desde el caché si están disponibles y no han expirado.
    
    Args:
        prefix: Prefijo del archivo de caché a cargar
        
    Returns:
        Diccionario con los datos en caché o None si no hay caché o ha expirado
    """
    cache_path = get_cache_path(prefix)
    
    if not os.path.exists(cache_path):
        logger.info("No se encontró archivo de caché")
        return None
        
    try:
        # Verificar cuándo se modificó por última vez el archivo
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_time > timedelta(days=CACHE_EXPIRY_DAYS):
            logger.info("El caché ha expirado")
            return None
            
        # Cargar datos del caché
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            
        logger.info("Datos cargados desde caché")
        return cached_data
        
    except Exception as e:
        logger.error(f"Error al cargar datos del caché: {str(e)}")
        return None

def save_data_to_cache(data: Dict[str, Any], prefix: str) -> None:
    """
    Guarda los datos en el caché.
    
    Args:
        data: Datos a guardar en caché
        prefix: Prefijo para el archivo de caché
    """
    try:
        cache_path = get_cache_path(prefix)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Datos guardados en caché: {cache_path}")
    except Exception as e:
        logger.error(f"Error al guardar en caché: {str(e)}")

def get_data_hash(data: pd.DataFrame) -> str:
    """
    Genera un hash único para el DataFrame.
    
    Args:
        data: DataFrame para generar el hash
        
    Returns:
        Hash MD5 del contenido del DataFrame
    """
    return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

def process_and_cache_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Procesa los datos y los guarda en caché, o los carga del caché si están disponibles.
    
    Returns:
        Tupla con (df_mayorista, df_minorista, df_combined)
    """
    from src.data_loader import load_and_preprocess_data
    
    # Intentar cargar datos del caché
    # --- NUEVO: Calcular hash de archivos fuente ---
    mayorista_path = os.path.join('data', 'mayorista.csv')
    minorista_path = os.path.join('data', 'minorista.csv')
    hash_mayorista = get_file_hash(mayorista_path)
    hash_minorista = get_file_hash(minorista_path)
    source_hash = f"{hash_mayorista}_{hash_minorista}"

    cached_data = load_cached_data("processed_data")
    if cached_data is not None:
        try:
            # Validar hash de archivos fuente
            if 'source_hash' in cached_data and cached_data['source_hash'] == source_hash:
                df_mayorista = cached_data['mayorista']
                df_minorista = cached_data['minorista']
                df_combined = cached_data['combined']
                if not df_mayorista.empty and not df_minorista.empty and not df_combined.empty:
                    logger.info("Usando datos del caché (hash fuente válido)")
                    return df_mayorista, df_minorista, df_combined
            else:
                logger.info("Hash de archivos fuente ha cambiado. Reprocesando datos...")
        except Exception as e:
            logger.warning(f"Error al cargar datos del caché: {str(e)}")

    # Si no hay caché válido, cargar y procesar los datos
    logger.info("Procesando datos...")
    df_mayorista, df_minorista, df_combined = load_and_preprocess_data()

    # Guardar en caché junto con el hash fuente
    try:
        cache_data = {
            'mayorista': df_mayorista,
            'minorista': df_minorista,
            'combined': df_combined,
            'timestamp': datetime.now(),
            'source_hash': source_hash
        }
        save_data_to_cache(cache_data, "processed_data")
    except Exception as e:
        logger.error(f"No se pudo guardar en caché: {str(e)}")

    return df_mayorista, df_minorista, df_combined
