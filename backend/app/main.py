from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import auth, markets, products, prices, predictions, dashboard
from backend.app.database import engine, Base
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear tablas en la base de datos (solo para desarrollo/MVP)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AgroDashRD API",
    description="API REST para la aplicación agrícola AgroDashRD",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (Permitir Flutter Web y Dev)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "*"  # Permitir todo en desarrollo, restringir en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(auth.router)
app.include_router(markets.router)
app.include_router(products.router)
app.include_router(prices.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)

@app.get("/", tags=["Health"])
async def root():
    """
    Endpoint de salud para verificar que el servidor está corriendo.
    """
    return {"status": "ok", "message": "AgroDashRD API is running 🚀"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AgroDashRD Backend...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AgroDashRD Backend...")
