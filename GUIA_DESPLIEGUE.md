# Guía de despliegue de AgroDashRD

Esta guía describe cómo desplegar lo que existe hoy: el paquete `agrodash` (catálogo, precios, ingesta de datos, cuentas y pronósticos) sobre PostgreSQL. La aplicación web (Streamlit) y la imagen Docker única llegan en las fases F5 y F6; esta guía se completará entonces.

## 1. Base de datos PostgreSQL

1. Creá una base PostgreSQL 16 gestionada (Render, Supabase, Neon o un VPS propio).
2. Copiá su URL de conexión, con la forma `postgresql://usuario:clave@host:5432/agrodash`.
3. No la escribas en ningún archivo del repositorio: va solo en variables de entorno del servidor.

## 2. Variables de entorno

| Variable | Valor en producción |
|---|---|
| `AGRODASH_ENV` | `production` (rechaza arrancar si la base no es PostgreSQL) |
| `AGRODASH_DATABASE_URL` | la URL del paso 1 |
| `AGRODASH_DRIVE_FILE_IDS` | opcional; por defecto, los reportes oficiales mayorista (.xlsx) y minorista (.xls) |

Definilas en el panel del proveedor o en el entorno del proceso, no en un `.env` dentro del servidor: `.env` se lee desde la carpeta en la que se ejecuta el comando.

## 3. Instalar y preparar la base

```bash
uv sync --frozen                      # instala las dependencias exactas de uv.lock
uv run alembic upgrade head           # crea o actualiza todas las tablas
uv run agrodash create-admin --email admin@ejemplo.com --superadmin
```

La contraseña del administrador se pide por teclado (o se lee de la entrada estándar), nunca como argumento.

## 4. Primera carga de datos

```bash
uv run agrodash ingest --drive        # descarga y carga la historia oficial desde Google Drive
uv run agrodash quality-report        # revisa rechazos, nombres sin revisar y series listas
```

La base de producción debe cargarse **desde cero** con la versión actual del código: una base cargada con versiones anteriores conserva factores de unidad viejos (por ejemplo, los rangos de peso como "Unidad/8-10 libra").

## 5. Tareas programadas

Programá una tarea semanal (cron del servidor o "cron job" del proveedor) que ejecute:

```bash
uv run agrodash ingest --drive
```

y, a continuación, el pronóstico:

```bash
uv run agrodash forecast
```

Volver a cargar un archivo ya cargado no hace nada, así que es seguro ejecutarla más seguido. El pronóstico tarda entre 4 y 5 minutos porque vuelve a medir los modelos alternativos contra el último precio; `--skip-benchmark` lo hace en segundos.

## 6. Copias de seguridad

Activá las copias automáticas diarias del proveedor de PostgreSQL. Las tablas importantes son `prices`, `products`, `product_aliases`, `units`, `users` e `ingestion_runs`; todo lo demás se puede regenerar.
