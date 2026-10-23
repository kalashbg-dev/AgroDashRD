# Guía de despliegue de AgroDashRD

AgroDashRD se despliega como **una sola imagen Docker** que sirve la aplicación web y ejecuta las tareas, más una base PostgreSQL. Esta guía cubre un servidor propio con Docker (opción recomendada) y, al final, un proveedor gestionado.

## Ruta rápida (servidor con Docker)

1. Instala Docker Engine 20.10 o superior y el plugin Compose v2 en el servidor (Ubuntu 22.04 o superior, 2 GB de RAM y 3 GB de disco libres como mínimo).
2. Clona el repositorio en `/srv/agrodash` y crea el archivo `.env`:

   ```bash
   git clone https://github.com/kalashbg-dev/AgroDashRD.git /srv/agrodash
   cd /srv/agrodash
   echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > .env
   ```

3. Levanta la base, la migración y la web. La primera vez construye la imagen (1,3 GB; entre 5 y 15 minutos según el servidor):

   ```bash
   docker compose up -d
   docker compose ps          # db y web "healthy" (la web, a los 10-30 s); migrate "exited (0)"
   ```

4. Crea la primera cuenta (la contraseña se pide por teclado, mínimo 12 caracteres):

   ```bash
   docker compose run --rm web create-admin --email admin@ejemplo.com --superadmin
   ```

5. Carga la historia oficial y calcula los pronósticos (unos 10 minutos):

   ```bash
   docker compose run --rm weekly
   ```

6. Publica la web con HTTPS (sección siguiente) y programa la tarea semanal.

Resultado esperado: la web responde en `http://127.0.0.1:8501` dentro del servidor, y en Precios aparecen los productos con su último precio.

## HTTPS con Caddy

La web se publica solo en `127.0.0.1:8501`; un proxy inverso la expone con HTTPS. Con [Caddy](https://caddyserver.com) instalado en el servidor y el dominio apuntando a su IP, el archivo `/etc/caddy/Caddyfile` queda así:

```
precios.ejemplo.com {
    reverse_proxy 127.0.0.1:8501
}
```

Caddy obtiene y renueva el certificado solo, y reenvía los WebSockets que usa Streamlit. Después: `sudo systemctl reload caddy`.

## Tarea semanal

Los informes oficiales se publican una vez por semana. Agrega esta línea con `crontab -e` (lunes a las 06:00):

```
0 6 * * 1  cd /srv/agrodash && docker compose run --rm weekly >> /var/log/agrodash-weekly.log 2>&1
```

| Paso | Qué hace | Si falla |
|---|---|---|
| `agrodash migrate` | Aplica los cambios de esquema pendientes (corre antes, como servicio aparte) | La tarea no arranca; revisa `docker compose logs migrate` |
| `agrodash ingest --drive` | Descarga los informes de Google Drive y carga solo lo nuevo | No calcula pronósticos. Un archivo que no se pudo cargar aparece como "Falló" en Administración → Cargar datos; una descarga fallida solo queda en `/var/log/agrodash-weekly.log` |
| `agrodash forecast` | Pronostica 4 semanas y, con 3 años de historia, 12 meses | La web sigue mostrando la última corrida correcta |

Volver a cargar un archivo ya cargado no hace nada y migrar una base al día tampoco, así que la tarea puede repetirse sin riesgo, incluso en un servidor nuevo o justo después de actualizar.

## Configuración

| Variable | Dónde | Valor |
|---|---|---|
| `POSTGRES_PASSWORD` | `.env` | Solo letras y dígitos: también va dentro de la URL de la base |
| `POSTGRES_USER`, `POSTGRES_DB` | `.env` (opcional) | `agrodash` por defecto |
| `AGRODASH_DRIVE_FILE_IDS` | `.env` (opcional) | Por defecto, los informes oficiales mayorista y minorista |

La imagen fija `AGRODASH_ENV=production`, que se niega a arrancar si la base no es PostgreSQL. No subas `.env` al repositorio (está en `.gitignore`).

## Copias de seguridad

Copia diaria comprimida, con cron (todos los días a las 02:00):

```
0 2 * * *  cd /srv/agrodash && docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' | gzip > /srv/backups/agrodash-$(date +\%F).sql.gz
```

Para restaurar en una base vacía:

```bash
gunzip -c /srv/backups/agrodash-AAAA-MM-DD.sql.gz | docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" "$POSTGRES_DB"'
```

Lo que no se puede regenerar son las cuentas (`users`), los reportes de campo y las decisiones del catálogo (`products`, `product_aliases`). Precios oficiales y pronósticos se reconstruyen con la tarea semanal.

## Actualizar a una versión nueva

```bash
cd /srv/agrodash
git pull
docker compose build
docker compose up -d       # migrate se ejecuta antes de arrancar la web
```

La próxima tarea semanal usa la imagen nueva sin pasos extra.

## Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `web` no arranca y `migrate` terminó con error | La base no está lista o la contraseña tiene símbolos | `docker compose logs migrate`; usa una contraseña de letras y dígitos |
| "No hay precios cargados" | La tarea semanal no se ejecutó | `docker compose run --rm weekly` |
| Un producto dice "sin pronóstico" | Menos de dos años de precios, semanas faltantes o sin precio en 8 semanas | Es correcto: la página indica el motivo |
| La web pierde la conexión detrás del proxy | El proxy no reenvía WebSockets | Usa Caddy como arriba, o habilita WebSockets en tu proxy |
| `web` queda en "health: starting" | Arranca en menos de 30 s; si pasa más, Streamlit no levantó | `docker compose logs web` |

## Alternativa: proveedor gestionado sin Docker

En un proveedor con Python y PostgreSQL gestionado (Render, Railway, un VPS sin Docker):

1. Crea la base PostgreSQL 16 y define en el panel `AGRODASH_ENV=production` y `AGRODASH_DATABASE_URL=postgresql+psycopg://usuario:clave@host:5432/agrodash`. No la escribas en ningún archivo del repositorio.
2. Comando de construcción: `uv sync --frozen --no-dev`.
3. Comando de inicio: `uv run agrodash migrate && uv run agrodash web --port $PORT`.
4. Tarea programada semanal: `uv run agrodash migrate && uv run agrodash ingest --drive && uv run agrodash forecast`.
5. Activa las copias automáticas diarias del proveedor.

La base de producción debe cargarse desde cero con la versión actual: una base cargada con versiones anteriores conserva factores de unidad viejos.
