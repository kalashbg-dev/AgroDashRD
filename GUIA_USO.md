# Guía de uso de AgroDashRD

AgroDashRD muestra los precios agrícolas oficiales de la República Dominicana: cuánto cuesta hoy cada producto, en qué meses suele estar caro o barato, dónde conviene comprar o vender y qué esperar en las próximas semanas y meses. Para leer no hace falta cuenta.

Todas las páginas públicas empiezan igual: elige **tipo de precio** (minorista o mayorista), **rubro** y **producto**. Los productos que los informes dejaron de publicar aparecen con "(hasta mm/aaaa)".

## Para todos

| Página | Úsala para |
|---|---|
| Precios | Ver el último precio, su historia, compararlo con otros productos y ver qué sube o baja más |
| Pronóstico | Saber qué precio esperar en 4 semanas, 3 meses o 12 meses |
| Temporadas | Saber en qué meses el producto suele estar caro o barato |
| Mercados | Saber dónde está más barato, cuánto se encarece del mayorista al minorista y cuánto vale una cosecha |

### Precios

- **Producto:** el último precio en cada mercado, con la fecha y el cambio frente a cuatro semanas antes. Abajo, la historia de hasta tres mercados en el periodo que elijas (6 meses a todo el historial). "Descargar historia (CSV)" baja esos datos para Excel.
- **Comparar:** agrega hasta tres productos más. Cada línea es el promedio de los mercados que publican ese producto.
- **Mayores cambios:** los productos que más subieron y más bajaron en cuatro semanas, de todos los rubros o de uno. "Descargar todos (CSV)" baja la lista completa.

### Pronóstico

1. Elige el mercado y el **plazo**: 4 semanas, 3 meses o 12 meses.
2. Arriba aparece el precio esperado al final del plazo y sus rangos: con 80 % de probabilidad el precio real cae en el primero, con 95 % en el segundo.
3. La tabla muestra cada semana o mes; en 3 y 12 meses, la columna **Temporada** dice si ese mes suele tener precio alto o bajo.

El valor pronosticado es el último precio publicado: en las pruebas con la historia oficial ningún modelo lo superó, a ningún plazo. Lo útil es el rango. Los plazos de 3 y 12 meses necesitan tres años de precios; sin ellos, la página lo indica.

Si un producto dice "sin pronóstico", la página explica por qué: tiene menos de dos años de precios, le faltan demasiadas semanas, no tiene precio en las últimas ocho semanas o fue retirado.

### Temporadas

- **Precio alto / Precio bajo:** los meses en que el producto suele estar 10 % o más por encima o por debajo del promedio del año.
- **Mes a mes:** cada barra es un mes; 100 es el promedio del año.
- **Año por año:** el mismo índice para cada año, para ver si el patrón se repite.
- **Todo el rubro:** todos los productos del rubro, mes a mes, para comparar temporadas.

Con menos de tres años completos, la página avisa que el patrón puede ser casualidad.

### Mercados

- **Dónde está más barato:** los mercados con precio reciente, del más barato al más caro; la línea punteada es el promedio.
- **Mayorista y minorista:** el precio mensual de cada tipo en los últimos tres años y cuánto más cuesta al por menor.
- **¿Cuánto vale mi cosecha?:** escribe la cantidad y la unidad (libras, quintales, kilogramos, unidades, cientos, docenas, litros o galones) y verás su valor en cada mercado. Quien produce suele recibir menos: el intermediario y el transporte se llevan una parte.

## Para técnicos

**Reportar precio** registra un precio observado en campo: producto, mercado, unidad, precio y fecha (no futura). El reporte queda **pendiente** hasta que un administrador lo valide; si se aleja más de 50 % de la mediana oficial reciente, se marca para revisión.

## Para administradores

### Administración

| Pestaña | Qué hacer |
|---|---|
| Cargar datos | Subir un informe (`.csv`, `.xlsx`, `.xls`, hasta 50 MB) o cargar desde Google Drive. "Últimas cargas" muestra cada carga con su estado, precios cargados y filas en cuarentena |
| Reportes | Validar o rechazar los reportes de campo pendientes |
| Cuarentena | Revisar las filas que no se cargaron y su motivo |
| Usuarios | Crear cuentas, desactivarlas, reactivarlas y desbloquearlas. Un ADMIN gestiona técnicos; un SUPERADMIN gestiona todas |

Una cuenta desactivada pierde el acceso en su siguiente clic, aunque tenga la sesión abierta.

### Catálogo

| Pestaña | Qué hacer |
|---|---|
| Repetidos | Unir el mismo producto publicado con nombres distintos según la época. Los grupos **seguros** (mismas palabras) se unen con un botón; los **sugeridos** (difieren en un "primera" que los informes viejos omitían o en una letra) se revisan uno por uno. Toda la historia queda en el nombre más reciente y los demás se retiran |
| Renombres | Productos cuyos precios terminan cuando empieza otro de nombre parecido; fusiónalos si son el mismo |
| Nombres | Confirmar a qué producto se asignó cada nombre de los informes |
| Productos | Renombrar, retirar o reactivar un producto |
| Mercados | Retirar o reactivar un mercado |

Al unir productos:

- Si los dos tienen precio oficial en la misma semana y mercado, la unión se rechaza y el producto queda sin unir, con el aviso correspondiente.
- No se puede unir con un producto retirado: reactívalo primero.
- Precios, Temporadas y Mercados muestran la historia unida al instante. El pronóstico se recalcula con ella en la siguiente tarea semanal; hasta entonces, el producto muestra el que tenía.
