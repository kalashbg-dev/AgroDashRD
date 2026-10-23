# ETL data cleanup

Objective: fix the ingestion problems found by the raw-data audit (`~/Downloads/data/auditoriaetl.html`), one problem at a time, verified against a baseline load.
Branch: `feat/etl-cleanup` · Delivery: ask-on-risk · Test runner: `uv run pytest` (pytest).

## Specs

- **S1 (G2, retail-average block inside the wholesale report)**: wholesale report rows 39141-39181 (Quinta, Julio 2025) hold the average of the six retail markets for that week. User: "no lo descartes"; chose option "A" (L5): the whole block goes to quarantine with its own detail, kept as read, no product or price created.
- **S2 (wholesale header)**: the wholesale report has no header row. User: "hay que agregarlos en cuestion. Producto, Rubro, Categoria, etc... el mismo informe dice como debe ir". Assumption: the ETL names the columns by position, as the retail header does; the source file is not edited.
- **S3 (wholesale market)**: User: "los dos mercados que deben tomarse en referencia para esos valores del dataset mayorista es al MERCADOM y al Mercaddo Nuevo". Today every wholesale price is stored under the market "Mercado Nuevo". How to represent both: pending.
- **S4 (fifth week)**: context, no change: "se cuadra un mes en 4 semanas pero en ocasiones se cuadra en 5 ya que el mes puede iniciar a mitad de una semana". The current period model (published weeks 1-5) already follows it.
- Order of the remaining problems (from the audit): G5, G4, G3, G6, G1, G7, then M1-M7. Each gets its own spec when the user answers its question.

## Tasks

| ID | Specs | Route | Status | Evidence |
|---|---|---|---|---|
| T1 | S1 | inline | done | RED/GREEN `test_a_week_of_retail_averages_in_the_wholesale_report_is_quarantined_whole`, `test_a_wholesale_week_with_a_few_products_sold_by_the_unit_still_loads`; suite 424 passed; real reload L6 |
| T2 | S2 | inline | open | |
| T3 | S3 | inline | open, needs decision (stored market names are existing data: high-risk item 1) | |

## Log

- **L1** (2026-10-23, user, verbatim): "ok, hay que ir corrigiendo, ya tienes el informe completo en un html leelo y procedemos con las soluciones en orden de prioridad y de forma que se rompan menos cosas, la idea es ir solucionando con la mayor facilidad posible, vamos a ir recapitulando sobre cada uno de los probemas y los vamos a ir resolviendo, revisando, corroborando, uno a uno, empieza y haz preguntas, a cada probema hazme preguntas que te respondere con la respuesta mas contexto adicional que te sirve de ayuda para entender mejor el dataset y las demas situaciones existentes. vamos"
- **L2** (2026-10-23, user, verbatim): "si me ubicas donde esta, y a partir de ahora en cada pregunta debes hacer eso, ubicarme el problema asi te lo reviso y te doy una mejor respuesta"
- **L3** (2026-10-23, user, verbatim): "no lo descartes, sucede que este dataset es un compilado de varios analistas que introduce valores que llegan desde el mercado, por razones ajenas a nosotros y por el ciclo natural de las semanas y los dias, a veces dependiendo de en que dia se introduzcan los datos , se cuadra un mes en 4 semanas pero en ocasiones se cuadra en 5 ya que el mes puede iniciar a mitad de una semana, porque el mes tiende a tener 4 semanas y algo mas de duracion. Algo importante que tambien creo que revisaste en la auditoria y es que efectivamente el dataset de mayorista no tiene la primera fila con los encabezados, hay que agregarlos en cuestion. Producto, Rubro, Categoria, etc... el mismo informe dice como debe ir, el mayorista no tiene mercado porque lo que sucede en republica Dominicana es lo siguiente: Los mayoristas llegan generalmente al Merca Santo Domingo o MERCADOM y al MERCADO NUEVO o Mercado de la Duarte, en el resto de mercados y en ellos mismos se trabaja en precio minorista. Solo estos mercados recibien Precio Mayorista. Por eso para los Fines tomamos a Mercado Nuevo como referente Mayorista, aunque los dos mercados que deben tomarse en referencia para esos valores del dataset mayorista es al MERCADOM y al Mercaddo Nuevo."
- **L4** (2026-10-23, analysis): baseline load into a scratch SQLite with the current ETL: wholesale 40,234 prices / 418 quarantined; retail 249,928 / 176. The G2 block equals the mean of the six retail markets (Supermercado excluded) in the retail report, rows 44815-44922; July 2025 has no real wholesale Quinta. Open question to the user: how to keep the block.
- **L5** (2026-10-23, user, verbatim): "A"
- **L6** (2026-10-23, T1 evidence): rule in `repair.retail_average_rows`: a wholesale week block of 5+ rows where most units are lb/unidad/doc/litro is quarantined whole (`unexpected_layout`, detail "a week of retail averages ..."). Real reload vs baseline: wholesale 40,234 -> 40,196 prices (-38, the block), quarantine 418 -> 458 (+41 block rows, -1 Leche row already counted), products 349 -> 318 (31 junk products gone), retail unchanged (249,928). Only other difference: 124 retail prices keep their values but their product is now named "Cerdo pierna" (retail spelling) instead of the junk wholesale "Cerdo (Pierna)". Risk: none (no stored-data or schema change; retail path untouched, pinned by test).
