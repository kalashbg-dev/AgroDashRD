"""Product names the user approved as one product (audit G7, feature log L17-L18).

Each pair is (old name, canonical name) as the reports wrote them. The reports mark a
product's variety, color, size or grade inside its name with "/", "-", commas or
parentheses; these pairs differ only by a dropped grade word or a typo, so they are
the same product. Sizes and classes with their own prices stay separate products.
Plátano (Macho x Hembra) mediano is not here: both names priced the same weeks.
"""

APPROVED_MERGES: tuple[tuple[str, str], ...] = (
    ("Aguacate (Benny)", "Aguacate (Benny), primera"),
    ("Aguacate (Criollo)/ pequeño", "Aguacate (Criollo)/ primera/ pequeño"),
    ("Aguacate (Criollo)/ primera", "Aguacate (Criollo)"),
    ("Aguacate (Criollo)/grande", "Aguacate (Criollo)/ primera/ grande"),
    ("Aguacate (Criollo)/mediano", "Aguacate (Criollo)/ primera/ mediano"),
    ("Aguacate (Semil-34)/ grande", "Aguacate (Semíl-34), primera, grande"),
    ("Ajo (Importado)/ primera", "Ajo (Importado)"),
    ("Ají (Morrón)", "Ají (Morrón)/ primera"),
    ("Ají/ Cachucha/ verde", "Ají (Cachucha)/ verde/ primera"),
    ("Arroz Superior", "Arroz (Superior)/ primera"),
    ("Cebolla roja (Importada) primera, mediana", "Cebolla roja (Importada),mediana"),
    ("Cebolla roja (Importada)/ grande", "Cebolla roja (Importada)/ grande/ primera"),
    ("Cerdo (Pierna)", "Cerdo (Pierna), primera"),
    ("Cerdo Chuleta (Ahumada)", "Cerdo (Chuleta Ahumada)/ primera"),
    ("Cerdo banda", "Cerdo (Banda)/ primera"),
    ("Cerdo chuleta fresca", "Cerdo (Chuleta Fresca)/ primera"),
    ("Chinola/ amarilla/ grande/ primera", "Chinola (Amarilla), Grande"),
    ("Guandul (Verde en Vaina),", "Guandul (Verde en vaina)/ primera"),
    ("Guandul (Verde en grano)/ primera", "Guandul (Verde en grano)"),
    ("Lechosa (Red Lady), grande", "Lechosa/ (Red Lady)/ grande/ primera"),
    ("Lechosa/ (Maradol)/ grande", "Lechosa (Maradol)/ grande/ primera"),
    ("Lechosa/ (Maradol)/ mediana", "Lechosa (Maradol)/ mediana/ primera"),
    ("Limón (Persa)", "Limón (Persa)/ primera"),
    ("Melón (Cantaloupe), grande", "Melón/ (Cantaloupe)/ grande/ primera"),
    ("Naranja (Agria), mediana", "Naranja (Agria)/ primera/ mediana"),
    ("Naranja (Agria), mediana, verde", "Naranja (Agria), mediana, verde (primera)"),
    ("Naranja (Valencia)/ primera/ pequeña", "Naranja (Valencia), pequeña"),
    ("Piña (MD2), grande", "Piña/ (MD2)/ grande/ primera"),
    ("Piña/ (Cayena Lisa)", "Piña (Cayena Lisa), primera"),
    ("Plátano (FHIA - 20)/ primera", "Plátano (FHIA - 20)"),
    ("Plátano (FHIA - 21)", "Plátano (FHIA - 21)/ primera"),
    ("Plátano/ (FHIA - 20)/ grande", "Plátano (FHIA - 20), primera (grande)"),
    ("Pollo (Procesado)", "Pollo (Procesado)/ primera"),
    ("Pollo matado", "Pollo (Matado)/ primera"),
    ("Pollo vivo", "Pollo (Vivo)/ primera"),
    ("Res banda", "Res (Banda)/ primera"),
    ("Res bola", "Res (Bola)/ primera"),
    ("Res cadera", "Res (Cadera)/ primera"),
    ("Res pecho", "Res (Pecho)/ primera"),
    ("Res rotí", "Res (Rotí)/ primera"),
    ("Tomate (Bugalú)", "Tomate (Bugalú)/ primera"),
    ("Yautía (Amarilla)/ primera", "Yautía (Amarilla)"),
    ("Yautía (Blanca)", "Yautía (Blanca)/ primera"),
    ("Yuca/ (Encerada)", "Yuca (Encerada)/ primera"),
    ("Ñame (Mina)", "Ñame (Mina)/ primera"),
)
