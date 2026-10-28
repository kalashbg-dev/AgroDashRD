"""Split a report's product name into the properties it writes inside it.

The reports write a product as its base name, its variety in parentheses, and its color,
size and grade after "/" or commas: "Berenjena/ (Pompadur)/ morada oscura/ grande",
"Ají (Gustoso), verde, segunda". Spelling and gender vary ("mediana", "2da", "medinao").
"""

import re
from dataclasses import dataclass

from agrodash.ingestion.normalize import clean_name, fold

GRADES = {
    "primera": "primera", "primero": "primera", "1ra": "primera",
    "segunda": "segunda", "segundo": "segunda", "2da": "segunda",
    "tercera": "tercera", "tercero": "tercera", "3ra": "tercera",
}  # fmt: skip
SIZES = {
    "grande": "grande", "mediano": "mediano", "mediana": "mediano", "medinao": "mediano",
    "pequeno": "pequeño", "pequena": "pequeño", "fino": "fino", "fno": "fino",
    "grueso": "grueso", "corta": "corto", "corto": "corto", "larga": "largo", "largo": "largo",
}  # fmt: skip
COLORS = {
    "verde": "verde", "maduro": "maduro", "madura": "maduro", "amarillo": "amarillo",
    "amarilla": "amarillo", "rojo": "rojo", "roja": "rojo", "blanco": "blanco",
    "blanca": "blanco", "negro": "negro", "negra": "negro", "pinta": "pinto",
    "morada oscura": "morado oscuro", "colores": "colores",
}  # fmt: skip
# Our own marker for a second listing of the same name (repair.repair_rows).
ROW_MARK = re.compile(r"\s*\(fila \d+\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class NameParts:
    base: str
    variety: str | None = None
    color: str | None = None
    size: str | None = None
    grade: str | None = None


def name_parts(name: str) -> NameParts:
    """Base name, variety, color, size and grade a product name states."""
    text = ROW_MARK.sub("", clean_name(name))
    found: dict[str, str] = {}
    varieties: list[str] = []

    def classify(piece: str) -> bool:
        """Record a color, size or grade word; False when the piece is none of them."""
        folded = fold(piece)
        for field, words in (("grade", GRADES), ("size", SIZES), ("color", COLORS)):
            if folded in words:
                found.setdefault(field, words[folded])
                return True
        return False

    pieces = [clean_name(p) for p in re.split(r"[/,]", re.sub(r"\([^)]*\)", " ", text))]
    pieces = [piece for piece in pieces if piece]
    first, *rest = pieces[0].split(" ", 1) if pieces else [""]
    # The base is the first word; what follows it is a variety or a property written
    # without parentheses ("Mango Keitt", "Cebolla roja", "Plátano (Maduro) Grande").
    # A parenthesis names the variety whole ("Francés Largo", "Bilin negra").
    loose = [(piece, False) for piece in rest]
    tagged = [(inside, True) for inside in re.findall(r"\(([^)]*)\)", text)]
    for piece, parenthesized in [*loose, *tagged, *[(p, False) for p in pieces[1:]]]:
        piece = clean_name(piece)
        if not piece or classify(piece):
            continue
        if parenthesized:
            varieties.append(piece)
            continue
        variety = " ".join(word for word in piece.split() if not classify(word))
        if variety:
            varieties.append(variety)
    return NameParts(
        base=first,
        variety=" ".join(varieties) or None,
        color=found.get("color"),
        size=found.get("size"),
        grade=found.get("grade"),
    )
