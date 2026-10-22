"""Spanish wording for the domain's error messages, which stay in English for the CLI."""

import re

_EXACT = {
    "Enter a valid email address": "Escribe un correo válido.",
    "Price must be greater than zero": "El precio debe ser mayor que cero.",
    "The observation date cannot be in the future": "La fecha no puede ser futura.",
    "This price is not a pending field report": "Este reporte ya fue revisado.",
    "Unknown account": "La cuenta no existe.",
    "Unknown product, market or unit": "Producto, mercado o unidad desconocidos.",
    "You cannot deactivate your own account": "No puedes desactivar tu propia cuenta.",
    "Your account cannot report prices": "Tu cuenta no puede reportar precios.",
    "Your account cannot review reports": "Tu cuenta no puede revisar reportes.",
    "Your account cannot change the catalog": "Tu cuenta no puede cambiar el catálogo.",
    "Your account cannot manage accounts": "Tu cuenta no puede gestionar cuentas.",
    "A product cannot be merged into itself": "Un producto no se puede fusionar consigo mismo.",
    "The name cannot be empty": "El nombre no puede estar vacío.",
}
_UNITS = {"lb": "libra", "unit": "unidad", "l": "litro"}
_PATTERNS = [
    (
        r"Password must have at least (\d+) characters",
        "La contraseña debe tener al menos {0} caracteres.",
    ),
    (r"You cannot create (\w+) accounts", "No puedes crear cuentas {0}."),
    (r"You cannot manage (\w+) accounts", "No puedes gestionar cuentas {0}."),
    (r"(\S+@\S+) is already registered", "{0} ya está registrado."),
    (r"A product named ('.+') already exists", "Ya existe un producto llamado {0}."),
    (
        r"(\d+) weeks? where both products have an official price; fix or delete one of them first",
        "Los dos productos tienen precio oficial en {0} semana(s) iguales; corrige o elimina "
        "uno de esos precios primero.",
    ),
    (
        r".+ (?:retail|wholesale) prices are per (\w+); choose a unit in \w+",
        "Esta serie se mide por {unit}: elige una unidad en {unit}.",
    ),
]


def spanish(message: str) -> str:
    if message in _EXACT:
        return _EXACT[message]
    for pattern, wording in _PATTERNS:
        match = re.fullmatch(pattern, message)
        if match:
            return wording.format(*match.groups(), unit=_UNITS.get(match.group(1), match.group(1)))
    return message
