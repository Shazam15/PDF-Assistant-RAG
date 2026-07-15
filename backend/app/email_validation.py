from __future__ import annotations

import re

SPECIAL_CHAR_PATTERN = re.compile(r'[!@#$%^&*(),.?":{}|<>\-_=+\[\]\\;/\'`~]')

DOMINIO_PERMITIDO = "utp.ac.pa"

def validate_email(email: str) -> None:
    # 1. Limpieza básica: quitar espacios y pasar a minúsculas
    email_limpio = email.strip().lower()
    
    # 2. Expresión regular para validar estructura estándar Y el dominio exacto
    # Explicación del regex:
    # ^[a-zA-Z0-9._%+-]+  -> Nombre de usuario (letras, números, puntos, etc.)
    # @                   -> El símbolo arroba obligatorio
    # {re.escape(DOMINIO_PERMITIDO)}$ -> El dominio exacto al final de la cadena
    regex_patron = rf"^[a-zA-Z0-9._%+-]+@{re.escape(DOMINIO_PERMITIDO)}$"
    
    # 3. Validación
    if not re.match(regex_patron, email_limpio):
        # Levantamos un error con un mensaje claro
        raise ValueError(f"Correo no válido. Debe usar un correo institucional (@{DOMINIO_PERMITIDO}).")