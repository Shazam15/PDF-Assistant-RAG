import pytest

from app.password_validation import validate_email



def test_validate_email_accepts_valid_institutional_email():
    # Debería pasar sin lanzar ninguna excepción
    validate_email("estudiante.ejemplo@utp.ac.pa")


def test_validate_email_accepts_uppercase_and_spaces():
    # Debería limpiar los espacios y normalizar las mayúsculas internamente
    validate_email("  ESTUDIANTE.ejemplo@UTP.AC.PA  ")


def test_validate_email_rejects_wrong_domain():
    with pytest.raises(ValueError, match="Debe usar un correo institucional"):
        validate_email("usuario@gmail.com")


def test_validate_email_rejects_subdomains():
    # Evita que alguien use un subdominio malicioso o no autorizado
    with pytest.raises(ValueError, match="Debe usar un correo institucional"):
        validate_email("ataque@malicioso.utp.ac.pa")


def test_validate_email_rejects_phishing_domain_structure():
    # Evita el truco de poner el dominio seguro antes del arroba
    with pytest.raises(ValueError, match="Debe usar un correo institucional"):
        validate_email("utp.ac.pa@gmail.com")


def test_validate_email_rejects_invalid_syntax():
    # Casos donde la estructura del correo está rota
    with pytest.raises(ValueError, match="Correo no válido"):
        validate_email("usuario_sin_arroba.utp.ac.pa")
        
    with pytest.raises(ValueError, match="Correo no válido"):
        validate_email("@utp.ac.pa")

