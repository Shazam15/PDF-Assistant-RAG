import pytest

from app.config import Settings


def test_production_settings_require_secret_key():
    with pytest.raises(ValueError):
        Settings(ENVIRONMENT="production", SECRET_KEY="")
