from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "tradeeval-secret-key"
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "api",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "tradeeval_backend.urls"

TEMPLATES = []

WSGI_APPLICATION = "tradeeval_backend.wsgi.application"

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# MongoDB (used later)
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "tradeeval_db"