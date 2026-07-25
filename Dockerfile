# Usamos una imagen oficial de Python ligera
FROM python:3.11-slim

# Evitamos que Python escriba archivos .pyc y forzamos el volcado de logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establecemos el directorio de trabajo
WORKDIR /app

# Instalamos las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto (Cloud Run usa el 8080 por defecto)
EXPOSE 8080

# Iniciamos el servidor de producción Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
