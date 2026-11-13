# Proxmino web app. Build from the project folder: docker compose -f deploy/docker-compose.yml up -d --build
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROXMINO_DATA=/data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY proxmino proxmino
COPY web web
COPY crew crew
COPY market market
COPY third_party third_party
COPY config.yaml LICENSE THIRD_PARTY_NOTICES.md ./
RUN useradd -m -u 1000 proxmino && mkdir -p /data /home/proxmino/.proxmino && chown -R proxmino /data /home/proxmino
USER proxmino
EXPOSE 3000
CMD ["python", "-m", "proxmino", "serve", "--host", "0.0.0.0", "--port", "3000"]
