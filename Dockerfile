# 1. Base Image kiválasztása
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Munkakönyvtár beállítása a konténeren belül
WORKDIR /app

# 3. Rendszerszintű függőségek telepítése
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
libgl1-mesa-glx \
libglib2.0-0 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# 4. Python függőségek másolása és telepítése
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 5. Alkalmazás kódjának és a futtató scriptnek a másolása
COPY ./src .
# 6. A futtató script végrehajthatóvá tétele
RUN chmod +x run.sh
# 7. Alapértelmezett parancs a konténer indításakor
CMD ["bash", "run.sh"]
