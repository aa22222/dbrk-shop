FROM python:3.13-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11==3.0.1
COPY reader_common.hpp reader.cpp /src/
RUN g++ -std=c++20 -O3 -march=x86-64-v3 -mtune=cascadelake -DNDEBUG -pthread -shared -fPIC $(python -m pybind11 --includes) /src/reader.cpp -o /src/shop_reader$(python-config --extension-suffix)

FROM python:3.13-slim-bookworm
RUN pip install --no-cache-dir polars==2.1.0 numpy==2.3.4
COPY --from=build /src/shop_reader*.so /app/
COPY pipeline.py /app/
ENV DATA_DIR=/data OUTPUT_DIR=/output NCPU=8 PYTHONDONTWRITEBYTECODE=1
USER 1000
ENTRYPOINT ["python","/app/pipeline.py"]
