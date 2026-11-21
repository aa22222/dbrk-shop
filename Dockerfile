FROM python:3.13-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --only-binary=:all: pyarrow==21.0.0
COPY native/pipeline.cpp native/output.hpp /src/
RUN set -eu; \
    arrow="$(python -c 'import pyarrow as p; p.create_library_symlinks(); print(p.get_library_dirs()[0])')"; \
    mkdir -p /app/lib; \
    g++ -std=c++20 -O3 -march=x86-64-v3 -mtune=cascadelake -DNDEBUG -pthread -I"$arrow/include" /src/pipeline.cpp \
        -L"$arrow" -Wl,-rpath-link,"$arrow" -lparquet -larrow -o /app/sessions; \
    LD_LIBRARY_PATH="$arrow" ldd /app/sessions > /tmp/linked; \
    awk -v dir="$arrow/" 'index($3, dir) == 1 {print $3}' /tmp/linked | xargs -r cp -Lt /app/lib/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /app /app
ENV LD_LIBRARY_PATH=/app/lib NCPU=8 DATA_DIR=/data OUTPUT_DIR=/output
RUN ldd /app/sessions > /tmp/deps && ! grep -q 'not found' /tmp/deps && rm /tmp/deps
USER 1000
ENTRYPOINT ["/app/sessions"]
