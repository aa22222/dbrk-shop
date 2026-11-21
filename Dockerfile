FROM rust:1.91-slim-bookworm AS build
WORKDIR /src
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs && cargo build --locked --release -j4
COPY src ./src
RUN touch src/main.rs && cargo build --locked --release -j4
FROM debian:bookworm-slim
COPY --from=build /src/target/release/shopping-sessions /app/sessions
ENV DATA_DIR=/data OUTPUT_DIR=/output NCPU=8
USER 1000
ENTRYPOINT ["/app/sessions"]
