FROM python:latest
WORKDIR /workdir
COPY . .
RUN curl -fsSL https://git.io/shellspec | sh -s -- --yes
ENV PATH="/root/.local/lib/shellspec:$PATH"