FROM ubuntu:latest
RUN apt-get update && apt install -y curl && apt install -y systemd
EXPOSE 11434
RUN curl -fsSL https://ollama.com/install.sh | sh
CMD ["ollama","serve"]

