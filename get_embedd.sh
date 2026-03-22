#!/bin/bash
# Pull all embedding and reranking models from Ollama with all versions

# === Embedding Models ===

# nomic-embed-text - high-performing open embedding model
ollama pull nomic-embed-text:latest
ollama pull nomic-embed-text:v1.5
ollama pull nomic-embed-text:137m-v1.5-fp16

# mxbai-embed-large - state-of-the-art from mixedbread.ai
ollama pull mxbai-embed-large:latest
ollama pull mxbai-embed-large:335m

# bge-m3 - BAAI's multi-functional embedding model
ollama pull bge-m3:latest
ollama pull bge-m3:567m

# bge-large - BAAI's embedding model
ollama pull bge-large:latest
ollama pull bge-large:335m

# all-minilm - sentence-transformers embedding models
ollama pull all-minilm:latest
ollama pull all-minilm:22m
ollama pull all-minilm:33m

# snowflake-arctic-embed - Snowflake's text embedding suite
ollama pull snowflake-arctic-embed:latest
ollama pull snowflake-arctic-embed:22m
ollama pull snowflake-arctic-embed:33m
ollama pull snowflake-arctic-embed:110m
ollama pull snowflake-arctic-embed:137m
ollama pull snowflake-arctic-embed:335m

# snowflake-arctic-embed2 - Snowflake's multilingual embedding model
ollama pull snowflake-arctic-embed2:latest
ollama pull snowflake-arctic-embed2:568m

# qwen3-embedding - Alibaba's text embedding models
ollama pull qwen3-embedding:latest
ollama pull qwen3-embedding:0.6b
ollama pull qwen3-embedding:4b
ollama pull qwen3-embedding:8b

# embeddinggemma - Google's 300M parameter embedding model
ollama pull embeddinggemma:latest
ollama pull embeddinggemma:300m

# granite-embedding - IBM's embedding models
ollama pull granite-embedding:latest
ollama pull granite-embedding:30m
ollama pull granite-embedding:278m

# nomic-embed-text-v2-moe - multilingual MoE text embedding
ollama pull nomic-embed-text-v2-moe:latest

# paraphrase-multilingual - sentence-transformers multilingual model
ollama pull paraphrase-multilingual:latest
ollama pull paraphrase-multilingual:278m

# === Reranking Models ===

# Qwen3-Reranker series
ollama pull dengcao/Qwen3-Reranker-0.6B
ollama pull dengcao/Qwen3-Reranker-4B
ollama pull dengcao/Qwen3-Reranker-8B

# BGE reranker
ollama pull dengcao/bge-reranker-v2-m3
ollama pull bbjson/bge-reranker-base
ollama pull kopens/bge-reranker-large

# BCE reranker (NetEase)
ollama pull dengcao/bce-reranker-base_v1

# mxbai reranker
ollama pull rjmalagon/mxbai-rerank-large-v2