# PDF RAG with Groq and Local Embeddings

A Retrieval-Augmented Generation (RAG) system built in Node.js that pairs local Hugging Face embedding models via `@xenova/transformers` with ultra-fast LLM generation using the Groq API.

## Features

- **Local Embeddings**: Generates vector embeddings directly on CPU using `@xenova/transformers` (default: `Xenova/all-MiniLM-L6-v2`) without incurring external embedding API costs.
- **Ultra-Fast Generation**: Queries Groq's high-speed inference engine (e.g. `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `qwen/qwen3.8-27b`) for context-aware answers.
- **PDF Extraction**: Ingests and parses PDF documents using `pdf-parse`.
- **Vector Caching**: Embeddings are persisted locally in `.cache/` to avoid recomputing vectors across runs.
- **Configurable `groq()` helper**: A standalone function `groq(embeddingModel, responseModel)` to easily integrate embeddings and completions.

## Prerequisites

- Node.js (v18 or newer recommended, tested on Node v24)
- A Groq API Key from [console.groq.com](https://console.groq.com/)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vishal43770/PDF_RAG.git
   cd PDF_RAG
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   Copy `.env.example` to `.env` and insert your Groq API key:
   ```bash
   GROQ_API_KEY="your_groq_api_key_here"
   ```

## Usage

### Quick Test
Test the embedding and Groq completion function:
```bash
npm test
```

### Run PDF RAG
Place your PDF files inside the `asserts/` directory and run:
```bash
npm run rag
```

### Code Example
```javascript
import { groq } from "./embeddings.js";

// Initialize with embedding model and Groq response model
const rag = groq("Xenova/all-MiniLM-L6-v2", "openai/gpt-oss-20b");

const documents = [
  "NIST Special Publication 800-53 provides security controls.",
  "PostgreSQL 18 introduces enhanced parallel indexing."
];

const result = await rag.query("What does NIST 800-53 provide?", documents);
console.log(result.answer);
```
