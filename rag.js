import fs from "fs/promises";
import path from "path";
import readline from "readline/promises";
import { stdin as input, stdout as output } from "process";
import pdfParse from "pdf-parse";
import { groq } from "./embeddings.js";

const CACHE_DIR = "./.cache";
const CACHE_FILE = path.join(CACHE_DIR, "embeddings_cache.json");
const ASSETS_DIR = "./asserts";

/**
 * Split text into overlapping chunks.
 */
function chunkText(text, chunkSize = 1000, overlap = 150) {
  const cleaned = text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  const chunks = [];
  let start = 0;

  while (start < cleaned.length) {
    const end = Math.min(start + chunkSize, cleaned.length);
    const chunk = cleaned.slice(start, end).trim();
    if (chunk.length > 50) {
      chunks.push(chunk);
    }
    start += chunkSize - overlap;
  }
  return chunks;
}

/**
 * Load and extract text chunks from all PDF files in asserts/.
 */
async function loadDocumentsFromAssets() {
  const files = await fs.readdir(ASSETS_DIR);
  const pdfFiles = files.filter((f) => f.toLowerCase().endsWith(".pdf"));

  const allChunks = [];

  for (const file of pdfFiles) {
    const filePath = path.join(ASSETS_DIR, file);
    console.log(`Reading PDF: ${file}...`);
    const dataBuffer = await fs.readFile(filePath);
    const pdfData = await pdfParse(dataBuffer);

    console.log(`Parsed ${file} (${pdfData.numpages} pages)`);
    const fileChunks = chunkText(pdfData.text);
    console.log(`Created ${fileChunks.length} chunks from ${file}`);

    for (let i = 0; i < fileChunks.length; i++) {
      allChunks.push({
        id: `${file}-chunk-${i + 1}`,
        source: file,
        text: fileChunks[i]
      });
    }
  }

  return allChunks;
}

/**
 * Load cached embeddings or generate new ones.
 */
async function getOrBuildVectorStore(rag, documents) {
  await fs.mkdir(CACHE_DIR, { recursive: true });

  try {
    const cacheRaw = await fs.readFile(CACHE_FILE, "utf-8");
    const cache = JSON.parse(cacheRaw);
    if (cache.model === rag.embeddingModel && cache.documents?.length === documents.length) {
      console.log(`Loaded ${cache.documents.length} cached embeddings from disk.`);
      return cache.documents;
    }
  } catch {
    // Cache miss or invalid, continue to build
  }

  console.log(`Generating embeddings for ${documents.length} chunks using "${rag.embeddingModel}"...`);
  const vectorStore = [];

  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    const vector = await rag.getEmbedding(doc.text);
    vectorStore.push({
      ...doc,
      vector
    });

    if ((i + 1) % 50 === 0 || i + 1 === documents.length) {
      console.log(`Embedded ${i + 1}/${documents.length} chunks...`);
    }
  }

  await fs.writeFile(
    CACHE_FILE,
    JSON.stringify({ model: rag.embeddingModel, documents: vectorStore }, null, 2)
  );
  console.log(`Saved vector cache to ${CACHE_FILE}`);

  return vectorStore;
}

/**
 * Query the vector store and call Groq for response.
 */
async function askQuestion(rag, vectorStore, question, topK = 4) {
  const queryVector = await rag.getEmbedding(question);

  const scored = vectorStore.map((doc) => ({
    ...doc,
    score: rag.cosineSimilarity(queryVector, doc.vector)
  }));

  scored.sort((a, b) => b.score - a.score);
  const topMatches = scored.slice(0, topK);

  const context = topMatches.map(
    (m, i) => `[Source: ${m.source} | Match #${i + 1}]\n${m.text}`
  );

  const answer = await rag.generateResponse(question, context);

  return { answer, topMatches };
}

async function main() {
  console.log("=== Groq RAG Engine ===");
  // Initialize with embedding model and response model
  const rag = groq("Xenova/all-MiniLM-L6-v2", "openai/gpt-oss-20b");

  console.log(`Embedding Model: ${rag.embeddingModel}`);
  console.log(`Groq Response Model: ${rag.responseModel}\n`);

  // Load and index documents
  const docs = await loadDocumentsFromAssets();
  const vectorStore = await getOrBuildVectorStore(rag, docs);

  const rl = readline.createInterface({ input, output });

  console.log("\nReady! Ask questions about your PDF documents (or type 'exit' to quit):\n");

  while (true) {
    const query = await rl.question("Question: ");
    if (!query || query.trim().toLowerCase() === "exit") {
      break;
    }

    console.log("\nSearching relevant chunks and querying Groq...");
    const { answer, topMatches } = await askQuestion(rag, vectorStore, query.trim());

    console.log("\n--- Top Sources ---");
    topMatches.forEach((m, idx) => {
      console.log(`[${idx + 1}] ${m.source} (Similarity: ${(m.score * 100).toFixed(1)}%)`);
    });

    console.log("\n--- Answer from Groq ---");
    console.log(answer);
    console.log("\n-------------------------\n");
  }

  rl.close();
}

if (process.argv[1] && process.argv[1].endsWith("rag.js")) {
  main().catch((err) => console.error("Error:", err));
}

