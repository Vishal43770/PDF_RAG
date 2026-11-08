import fs from "fs/promises";
import fsSync from "fs";
import path from "path";
import readline from "readline/promises";
import { stdin as input, stdout as output } from "process";
import { pipeline } from "@xenova/transformers";
import Groq from "groq-sdk";
import dotenv from "dotenv";
// Use legacy build of pdfjs-dist for Node.js compatibility
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

dotenv.config();

// -------------------------------------------------------------
// Terminal Output Logger: Appends all terminal output to app.log with Indian Standard Time (IST)
// -------------------------------------------------------------
export function getISTTime() {
  return new Date().toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });
}

const LOG_FILE = "app.log";
// 'flags: a' ensures file is NEVER overwritten, always appended to the end
const logStream = fsSync.createWriteStream(LOG_FILE, { flags: "a" });
logStream.write(`\n\n=================================================================\n`);
logStream.write(`[SESSION STARTED]: ${getISTTime()} IST\n`);
logStream.write(`=================================================================\n`);

const originalStdoutWrite = process.stdout.write.bind(process.stdout);
const originalStderrWrite = process.stderr.write.bind(process.stderr);

process.stdout.write = (chunk, encoding, callback) => {
  logStream.write(chunk);
  return originalStdoutWrite(chunk, encoding, callback);
};

process.stderr.write = (chunk, encoding, callback) => {
  logStream.write(chunk);
  return originalStderrWrite(chunk, encoding, callback);
};

const CACHE_DIR = "./.cache";
const CACHE_FILE = path.join(CACHE_DIR, "embeddings_cache.json");
const ASSETS_DIR = "./asserts";

/**
 * =================================================================
 * RAG Core Engine: BGE Embeddings + Recursive Splitter + Groq LLM
 * =================================================================
 *
 * @param {string} [embeddingModel='Xenova/bge-large-en-v1.5'] - 1024-dim vector model
 * @param {string} [responseModel='openai/gpt-oss-20b'] - Groq LLM model
 */
export function groq(
  embeddingModel = "Xenova/bge-large-en-v1.5",
  responseModel = "openai/gpt-oss-20b"
) {
  const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });
  let embedder = null;

  // 1. Text to Vectors (1024 dimensions)
  async function getEmbedder() {
    if (!embedder) {
      embedder = await pipeline("feature-extraction", embeddingModel);
    }
    return embedder;
  }

  async function getEmbedding(text, isQuery = false) {
    const pipe = await getEmbedder();
    let inputText = text;
    // BGE models use an instruction prefix on queries for higher retrieval accuracy
    if (isQuery && embeddingModel.toLowerCase().includes("bge")) {
      inputText = `Represent this sentence for searching relevant passages: ${text}`;
    }
    const result = await pipe(inputText, { pooling: "mean", normalize: true });
    return Array.from(result.data);
  }

  // 2. Cosine Similarity
  function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
  }

  // 3. Groq LLM Generation
  async function generateResponse(question, context) {
    const messages = [
      {
        role: "system",
        content: `You are a helpful assistant. Use ONLY the provided context to answer the question accurately.\n\nContext:\n${context}`
      },
      { role: "user", content: question }
    ];

    const completion = await groqClient.chat.completions.create({
      model: responseModel,
      messages,
      temperature: 0.2
    });

    return completion.choices[0]?.message?.content || "";
  }

  // 4. Recursive Character Text Splitter (keeps paragraphs, sentences, and words intact)
  function chunkText(
    text,
    chunkSize = 1500,
    overlap = 250,
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
  ) {
    if (!text || text.trim().length === 0) return [];

    function splitOnSeparator(textToSplit, seps) {
      if (textToSplit.length <= chunkSize) {
        return [textToSplit];
      }
      if (seps.length === 0) {
        const chunks = [];
        for (let i = 0; i < textToSplit.length; i += chunkSize - overlap) {
          chunks.push(textToSplit.slice(i, i + chunkSize));
        }
        return chunks;
      }

      const sep = seps[0];
      const remainingSeps = seps.slice(1);
      const splits = sep === "" ? Array.from(textToSplit) : textToSplit.split(sep);

      const goodPieces = [];
      for (const piece of splits) {
        if (piece.length <= chunkSize) {
          goodPieces.push(piece);
        } else {
          goodPieces.push(...splitOnSeparator(piece, remainingSeps));
        }
      }

      // Group pieces into chunks up to chunkSize respecting overlap
      const chunks = [];
      let currentGroup = [];
      let currentLen = 0;

      for (const piece of goodPieces) {
        const trimmed = piece.trim();
        if (!trimmed) continue;
        const addedLen = currentGroup.length > 0 ? sep.length + trimmed.length : trimmed.length;

        if (currentLen + addedLen <= chunkSize) {
          currentGroup.push(trimmed);
          currentLen += addedLen;
        } else {
          if (currentGroup.length > 0) {
            chunks.push(currentGroup.join(sep));
          }
          let overlapLen = 0;
          const overlapPieces = [];
          for (let i = currentGroup.length - 1; i >= 0; i--) {
            if (overlapLen + currentGroup[i].length <= overlap) {
              overlapPieces.unshift(currentGroup[i]);
              overlapLen += currentGroup[i].length + sep.length;
            } else {
              break;
            }
          }
          currentGroup = [...overlapPieces, trimmed];
          currentLen = currentGroup.join(sep).length;
        }
      }

      if (currentGroup.length > 0) {
        chunks.push(currentGroup.join(sep));
      }

      return chunks;
    }

    const rawChunks = splitOnSeparator(text, separators);
    return rawChunks
      .map((chunk, idx) => ({
        id: idx + 1,
        text: chunk.trim()
      }))
      .filter((c) => c.text.length > 0);
  }

  // 5. Query Chunks
  async function query(question, chunks, topK = 3, minScore = 0.4) {
    console.log(`\n1. [Embed Question] Embedding: "${question}"`);
    const questionVec = await getEmbedding(question, true);

    console.log(`2. [Similarity Search] Scoring ${chunks.length} chunks...`);
    const scoredChunks = await Promise.all(
      chunks.map(async (chunk) => {
        const text = typeof chunk === "string" ? chunk : chunk.text;
        const chunkVec = await getEmbedding(text);
        const score = cosineSimilarity(questionVec, chunkVec);
        return {
          id: chunk.id || 0,
          text,
          score
        };
      })
    );

    const qualifyingChunks = scoredChunks.filter((c) => c.score >= minScore);
    qualifyingChunks.sort((a, b) => b.score - a.score);
    const topMatches = qualifyingChunks.slice(0, topK);

    console.log(`3. [Top Matches Found (Score >= ${minScore})]:`);
    if (topMatches.length === 0) {
      const highestScore = scoredChunks.length > 0 ? Math.max(...scoredChunks.map((c) => c.score)) : 0;
      console.log(`   - No chunks met threshold >= ${minScore}. Highest score was ${highestScore.toFixed(4)}.`);
      return {
        answer: `No context found with a similarity score >= ${minScore}.`,
        topMatches: []
      };
    }

    topMatches.forEach((m) => {
      console.log(`   - Chunk #${m.id} (Score: ${m.score.toFixed(4)}): "${m.text.slice(0, 70)}..."`);
    });

    const context = topMatches.map((m) => m.text).join("\n\n---\n\n");
    console.log(`4. [Groq Generation] Querying Groq (${responseModel})...\n`);
    const answer = await generateResponse(question, context);

    return { answer, topMatches };
  }

  return {
    embeddingModel,
    responseModel,
    getEmbedding,
    cosineSimilarity,
    chunkText,
    generateResponse,
    query
  };
}

/**
 * =================================================================
 * PDF Ingestion (pdfjs-dist)
 * =================================================================
 */
export async function extractPdfText(filePath) {
  const fileBuffer = await fs.readFile(filePath);
  const data = new Uint8Array(fileBuffer);

  const loadingTask = pdfjsLib.getDocument({
    data,
    useSystemFonts: true,
    disableFontFace: true
  });

  // Skip Table of Contents / Front Matter for specific PDFs
  let startPage = 1;
  const fileName = path.basename(filePath).toLowerCase();

  if (fileName.includes("postgresql-18")) {
    startPage = 41; // Skip first 40 pages of TOC
  } else if (fileName.includes("nist")) {
    startPage = 27; // Skip first 26 pages of TOC & front matter
  }

  const pdfDoc = await loadingTask.promise;
  const pageTexts = [];

  for (let pageNum = startPage; pageNum <= pdfDoc.numPages; pageNum++) {
    const page = await pdfDoc.getPage(pageNum);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      .map((item) => (item.str ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();

    if (pageText.length > 0) {
      pageTexts.push(pageText);
    }
  }

  return {
    text: pageTexts.join("\n\n"),
    numPages: pdfDoc.numPages
  };
}

/**
 * Load and extract text chunks from all PDF files in asserts/.
 */
async function loadDocumentsFromAssets(rag, chunkSize = 1500, overlap = 250) {
  const files = await fs.readdir(ASSETS_DIR);
  const pdfFiles = files.filter((f) => f.toLowerCase().endsWith(".pdf"));

  const allChunks = [];

  for (const file of pdfFiles) {
    const filePath = path.join(ASSETS_DIR, file);
    console.log(`Reading PDF with pdfjs-dist: ${file}...`);

    const { text, numPages } = await extractPdfText(filePath);
    console.log(`Parsed ${file} (${numPages} pages)`);

    const rawChunks = rag.chunkText(text, chunkSize, overlap);
    console.log(`Created ${rawChunks.length} chunks from ${file}`);

    for (let i = 0; i < rawChunks.length; i++) {
      allChunks.push({
        id: `${file}-chunk-${i + 1}`,
        source: file,
        text: rawChunks[i].text || rawChunks[i]
      });
    }
  }

  return allChunks;
}

/**
 * =================================================================
 * Checkpointing Vector Store (Auto-save on Ctrl+C and resume)
 * =================================================================
 */
async function getOrBuildVectorStore(rag, documents, checkpointInterval = 25) {
  await fs.mkdir(CACHE_DIR, { recursive: true });

  const cachedMap = new Map();

  try {
    const cacheRaw = await fs.readFile(CACHE_FILE, "utf-8");
    const cache = JSON.parse(cacheRaw);
    if (cache.model === rag.embeddingModel && Array.isArray(cache.documents)) {
      for (const item of cache.documents) {
        if (item && item.id && item.vector) {
          cachedMap.set(item.id, item);
        }
      }

      if (cachedMap.size === documents.length) {
        console.log(`Loaded all ${cachedMap.size} cached embeddings from disk.`);
        return cache.documents;
      }
    }
  } catch {
    // Cache miss or invalid, continue
  }

  if (cachedMap.size > 0) {
    console.log(`Checkpoint found! Resuming from chunk ${cachedMap.size + 1}/${documents.length} (${cachedMap.size} already cached)...`);
  } else {
    console.log(`Generating embeddings for ${documents.length} chunks using "${rag.embeddingModel}"...`);
  }

  const vectorStore = [];
  let newlyEmbeddedCount = 0;

  async function saveCacheToDisk() {
    await fs.writeFile(
      CACHE_FILE,
      JSON.stringify({ model: rag.embeddingModel, documents: vectorStore }, null, 2)
    );
  }

  let isSavingOnExit = false;
  const handleExit = async () => {
    if (isSavingOnExit) return;
    isSavingOnExit = true;
    console.log(`\n\n[Checkpoint] Ctrl+C detected! Saving ${vectorStore.length}/${documents.length} completed chunks to disk...`);
    try {
      await saveCacheToDisk();
      console.log(`[Checkpoint] Safely saved to ${CACHE_FILE}. Resumes from chunk ${vectorStore.length + 1} next time!`);
    } catch (err) {
      console.error("[Checkpoint] Failed to save on exit:", err);
    }
    process.exit(0);
  };

  process.on("SIGINT", handleExit);
  process.on("SIGTERM", handleExit);

  try {
    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];

      if (cachedMap.has(doc.id)) {
        vectorStore.push(cachedMap.get(doc.id));
        continue;
      }

      const vector = await rag.getEmbedding(doc.text);
      vectorStore.push({
        ...doc,
        vector
      });
      newlyEmbeddedCount++;

      if (newlyEmbeddedCount % checkpointInterval === 0) {
        await saveCacheToDisk();
        console.log(`[Checkpoint] Auto-saved ${vectorStore.length}/${documents.length} chunks to disk...`);
      } else if ((i + 1) % 50 === 0 || i + 1 === documents.length) {
        console.log(`Embedded ${i + 1}/${documents.length} chunks...`);
      }
    }

    await saveCacheToDisk();
    console.log(`Saved vector cache to ${CACHE_FILE} (${vectorStore.length} total chunks).`);
  } finally {
    process.off("SIGINT", handleExit);
    process.off("SIGTERM", handleExit);
  }

  return vectorStore;
}

/**
 * =================================================================
 * Query Execution
 * =================================================================
 */
export async function askQuestion(rag, vectorStore, question, topK = 3, minScore = 0.35) {
  const queryVector = await rag.getEmbedding(question, true);

  const scored = vectorStore.map((doc) => ({
    ...doc,
    score: rag.cosineSimilarity(queryVector, doc.vector)
  }));

  const qualifyingChunks = scored.filter((doc) => doc.score >= minScore);
  qualifyingChunks.sort((a, b) => b.score - a.score);
  const topMatches = qualifyingChunks.slice(0, topK);

  if (topMatches.length === 0) {
    const highestScore = scored.length > 0 ? Math.max(...scored.map((d) => d.score)) : 0;
    return {
      answer: `No relevant content found with similarity score >= ${minScore} (Highest score found was ${(highestScore * 100).toFixed(1)}%).`,
      topMatches: []
    };
  }

  const context = topMatches.map(
    (m, i) => `[Source: ${m.source} | Match #${i + 1} | Score: ${(m.score * 100).toFixed(1)}%]\n${m.text}`
  ).join("\n\n---\n\n");

  const answer = await rag.generateResponse(question, context);
  return { answer, topMatches };
}

/**
 * =================================================================
 * Self-Test Routine (for `npm test`)
 * =================================================================
 */
async function runSelfTest() {
  console.log("=== STEP 5: COMPLETE RAG PIPELINE DEMO ===\n");
  const rag = groq("Xenova/bge-large-en-v1.5", "openai/gpt-oss-20b");

  const knowledgeDoc = `
PostgreSQL 18 brings enhanced parallel indexing and optimized query planner features for complex analytics.
It significantly reduces memory overhead when running join operations on multi-gigabyte tables.
In comparison, NIST Special Publication 800-53 rev 5 provides security and privacy controls for information systems.
Federal agencies follow NIST guidelines to ensure confidentiality, integrity, and availability of sensitive records.
For dinner tonight, Mario prepared a traditional spaghetti bolognese using fresh tomatoes, minced beef, and basil.
`;

  console.log("Splitting document with Recursive Character Text Splitter...");
  const chunks = rag.chunkText(knowledgeDoc, 180, 40);
  console.log(`Created ${chunks.length} chunks.\n`);

  const userQuestion = "What performance improvements does PostgreSQL 18 have?";
  console.log(`User Question: "${userQuestion}"`);

  const result = await rag.query(userQuestion, chunks, 3, 0.4);

  console.log("==========================================");
  console.log("            GROQ FINAL ANSWER             ");
  console.log("==========================================");
  console.log(result.answer);
  console.log("==========================================");
}

/**
 * =================================================================
 * Main CLI Entry Point
 * =================================================================
 */
async function main() {
  // If run with --test, execute self-test demo
  if (process.argv.includes("--test")) {
    await runSelfTest();
    return;
  }

  console.log("=== Groq RAG Engine (BGE-large 1024-dim + pdfjs-dist) ===");
  const rag = groq("Xenova/bge-large-en-v1.5", "openai/gpt-oss-20b");

  console.log(`Embedding Model: ${rag.embeddingModel} (1024 dims)`);
  console.log(`Groq Response Model: ${rag.responseModel}\n`);

  const docs = await loadDocumentsFromAssets(rag);
  const vectorStore = await getOrBuildVectorStore(rag, docs);

  // If question passed directly as CLI argument:
  const cliArgs = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
  const cliQuestion = cliArgs.join(" ").trim();

  if (cliQuestion) {
    console.log(`\nQuestion: "${cliQuestion}"`);
    console.log("Searching relevant chunks and querying Groq...");
    const { answer, topMatches } = await askQuestion(rag, vectorStore, cliQuestion);

    console.log("\n--- Top Sources ---");
    topMatches.forEach((m, idx) => {
      console.log(`[${idx + 1}] ${m.source} (Similarity: ${(m.score * 100).toFixed(1)}%)`);
    });

    console.log("\n--- Answer from Groq ---");
    console.log(answer);
    console.log("\n-------------------------\n");
    return;
  }

  // Interactive chat session
  const rl = readline.createInterface({ input, output });
  console.log("\nReady! Ask questions about your PDF documents (or type 'exit' to quit):\n");

  while (true) {
    const query = await rl.question("Question: ");
    if (!query || query.trim().toLowerCase() === "exit") {
      logStream.write(`[${getISTTime()} IST] [SESSION ENDED]\n`);
      break;
    }
    logStream.write(`[${getISTTime()} IST] [USER QUESTION]: ${query}\n`);

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
