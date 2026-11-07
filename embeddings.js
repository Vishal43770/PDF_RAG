import Groq from "groq-sdk";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

dotenv.config();

/**
 * Cosine similarity between two numeric vectors.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number}
 */
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Creates a Groq RAG helper combining an embedding model with a Groq response model.
 * 
 * Note: Groq specializes in ultra-fast LLM text generation and does NOT host native
 * embedding models. This function pairs an open-source embedding model (via @xenova/transformers)
 * for vector search with Groq's high-speed response models (e.g., llama-3.3-70b-versatile).
 *
 * @param {string|object} embeddingModelOrConfig - Model name (e.g., 'Xenova/all-MiniLM-L6-v2') or config object.
 * @param {string} [responseModelParam] - Groq model for response (e.g., 'llama-3.3-70b-versatile').
 * @returns {object} Pipeline instance with getEmbedding, generateResponse, and query methods.
 */
export function groq(embeddingModelOrConfig, responseModelParam) {
  let embeddingModel = "Xenova/all-MiniLM-L6-v2";
  let responseModel = "openai/gpt-oss-20b";
  let apiKey = process.env.GROQ_API_KEY;

  if (typeof embeddingModelOrConfig === "object" && embeddingModelOrConfig !== null) {
    if (embeddingModelOrConfig.embeddingModel) embeddingModel = embeddingModelOrConfig.embeddingModel;
    if (embeddingModelOrConfig.responseModel) responseModel = embeddingModelOrConfig.responseModel;
    if (embeddingModelOrConfig.apiKey) apiKey = embeddingModelOrConfig.apiKey;
  } else {
    if (embeddingModelOrConfig) embeddingModel = embeddingModelOrConfig;
    if (responseModelParam) responseModel = responseModelParam;
  }

  const groqClient = new Groq({ apiKey });
  let embedderPromise = null;

  // Lazily load the embedding pipeline
  async function getEmbedder() {
    if (!embedderPromise) {
      embedderPromise = pipeline("feature-extraction", embeddingModel);
    }
    return embedderPromise;
  }

  /**
   * Generates a normalized embedding vector for the given text.
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async function getEmbedding(text) {
    const embedder = await getEmbedder();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }

  /**
   * Generates a completion from Groq using the retrieved context.
   * @param {string} prompt - User question or prompt
   * @param {string|string[]} [context] - Context retrieved from documents
   * @param {object} [options] - Optional Groq chat completion settings
   * @returns {Promise<string>}
   */
  async function generateResponse(prompt, context = "", options = {}) {
    const contextText = Array.isArray(context) ? context.join("\n\n") : context;

    const messages = [
      {
        role: "system",
        content: contextText
          ? `You are a helpful assistant. Use the following context to accurately answer the user question:\n\n${contextText}`
          : "You are a helpful assistant."
      },
      {
        role: "user",
        content: prompt
      }
    ];

    const completion = await groqClient.chat.completions.create({
      model: responseModel,
      messages,
      temperature: options.temperature ?? 0.2,
      max_completion_tokens: options.maxTokens ?? 1024,
      ...options
    });

    return completion.choices[0]?.message?.content || "";
  }

  /**
   * Performs an end-to-end RAG query over a list of documents.
   * @param {string} queryText - The user's question
   * @param {Array<string|{text: string}>} documents - Knowledge base documents
   * @param {number} [topK=3] - Number of top relevant chunks to include
   * @returns {Promise<{ answer: string, retrieved: Array<{ text: string, score: number }> }>}
   */
  async function query(queryText, documents, topK = 3) {
    if (!documents || documents.length === 0) {
      const answer = await generateResponse(queryText);
      return { answer, retrieved: [] };
    }

    const queryVector = await getEmbedding(queryText);

    // Score documents by cosine similarity
    const scoredDocs = await Promise.all(
      documents.map(async (doc) => {
        const text = typeof doc === "string" ? doc : doc.text;
        const docVector = await getEmbedding(text);
        const score = cosineSimilarity(queryVector, docVector);
        return { text, score };
      })
    );

    scoredDocs.sort((a, b) => b.score - a.score);
    const topMatches = scoredDocs.slice(0, topK);
    const context = topMatches.map((item) => item.text);

    const answer = await generateResponse(queryText, context);
    return {
      answer,
      retrieved: topMatches
    };
  }

  return {
    embeddingModel,
    responseModel,
    getEmbedding,
    generateResponse,
    query,
    cosineSimilarity
  };
}

// Example usage test when run directly:
if (process.argv[1] && process.argv[1].endsWith("embeddings.js")) {
  (async () => {
    try {
      console.log("Initializing Groq RAG helper...");

      // Pass embedding model and response model
      const rag = groq("Xenova/all-MiniLM-L6-v2", "openai/gpt-oss-20b");

      const documents = [
        "PostgreSQL 18 introduces enhanced parallel indexing and improved query plan optimizations.",
        "NIST Special Publication 800-53 rev 5 provides a catalog of security and privacy controls for information systems.",
        "Groq LPUs deliver ultra-low latency token generation for open-weight models like Llama 3."
      ];

      const question = "What does NIST SP 800-53 focus on?";
      console.log(`\nQuery: "${question}"`);

      console.log("Retrieving relevant context and generating response with Groq...");
      const result = await rag.query(question, documents);

      console.log("\nRetrieved Context Chunks:");
      result.retrieved.forEach((chunk, i) => {
        console.log(`[${i + 1}] (Score: ${chunk.score.toFixed(4)}): ${chunk.text}`);
      });

      console.log("\nResponse from Groq:\n" + result.answer);
    } catch (err) {
      console.error("Error running RAG pipeline:", err.message);
    }
  })();
}

