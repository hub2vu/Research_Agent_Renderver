import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import type { IncomingMessage, ServerResponse } from 'http';

// Node colors file path
const NODE_COLORS_PATH = '/app/output/graph/node_colors.json';

// NeurIPS data paths
const NEURIPS_METADATA_PATH = '/app/data/embeddings_Neu/metadata.csv';
const NEURIPS_EMBEDDINGS_PATH = '/app/data/embeddings_Neu/embeddings.npy';

// Parse CSV helper
function parseCSV(content: string): Array<Record<string, string>> {
  const lines = content.split('\n');
  if (lines.length < 2) return [];

  // Parse header
  const headers = parseCSVLine(lines[0]);
  const results: Array<Record<string, string>> = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = parseCSVLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx] || '';
    });
    results.push(row);
  }

  return results;
}

// Parse a single CSV line handling quoted fields
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current);

  return result;
}

// Ensure graph directory exists
function ensureGraphDir() {
  const dir = path.dirname(NODE_COLORS_PATH);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// Read request body as JSON
function readJsonBody(req: IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'node-colors-api',
      configureServer(server) {
        // API: GET /api/node-colors - Load colors
        server.middlewares.use('/api/node-colors', async (req: IncomingMessage, res: ServerResponse, next) => {
          // Handle CORS preflight
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            if (req.method === 'GET') {
              // Load colors from file
              if (fs.existsSync(NODE_COLORS_PATH)) {
                const stat = fs.statSync(NODE_COLORS_PATH);
                const data = fs.readFileSync(NODE_COLORS_PATH, 'utf-8');
                res.setHeader('X-Last-Modified', stat.mtimeMs.toString());
                res.end(data);
              } else {
                res.end(JSON.stringify({ colors: {}, timestamp: 0 }));
              }
            } else if (req.method === 'POST') {
              // Save colors to file
              ensureGraphDir();
              const body = await readJsonBody(req);
              const data = {
                colors: body.colors || {},
                timestamp: Date.now()
              };
              fs.writeFileSync(NODE_COLORS_PATH, JSON.stringify(data, null, 2));
              res.end(JSON.stringify({ success: true, timestamp: data.timestamp }));
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method not allowed' }));
            }
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });

        // API: GET /api/node-colors/check - Check for updates (lightweight)
        server.middlewares.use('/api/node-colors/check', (req: IncomingMessage, res: ServerResponse, next) => {
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            if (fs.existsSync(NODE_COLORS_PATH)) {
              const stat = fs.statSync(NODE_COLORS_PATH);
              res.end(JSON.stringify({ exists: true, mtime: stat.mtimeMs }));
            } else {
              res.end(JSON.stringify({ exists: false, mtime: 0 }));
            }
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });
      }
    },
    {
      name: 'neurips-api',
      configureServer(server) {
        // API: GET /api/neurips/papers - Load NeurIPS metadata
        server.middlewares.use('/api/neurips/papers', (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            if (!fs.existsSync(NEURIPS_METADATA_PATH)) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: 'metadata.csv not found' }));
              return;
            }

            const content = fs.readFileSync(NEURIPS_METADATA_PATH, 'utf-8');
            const papers = parseCSV(content);

            res.end(JSON.stringify({ papers, count: papers.length }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });

        // API: POST /api/chat - Forward chat to MCP server
        server.middlewares.use('/api/chat', async (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
            res.statusCode = 204;
            res.end();
            return;
          }

          if (req.method !== 'POST') {
            res.statusCode = 405;
            res.end(JSON.stringify({ error: 'Method not allowed' }));
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            const body = await readJsonBody(req);
            const message = body.message || '';

            // Forward to MCP server's chat endpoint
            const mcpUrl = process.env.MCP_SERVER_URL || 'http://mcp-server:8000';
            const chatRes = await fetch(`${mcpUrl}/chat`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message, history: body.history || [] })
            });

            if (chatRes.ok) {
              const data = await chatRes.json();
              res.end(JSON.stringify(data));
            } else {
              // Fallback response if MCP doesn't have chat endpoint
              res.end(JSON.stringify({
                response: `Chat endpoint not available. To use the LLM agent, run:\n\ndocker compose run --rm agent\n\nYour message was: "${message}"`
              }));
            }
          } catch (err) {
            res.end(JSON.stringify({
              response: `Chat service unavailable. Run the agent with:\n\ndocker compose run --rm agent`
            }));
          }
        });

        // API: GET /api/neurips/similarities - Get pre-computed similarities
        // This returns edges based on embedding similarities (computed from embeddings.npy)
        server.middlewares.use('/api/neurips/similarities', (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            // Check if pre-computed similarities file exists
            const simPath = '/app/data/embeddings_Neu/similarities.json';
            if (fs.existsSync(simPath)) {
              const data = fs.readFileSync(simPath, 'utf-8');
              res.end(data);
            } else {
              // Return empty - will be computed by Python script
              res.end(JSON.stringify({ edges: [], message: 'Run compute_similarities.py to generate' }));
            }
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });

        // API: GET /api/neurips/clusters?k=15 - Get KMeans clusters
        server.middlewares.use('/api/neurips/clusters', (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            // Parse k from query string (default: 15)
            const url = new URL(req.url || '', `http://${req.headers.host}`);
            const kParam = url.searchParams.get('k');
            const k = kParam ? parseInt(kParam, 10) : 15;
            const validK = Math.max(5, Math.min(30, k)); // Clamp to 5-30

            const clusterPath = `/app/data/embeddings_Neu/neurips_clusters_k${validK}.json`;
            if (fs.existsSync(clusterPath)) {
              const data = fs.readFileSync(clusterPath, 'utf-8');
              res.end(data);
            } else {
              res.end(JSON.stringify({ paper_id_to_cluster: {}, k: 0 }));
            }
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });
      }
    },
    {
      name: 'serve-output-files',
      configureServer(server) {
        // Middleware to serve files from /output directory
        server.middlewares.use('/output', (req, res, next) => {
          try {
            // Decode URL and construct file path
            const decodedUrl = decodeURIComponent(req.url || '');
            const filePath = path.join('/app/output', decodedUrl);

            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              const ext = path.extname(filePath).toLowerCase();

              // Set appropriate content type
              const contentTypes: Record<string, string> = {
                '.json': 'application/json',
                '.txt': 'text/plain',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
              };

              res.setHeader('Content-Type', contentTypes[ext] || 'application/octet-stream');
              res.setHeader('Access-Control-Allow-Origin', '*');
              fs.createReadStream(filePath).pipe(res);
            } else {
              // File not found
              res.statusCode = 404;
              res.end(`File not found: ${decodedUrl}`);
            }
          } catch (err) {
            next(err);
          }
        });
      }
    }
  ],
  root: '.',
  publicDir: 'public',
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.MCP_SERVER_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    },
    fs: {
      // Allow serving files from output directory
      allow: ['.', '/app/output', '../output']
    }
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.jsx', '.js']
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'd3']
  }
});
