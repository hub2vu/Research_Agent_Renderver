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

// ICLR data paths
const ICLR_METADATA_PATH = '/app/data/embeddings_ICLR/ICLR2025_accepted_meta.csv';
const ICLR_EMBEDDINGS_PATH = '/app/data/embeddings_ICLR/ICLR2025_accepted_bge_large_en_v1_5.npy';

// Parse CSV helper - supports multiline fields in quotes
function parseCSV(content: string): Array<Record<string, string>> {
  const result: Array<Record<string, string>> = [];
  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;
  let headers: string[] = [];
  let headersDone = false;

  for (let i = 0; i < content.length; i++) {
    const char = content[i];

    if (char === '"') {
      if (inQuotes && content[i + 1] === '"') {
        // Escaped quote
        currentField += '"';
        i++;
      } else {
        // Toggle quote state
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      // Field separator
      currentRow.push(currentField);
      currentField = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      // Row separator (only when not in quotes)
      if (char === '\r' && content[i + 1] === '\n') {
        i++; // Skip \n in \r\n
      }
      if (currentField || currentRow.length > 0) {
        currentRow.push(currentField);
        currentField = '';

        if (!headersDone) {
          // First row is headers - trim whitespace from each header
          headers = currentRow.map(h => h.trim());
          headersDone = true;
        } else {
          // Data row
          const row: Record<string, string> = {};
          headers.forEach((h, idx) => {
            row[h] = (currentRow[idx] || '').trim();
          });
          result.push(row);
        }
        currentRow = [];
      }
    } else {
      currentField += char;
    }
  }

  // Handle last row if no trailing newline
  if (currentField || currentRow.length > 0) {
    currentRow.push(currentField);
    if (headersDone && currentRow.length > 0) {
      const row: Record<string, string> = {};
      headers.forEach((h, idx) => {
        row[h] = (currentRow[idx] || '').trim();
      });
      result.push(row);
    }
  }

  return result;
}

// Parse a single CSV line handling quoted fields (kept for backward compatibility)
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

        // API: POST /api/chat - Forward chat to Agent server
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

            // [수정됨] Agent 서버(8001)로 요청 전달
            const agentUrl = 'http://research-agent:8001';
            
            const chatRes = await fetch(`${agentUrl}/chat`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message, history: body.history || [] })
            });

            if (chatRes.ok) {
              const data = await chatRes.json();
              res.end(JSON.stringify(data));
            } else {
              res.end(JSON.stringify({
                response: `Agent server returned error: ${chatRes.statusText}`
              }));
            }
          } catch (err) {
            res.end(JSON.stringify({
              response: `Chat service unavailable. Error: ${String(err)}`
            }));
          }
        });

        // API: GET /api/neurips/similarities - Get pre-computed similarities
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
            const simPath = '/app/data/embeddings_Neu/similarities.json';
            if (fs.existsSync(simPath)) {
              const data = fs.readFileSync(simPath, 'utf-8');
              res.end(data);
            } else {
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
            const url = new URL(req.url || '', `http://${req.headers.host}`);
            const kParam = url.searchParams.get('k');
            const k = kParam ? parseInt(kParam, 10) : 15;
            const validK = Math.max(5, Math.min(30, k));

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
      name: 'iclr-api',
      configureServer(server) {
        // API: GET /api/iclr/papers - Load ICLR metadata
        server.middlewares.use('/api/iclr/papers', (req: IncomingMessage, res: ServerResponse, next) => {
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
            if (!fs.existsSync(ICLR_METADATA_PATH)) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: 'ICLR metadata.csv not found' }));
              return;
            }

            // Read with BOM handling
            let content = fs.readFileSync(ICLR_METADATA_PATH, 'utf-8');
            // Remove BOM if present
            if (content.charCodeAt(0) === 0xFEFF) {
              content = content.slice(1);
            }
            const papers = parseCSV(content);

            res.end(JSON.stringify({ papers, count: papers.length }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });

        // API: GET /api/iclr/similarities - Get pre-computed similarities
        server.middlewares.use('/api/iclr/similarities', (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            const simPath = '/app/data/embeddings_ICLR/similarities.json';
            if (fs.existsSync(simPath)) {
              const data = fs.readFileSync(simPath, 'utf-8');
              res.end(data);
            } else {
              res.end(JSON.stringify({ edges: [], message: 'Run compute_similarities.py to generate' }));
            }
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(err) }));
          }
        });

        // API: GET /api/iclr/clusters?k=15 - Get KMeans clusters
        server.middlewares.use('/api/iclr/clusters', (req: IncomingMessage, res: ServerResponse, next) => {
          if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.statusCode = 204;
            res.end();
            return;
          }

          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Content-Type', 'application/json');

          try {
            const url = new URL(req.url || '', `http://${req.headers.host}`);
            const kParam = url.searchParams.get('k');
            const k = kParam ? parseInt(kParam, 10) : 15;
            const validK = Math.max(5, Math.min(30, k));

            const clusterPath = `/app/data/embeddings_ICLR/iclr_clusters_k${validK}.json`;
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
        server.middlewares.use('/output', (req, res, next) => {
          try {
            const decodedUrl = decodeURIComponent(req.url || '');
            const safeUrl = decodedUrl.replace(/^\/+/, '');
            const filePath = path.join('/app/output', safeUrl);
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              const ext = path.extname(filePath).toLowerCase();
              const contentTypes: Record<string, string> = {
                '.json': 'application/json',
                '.txt': 'text/plain',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.pdf': 'application/pdf'
              };

              res.setHeader('Content-Type', contentTypes[ext] || 'application/octet-stream');
              res.setHeader('Access-Control-Allow-Origin', '*');
              fs.createReadStream(filePath).pipe(res);
            } else {
              res.statusCode = 404;
              res.end(`File not found: ${decodedUrl}`);
            }
          } catch (err) {
            next(err);
          }
        });
      }
        },
    {
      name: 'serve-pdf-files',
      configureServer(server) {
        server.middlewares.use('/pdf', (req, res, next) => {
          try {
            const decodedUrl = decodeURIComponent(req.url || '');
            const safeUrl = decodedUrl.replace(/^\/+/, '');
            const filePath = path.join('/app/pdf', safeUrl);

            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              res.setHeader('Content-Type', 'application/pdf');
              res.setHeader('Access-Control-Allow-Origin', '*');
              fs.createReadStream(filePath).pipe(res);
            } else {
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
      allow: ['.', '/app/output', '../output', '/app/pdf', '../pdf']
    }
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.jsx', '.js']
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'd3', 'react-markdown']
  }
});