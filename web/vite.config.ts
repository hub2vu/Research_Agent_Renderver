import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import type { IncomingMessage, ServerResponse } from 'http';

// Node colors file path
const NODE_COLORS_PATH = '/app/output/graph/node_colors.json';

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
