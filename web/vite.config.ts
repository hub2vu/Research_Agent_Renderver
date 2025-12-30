import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
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
