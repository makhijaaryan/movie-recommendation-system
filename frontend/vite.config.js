import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend (uvicorn src.api:app --port 8000) is proxied so the app can use relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/figures': 'http://127.0.0.1:8000',
    },
  },
})
