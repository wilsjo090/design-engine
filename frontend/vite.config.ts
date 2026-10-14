import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `npm run dev` serves the UI on :5173 and forwards API calls to the Python server (python app.py).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
})
