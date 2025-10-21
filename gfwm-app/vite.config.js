import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  root: '.', 
  base: './', // This is the directory where Vite will serve the files from
  server: {
    port: 3000, // Vite will serve on this port
  },
  build: {
    outDir: 'build', // Where the build files will go
  }
})
