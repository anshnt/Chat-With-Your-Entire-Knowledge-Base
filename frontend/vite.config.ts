import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API is proxied rather than hard-coded, so the app is same-origin in
    // development and needs no CORS handling or environment variable to run.
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/files": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
