import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "./", // relative paths so the build works on any static host / sub-path
  plugins: [react(), tailwindcss()],
  optimizeDeps: { exclude: ["onnxruntime-web"] }, // ort loads its own .wasm files
});
