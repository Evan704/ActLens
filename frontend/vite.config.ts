import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT ?? 5173),
    // ACTLENS_API lets several dev servers point at different backends (e.g. git worktrees)
    proxy: { "/api": process.env.ACTLENS_API ?? "http://localhost:8000" },
  },
  test: { environment: "node" },
});
