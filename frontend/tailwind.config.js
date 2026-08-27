/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        // Verdict colours, defined once so the chat, the source pane and the
        // legend cannot drift apart.
        supported: "#10b981",
        partial: "#f59e0b",
        unsupported: "#ef4444",
        uncited: "#a855f7",
      },
      keyframes: {
        "fade-in": { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
      },
      animation: { "fade-in": "fade-in 150ms ease-out" },
    },
  },
  plugins: [],
};
