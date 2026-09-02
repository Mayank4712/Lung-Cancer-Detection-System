import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base deep-space palette
        base: {
          DEFAULT: "#0B0F17",
          surface: "#131927",
          elevated: "#1A2233",
        },
        accent: {
          300: "#67e8f9",
          400: "#22d3ee",
          500: "#06b6d4",
          600: "#0891b2",
        },
        medical: {
          cyan: "#22d3ee",
          emerald: "#10b981",
          amber: "#f59e0b",
          crimson: "#e11d48",
        },
      },
      boxShadow: {
        glow: "0 0 24px -6px rgba(34, 211, 238, 0.35)",
        "glow-emerald": "0 0 24px -6px rgba(16, 185, 129, 0.35)",
        "panel": "0 8px 40px -12px rgba(0, 0, 0, 0.6)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(34, 211, 238, 0.0)" },
          "50%": { boxShadow: "0 0 0 6px rgba(34, 211, 238, 0.25)" },
        },
      },
      animation: {
        shimmer: "shimmer 2s linear infinite",
        "fade-in-up": "fade-in-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both",
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
