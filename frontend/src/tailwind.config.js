/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
      },
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#e0eaff',
          500: '#4f6ef7',
          600: '#3b5be8',
          700: '#2d48d0',
        },
        danger:  { 400: '#f87171', 500: '#ef4444', 100: '#fee2e2' },
        warning: { 400: '#fbbf24', 500: '#f59e0b', 100: '#fef3c7' },
        safe:    { 400: '#34d399', 500: '#10b981', 100: '#d1fae5' },
      },
    },
  },
  plugins: [],
}
