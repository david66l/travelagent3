import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#FF8400',
          foreground: '#111111',
        },
        background: '#F2F3F0',
        foreground: '#111111',
        card: {
          DEFAULT: '#FFFFFF',
          foreground: '#111111',
        },
        muted: {
          DEFAULT: '#F2F3F0',
          foreground: '#666666',
        },
        border: '#CBCCC9',
        accent: {
          DEFAULT: '#F2F3F0',
          foreground: '#111111',
        },
        destructive: {
          DEFAULT: '#D93C15',
          foreground: '#FFFFFF',
        },
        success: {
          DEFAULT: '#DFE6E1',
          foreground: '#004D1A',
        },
        warning: {
          DEFAULT: '#E9E3D8',
          foreground: '#804200',
        },
        info: {
          DEFAULT: '#DFDFE6',
          foreground: '#000066',
        },
        surface: {
          DEFAULT: '#ffffff',
          dark: '#0f172a',
        },
      },
      fontFamily: {
        mono: ['var(--font-jetbrains)', 'monospace'],
        sans: ['var(--font-geist)', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '16px',
        '3xl': '24px',
        '4xl': '26px',
      },
      backdropBlur: {
        'panel': '24px',
        'card': '30px',
      },
    },
  },
  plugins: [typography],
};

export default config;
