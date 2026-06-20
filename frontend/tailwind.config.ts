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
        // Wise-inspired token palette
        primary: {
          DEFAULT: '#9fe870',
          foreground: '#0e0f0c',
          active: '#cdffad',
          neutral: '#c5edab',
          pale: '#e2f6d5',
        },
        ink: '#0e0f0c',
        'ink-deep': '#163300',
        body: '#454745',
        mute: '#868685',
        canvas: {
          DEFAULT: '#ffffff',
          soft: '#e8ebe6',
        },
        hairline: {
          DEFAULT: '#e5e5e5',
          soft: '#ededed',
        },
        positive: {
          DEFAULT: '#2ead4b',
          deep: '#054d28',
          pale: '#e2f6d5',
        },
        negative: {
          DEFAULT: '#d03238',
          deep: '#a72027',
          darkest: '#a7000d',
          bg: '#320707',
        },
        warning: {
          DEFAULT: '#ffd11a',
          deep: '#b86700',
          content: '#4a3b1c',
        },
        'accent-orange': '#ffc091',
        'accent-cyan': '#38c8ff',

        // Semantic aliases for Tailwind defaults
        background: '#e8ebe6',
        foreground: '#0e0f0c',
        card: {
          DEFAULT: '#ffffff',
          foreground: '#0e0f0c',
        },
        muted: {
          DEFAULT: '#e8ebe6',
          foreground: '#454745',
        },
        border: '#e5e5e5',
        accent: {
          DEFAULT: '#e2f6d5',
          foreground: '#0e0f0c',
        },
        destructive: {
          DEFAULT: '#d03238',
          foreground: '#ffffff',
        },
        success: {
          DEFAULT: '#e2f6d5',
          foreground: '#054d28',
        },
        info: {
          DEFAULT: '#e8ebe6',
          foreground: '#0e0f0c',
        },
        surface: {
          DEFAULT: '#ffffff',
          dark: '#0e0f0c',
        },
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains)', 'monospace'],
      },
      borderRadius: {
        '2xl': '16px',
        '3xl': '24px',
        '4xl': '32px',
      },
      boxShadow: {
        panel: '0 4px 24px rgba(0, 0, 0, 0.04)',
        card: '0 4px 24px rgba(0, 0, 0, 0.06)',
        float: '0 12px 30px rgba(0, 0, 0, 0.10)',
      },
      backdropBlur: {
        panel: '24px',
        card: '30px',
      },
    },
  },
  plugins: [typography],
};

export default config;
