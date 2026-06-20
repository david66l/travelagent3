---
version: alpha
name: TravelAgent2-Wise-Inspired
description: >
  A light-green travel-assistant UI inspired by Wise's Scandinavian fintech language.
  The canvas is a pale sage (#e8ebe6), cards are pure white (#ffffff),
  and the single CTA accent is a vivid lime green (#9fe870).
  The system is friendly, generous, and information-dense enough for chat,
  itinerary dashboards, budget panels, and export workflows.

colors:
  primary: "#9fe870"
  on-primary: "#0e0f0c"
  primary-active: "#cdffad"
  primary-neutral: "#c5edab"
  primary-pale: "#e2f6d5"
  ink: "#0e0f0c"
  ink-deep: "#163300"
  body: "#454745"
  mute: "#868685"
  canvas: "#ffffff"
  canvas-soft: "#e8ebe6"
  hairline: "#e5e5e5"
  hairline-soft: "#ededed"
  positive: "#2ead4b"
  positive-deep: "#054d28"
  positive-pale: "#e2f6d5"
  warning: "#ffd11a"
  warning-deep: "#b86700"
  warning-content: "#4a3b1c"
  negative: "#d03238"
  negative-deep: "#a72027"
  negative-darkest: "#a7000d"
  negative-bg: "#320707"
  accent-orange: "#ffc091"
  accent-cyan: "#38c8ff"

# Semantic mapping for this app
semantic:
  page-bg: "{colors.canvas-soft}"
  panel-bg: "{colors.canvas}"
  panel-border: "{colors.hairline}"
  text-headline: "{colors.ink}"
  text-body: "{colors.body}"
  text-muted: "{colors.mute}"
  cta: "{colors.primary}"
  cta-text: "{colors.on-primary}"
  cta-hover: "{colors.primary-active}"

fonts:
  display: "Inter, system-ui, -apple-system, sans-serif"
  body: "Inter, system-ui, -apple-system, sans-serif"
  mono: "JetBrains Mono, ui-monospace, monospace"

radii:
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  full: 9999px

spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  2xl: 32px
  3xl: 48px

components:
  page:
    background: "{colors.canvas-soft}"
    text: "{colors.ink}"
    font: "{fonts.body}"

  nav-bar:
    background: "{colors.canvas}"
    text: "{colors.ink}"
    border: "1px solid {colors.hairline}"
    radius: "{radii.xl}"

  panel:
    background: "{colors.canvas}"
    border: "1px solid {colors.hairline}"
    radius: "{radii.xl}"
    shadow: "0 4px 24px rgba(0,0,0,0.04)"
    padding: "{spacing.lg}"

  panel-soft:
    background: "{colors.canvas-soft}"
    border: "1px solid {colors.hairline-soft}"
    radius: "{radii.lg}"
    padding: "{spacing.md}"

  button-primary:
    background: "{colors.primary}"
    text: "{colors.on-primary}"
    radius: "{radii.xl}"
    padding: "12px 24px"
    weight: 600
    hover: "{colors.primary-active}"

  button-secondary:
    background: "{colors.canvas-soft}"
    text: "{colors.ink}"
    radius: "{radii.xl}"
    padding: "12px 24px"
    weight: 600

  button-tertiary:
    background: "{colors.canvas}"
    text: "{colors.ink}"
    border: "1px solid {colors.ink}"
    radius: "{radii.xl}"
    padding: "12px 24px"
    weight: 600

  button-icon-circular:
    background: "{colors.canvas}"
    text: "{colors.ink}"
    border: "1px solid {colors.hairline}"
    size: 32px
    radius: "{radii.full}"

  text-input:
    background: "{colors.canvas}"
    text: "{colors.ink}"
    border: "1px solid {colors.hairline}"
    radius: "{radii.md}"
    padding: "12px 16px"
    focus: "2px solid {colors.primary}"

  pill-tab:
    background: "{colors.canvas-soft}"
    text: "{colors.body}"
    radius: "{radii.full}"
    padding: "8px 16px"
    active-bg: "{colors.ink}"
    active-text: "{colors.canvas}"

  badge:
    background: "{colors.primary-pale}"
    text: "{colors.positive-deep}"
    radius: "{radii.full}"
    padding: "4px 12px"
    size: 12px
    weight: 600

  progress-bar:
    track: "{colors.hairline-soft}"
    fill: "{colors.primary}"

  timeline-dot:
    attraction: "{colors.accent-orange}"
    restaurant: "{colors.negative}"
    hotel: "{colors.accent-cyan}"
    transport: "{colors.positive}"

  message-user:
    background: "{colors.ink}"
    text: "{colors.canvas}"
    radius: "{radii.lg}"

  message-ai:
    background: "{colors.canvas}"
    text: "{colors.ink}"
    border: "1px solid {colors.hairline}"
    radius: "{radii.lg}"

  map-route: "{colors.ink}"

---

## Overview

TravelAgent2 uses a **light-green, friendly, card-first** design language.
The page sits on a **pale sage canvas** (`{colors.canvas-soft}` — #e8ebe6),
content lives inside **rounded white cards** (`{colors.canvas}` — #ffffff),
and the single action color is a **vivid lime green** (`{colors.primary}` — #9fe870).
The result feels calm, trustworthy, and distinctly travel-oriented — like a
Scandinavian travel magazine rather than a dark dashboard.

**Key characteristics:**
- One brand accent: Wise lime (`{colors.primary}`) is reserved for primary CTAs,
  active states, progress bars, and the send button.
- White cards on sage canvas create natural elevation without heavy shadows.
- Generous 24 px (`{radii.xl}`) pill radius for buttons and cards.
- Inter for all UI prose; JetBrains Mono only for technical micro-copy.
- Near-black ink (`{colors.ink}`) for headlines, secondary body gray for descriptions,
  muted gray for captions.
- Semantic greens, yellows, and reds for status, budget, and errors — but never
  replace the lime CTA with a generic success green.

## Colors

### Brand & Accent
- **Wise Lime** (`{colors.primary}` — #9fe870): primary CTA, send button,
  active pill tabs, selected switches, progress-bar fills.
- **Lime Hover** (`{colors.primary-active}` — #cdffad): hover/pressed CTA.
- **Lime Neutral** (`{colors.primary-neutral}` — #c5edab): subtle active fills.
- **Lime Pale** (`{colors.primary-pale}` — #e2f6d5): badge backgrounds, soft highlights.

### Surfaces
- **Canvas** (`{colors.canvas}` — #ffffff): card backgrounds, inputs, top nav.
- **Canvas Soft** (`{colors.canvas-soft}` — #e8ebe6): page background.
- **Hairline** (`{colors.hairline}` — #e5e5e5): card and input borders.
- **Hairline Soft** (`{colors.hairline-soft}` — #ededed): dividers inside cards.

### Text
- **Ink** (`{colors.ink}` — #0e0f0c): headlines, primary text, user-message fill.
- **Body** (`{colors.body}` — #454745): secondary body copy.
- **Mute** (`{colors.mute}` — #868685): captions, placeholders, disabled text.

### Semantic
- **Positive** green family for success / transport / completion states.
- **Warning** yellow family for cautions / budget thresholds.
- **Negative** red family for errors / delete actions.

## Typography

| Role | Size | Weight | Line height | Use |
|------|------|--------|-------------|-----|
| Display | 40 px | 900 | 1.1 | Hero headline on landing/login |
| Heading | 24 px | 600 | 1.25 | Panel titles |
| Subheading | 18 px | 600 | 1.4 | Card section headers |
| Body | 16 px | 400 | 1.5 | Default body text |
| Body Strong | 16 px | 600 | 1.5 | Emphasised body |
| Caption | 14 px | 400 | 1.43 | Meta text, POI details |
| Caption Strong | 14 px | 600 | 1.43 | Labels, small buttons |
| Micro | 12 px | 500 | 1.33 | Timestamps, tags |
| Button | 16 px | 600 | 1.5 | Primary and secondary buttons |

## Layout

- Page padding: 20–24 px.
- Panel gap: 16 px.
- Panel internal padding: 16 px.
- Card radius: 24 px (`rounded-3xl`).
- Button radius: 24 px pill (`rounded-3xl`).
- Input radius: 12 px (`rounded-xl`).
- Sidebar width: 250 px.
- Preview / budget side panel: 350–360 px.

## Components

### Page
- Background `{colors.canvas-soft}`.
- All content floats inside white panels; no heavy global shadows.

### Nav Bar (`TopBar`)
- White rounded pill, 1 px hairline border, 64–72 px height.
- Logo: circular lime dot + bold ink wordmark.
- Avatar: circular white button with hairline border.

### Sidebar
- White rounded card, hairline border.
- Primary action "新建对话" uses `{button-primary}` (lime pill).
- Active item uses ink fill + white text.
- Inactive item uses canvas-soft hover.

### Chat Panel
- White panel with hairline border.
- Empty-state headline in Display weight.
- User messages: ink fill, white text, rounded-lg.
- AI messages: white card with hairline border, rounded-lg.
- Input area: white inner card, 1 px border, lime send button.
- Streaming/thinking bubble: white card with spinner.

### Itinerary Panel
- White panel.
- Day tabs: pill tabs; active tab is ink fill, inactive is canvas-soft.
- Export button: secondary sage or lime primary.
- Stage progress: hairline track, lime fill.
- Toolbar: white card with hairline border, small secondary buttons.

### Day Card & Activity Timeline
- Day badge: ink pill.
- Timeline dot color per category:
  - attraction → accent orange (#ffc091)
  - restaurant → negative red (#d03238)
  - hotel → accent cyan (#38c8ff)
  - transport → positive green (#2ead4b)
- Category labels: pale tint of the dot color.
- Tags: canvas-soft background, muted text.
- Action buttons: canvas-soft background, ink text.

### Panel Sidebar (Budget / Preferences)
- White panel.
- Budget cards: canvas-soft background, ink value, mute label.
- Progress bars: hairline-soft track, lime fill.
- Preference rows: label in mute, value in ink.

### Settings Panel
- White panel, preference cards in canvas-soft.
- Option chips: inactive canvas-soft, active ink fill.
- Save button: lime primary.

### Export Center
- White panel.
- Format list: inactive canvas-soft, active ink fill.
- Export button: lime primary when enabled, disabled hairline.

### Login Page
- Sage canvas background.
- Centered white card (`{panel}`) with 24 px padding.
- Title in Display weight, subtitle in Body Mute.
- Inputs: white with hairline border; focus ring in lime.
- Primary login button: lime pill.
- Secondary actions: ghost/text links in body color.

## Do's and Don'ts

### Do
- Reserve `{colors.primary}` lime for the single primary action per view.
- Put white cards on the sage canvas; let surface contrast create elevation.
- Use `{radii.xl}` (24 px) for all buttons and main panels.
- Use Inter for every UI label and body paragraph.
- Use the semantic palette for status; use lime only for CTAs and active states.

### Don't
- Don't add a second bright accent competing with lime.
- Don't use lime for body text or large backgrounds.
- Don't use sharp rectangles for primary buttons.
- Don't place lime CTAs on green-tinted backgrounds; lime must sit on white or ink.
