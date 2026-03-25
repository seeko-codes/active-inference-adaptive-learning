# Prompt for Claude in PowerPoint

Copy everything below this line and paste it to Claude when working in PowerPoint.

---

Create a research poster (36" wide × 48" tall, portrait orientation) for UCF's Student Research Week symposium. Use PowerPoint custom slide size: 36 × 48 inches.

## Layout

Use this 3-column layout with the title banner spanning the full width:

```
┌─────────────────────────────────────────────────────────┐
│              TITLE (centered, 64pt+ bold)               │
│         Author · Department · UCF · Mentor              │
│      [UCF SRW Logo left]    [OUR Logo right]            │
├──────────────┬──────────────────────┬───────────────────┤
│              │                      │                   │
│  1. INTRO    │  3. RESULTS          │  5. FUTURE WORK   │
│              │                      │                   │
│              │  [Bar Chart]         │                   │
│              │                      ├───────────────────┤
├──────────────┤  [Line Chart]        │                   │
│              │                      │  6. REFERENCES    │
│  2. METHODS  │  [Heatmap]           │                   │
│              │                      │                   │
│  [System     │  Key findings        ├───────────────────┤
│   Diagram]   │  as callout boxes    │  7. ACKNOWLEDGE-  │
│              │                      │     MENTS         │
└──────────────┴──────────────────────┴───────────────────┘
```

Column widths: Left ~30%, Center ~40%, Right ~30%.
Flow reads top-left → down → center → right (newspaper style).

## Design Rules

**Fonts:**
- Title: 64pt bold, dark color
- Section headings: 36pt bold
- Body text: 24pt minimum
- Pick one serif font for headings (e.g., Georgia) and one sans-serif for body (e.g., Calibri)
- Left-align body text. Center only headings.

**Colors (UCF-inspired palette):**
- Background: White or very light warm gray (#F8F7F5)
- Title banner background: UCF Black (#000000) with UCF Gold (#FFC904) accent line
- Section heading color: UCF Black
- Accent/highlight color: UCF Gold (#FFC904)
- Body text: Dark gray (#333333)
- Chart colors: Use a 5-color sequential palette distinguishable in grayscale

**Visual style:**
- Clean, minimal, lots of white space
- Section boxes with subtle borders or light background tints — not heavy boxes
- One dominant visual element: the Results center column with 3 charts
- System diagram in Methods should be ~1/3 the width of the left column
- No random emoji or decorative elements

**Charts (placeholders for now — I'll insert the real ones later):**
- Create placeholder rectangles where charts will go, labeled:
  - "CHART 1: Policy Comparison (bar chart)" — center column top
  - "CHART 2: Learning Trajectories (line chart)" — center column middle
  - "CHART 3: Learning Landscape (heatmap)" — center column bottom
- Style the placeholders with a light gray fill (#EEEEEE) and dashed border
- Size each chart placeholder at roughly 12" wide × 6" tall

**Logos:**
- UCF Student Research Week logo: bottom-left corner
- If available, UCF Office of Undergraduate Research logo: bottom-right corner

## Section Content

Paste the text from POSTER_TEXT.md into each section. Key formatting:

- **Introduction:** 3 short paragraphs. Bold the key phrases: "normative method" and "deductively determine the most effective learning path"
- **Methods:** Bullet list with bold dimension names. The 5-policy comparison table. Keep the system diagram.
- **Results:** Chart placeholders with 2-3 bullet callouts beneath each chart. Use bold for key numbers.
- **Conclusion:** Single paragraph, ~60 words
- **Future Work:** 3 bullets only
- **References:** Numbered list, smaller font (20pt) is OK here
- **Acknowledgements:** Single sentence, smallest text (20pt)

## Important

- Do NOT shrink text below 20pt for any reason. If it doesn't fit, cut words.
- The center Results column should feel like the visual anchor — biggest area, most color, most charts.
- Keep the left and right columns text-forward but concise.
- Total text on the poster should be readable in under 5 minutes at arm's length.
