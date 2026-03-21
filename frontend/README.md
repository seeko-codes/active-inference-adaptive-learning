# Frontend — Adaptive Learning UI

React + TypeScript interface for the active inference tutoring system.

## Session Flow

```
login → problem → confidence → feedback → (next problem)
```

1. **Login** — enter a student ID (or leave blank for a default)
2. **Problem** — answer an algebra question with an explanation
3. **Confidence** — self-report how confident you are (1–5)
4. **Feedback** — see if you were correct and what the right answer is
5. Repeat — the backend's active inference agent picks the next problem

## Components

| File | Purpose |
|---|---|
| `App.tsx` | Top-level session state machine and API orchestration |
| `ProblemCard.tsx` | Renders the problem, answer input, and explanation box |
| `ConfidencePrompt.tsx` | 1–5 confidence self-report after answering |
| `FeedbackView.tsx` | Correct/incorrect display with expected answer |
| `SessionDashboard.tsx` | Sidebar: live belief state, accuracy, action reasoning |
| `MathDisplay.tsx` | LaTeX math rendering |
| `api.ts` | Typed API client for backend endpoints |

## Running

```bash
npm install
npm run dev       # dev server at http://localhost:5173
npm run build     # production build to dist/
```

Expects the FastAPI backend running at `http://localhost:8000`.
See the root [README](../README.md) for backend setup.
