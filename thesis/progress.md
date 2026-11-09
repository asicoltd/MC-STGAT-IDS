# MC-STGAT-IDS Thesis — Progress Tracker

> Last updated: 2026-10-20

## File Status

### Front Matter
| # | File | Type | Status |
|---|------|------|--------|
| 1 | `preamble.tex` | LaTeX | ✅ Completed |
| 2 | `title_page.tex` | LaTeX | ✅ Completed |
| 3 | `declaration.tex` | LaTeX | ✅ Completed |
| 4 | `acknowledgment.tex` | LaTeX | ✅ Completed |
| 5 | `approval_page.tex` | LaTeX | ✅ Completed |
| 6 | `abstract.tex` | LaTeX | ✅ Completed |
| 7 | `keywords.tex` | LaTeX | ✅ Completed |
| 8 | `abbreviations.tex` | LaTeX | ✅ Completed |

### Main Chapters
| # | File | Type | Status |
|---|------|------|--------|
| 1 | `chapters/01_intro.tex` | LaTeX | ✅ Completed |
| 2 | `chapters/02_literature_review.tex` | LaTeX | ✅ Completed |
| 3 | `chapters/03_dataset_eda.tex` | LaTeX | ✅ Completed |
| 4 | `chapters/04_methodology.tex` | LaTeX | ✅ Completed |
| 5 | `chapters/05_experimental_setup.tex` | LaTeX | ✅ Completed |
| 6 | `chapters/06_results.tex` | LaTeX | ✅ Completed |
| 7 | `chapters/07_discussion.tex` | LaTeX | ✅ Completed |
| 8 | `chapters/08_conclusion.tex` | LaTeX | ✅ Completed |

### Back Matter
| # | File | Type | Status |
|---|------|------|--------|
| 1 | `references.bib` | BibTeX | ✅ Completed |

### Master & Output
| # | File | Type | Status |
|---|------|------|--------|
| 1 | `thesis.tex` | LaTeX | ✅ Completed |
| 2 | `thesis.pdf` | PDF | ✅ Compiled |

## Progress Count

| Metric | Count |
|--------|-------|
| Total files | 18 |
| Completed | 18 |
| In progress | 0 |
| Remaining | 0 |
| Completion | 100% |

## Compilation Status

| Attempt | Date | Status | Notes |
|---------|------|--------|-------|
| 1 | 2026-10-20 | ✅ Success | node-latex-compiler (Tectonic), 179 KB PDF |

## Tectonic Compatibility Fixes Applied

- `\frontmatter` → `\pagestyle{plain}\pagenumbering{roman}` (manual Roman numbering)
- `\mainmatter` → `\cleardoublepage\pagenumbering{arabic}` (manual Arabic numbering)
- `\backmatter` → `\clearpage` (removed unsupported command)
- `\usepackage{itemset}` → `\usepackage{enumitem}` (Tectonic can't resolve `itemset`)
- `C&C` → `C\&C` (escaped ampersand)
- `\_` in `\texttt{}` → plain `_` or `\verb|...|` (underscore escaping)
- `\includegraphics` → TikZ diagram (missing figure file)
- `\ReLU` → added `\newcommand{\ReLU}{\mathrm{ReLU}}` to preamble
- `\documentclass` duplicate → removed from `preamble.tex`

## Notes

- 
- 
