# Human Evaluation Template
## AI Movie Recommender — LLM Explanation Quality

---

## Instructions for Evaluators

1. Run the app (`streamlit run app.py`) and select a user from the sidebar.
2. Click **Get Recommendations** to generate recommendations and LLM explanations.
3. For each recommended movie, read the explanation the LLM produced and score it on the four criteria below.
4. Use the **1–5 scale** for every cell — do not leave cells blank.
5. Add brief notes in the final column for any score below 3 or above 4.
6. Fill in the **Summary** section after scoring all movies.

---

## Scoring Criteria

| Criterion | 1 — Poor | 3 — Acceptable | 5 — Excellent |
|---|---|---|---|
| **Relevance** | Explanation is unrelated to the movie or user's history | Mentions the movie but connection to user taste is vague | Clearly explains why *this user* would enjoy *this specific movie* |
| **Specificity** | Only generic statements ("great film", "you'll love it") | Some specific detail (genre, tone) but misses key elements | References concrete themes, directors, actors, or narrative elements |
| **Helpfulness** | Adds no information beyond what the title conveys | Gives the user something to act on, but partially | Gives the user a clear reason to watch (or skip) the movie |
| **Naturalness** | Reads like a template / robotic output | Mostly natural with occasional awkward phrasing | Reads like a knowledgeable friend's recommendation |

---

## Evaluation Table

**Evaluator name / ID:** ________________________________  
**User ID evaluated:** ________________________________  
**Date:** ________________________________

| # | Movie Title | LLM Explanation (paste here) | Relevance (1–5) | Specificity (1–5) | Helpfulness (1–5) | Naturalness (1–5) | Notes |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |

---

## Summary

| Criterion | Movie 1 | Movie 2 | Movie 3 | Movie 4 | Movie 5 | **Row Average** |
|---|---|---|---|---|---|---|
| Relevance | | | | | | |
| Specificity | | | | | | |
| Helpfulness | | | | | | |
| Naturalness | | | | | | |
| **Col Average** | | | | | | |

**Overall average score (all cells):** __________ / 5

---

## Evaluator Comments

**What worked well?**

> _(free text)_

**What needs improvement?**

> _(free text)_

**Suggested prompt or system changes?**

> _(free text)_

---

## Inter-Evaluator Agreement

Complete this section once all evaluators have submitted their forms.

| Evaluator | Overall Average | Std Dev across movies |
|---|---|---|
| Evaluator 1 | | |
| Evaluator 2 | | |
| Evaluator 3 | | |
| Evaluator 4 | | |
| Evaluator 5 | | |
| **Grand average** | | |

> **Target:** grand average ≥ 3.5 / 5 across all criteria before considering the system production-ready.  
> **Repeat** this form for **3–5 independent evaluators** to ensure scores are not idiosyncratic.

---

*Template version 1.0 — AI Movie Recommender project*
