# Writing skills

Prompt-level guidance the writing agents load as system-prompt overlays. Each file is one concern,
so an agent gets only what its job needs: the Outliner and KeyPointer read `narrative.md` +
`sections.md`, the SectionDrafter `style.md` + `sections.md`, the Panelists and PaperReviewers
`review.md`; the one-shot Writer reads `narrative.md` + `style.md` and its Reviewer `review.md`.

Distilled from public advice by Neel Nanda ("How to write ML papers"), Sebastian Farquhar
(5-sentence abstract), Gopen & Swan ("The Science of Scientific Writing"), Zachary Lipton
("Heuristics for scientific writing"), Ethan Perez ("Easy paper writing tips"), Jacob Steinhardt,
Michael Black ("Writing a good scientific paper"), and the NeurIPS/ICML reviewer guidelines.

The guidance is advice to the model; the *enforced* rules live in code: number grounding
(`registry.py`), evidence grounding of key points, and the prose lint (`writing.py::lint_prose`).
