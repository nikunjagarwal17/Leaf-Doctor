---
description: "Use when improving, refactoring, or adding features in this project. Enforce simple, safe changes: prefer built-in and library methods (NumPy, Pandas, TensorFlow, scikit-learn) over manual implementations, preserve file structure, avoid old-ui changes unless requested, and keep documentation updates in _features.md only."
name: "Leaf Doctor Simplicity and Safety Rules"
applyTo: "**"
---

# Leaf Doctor Simplicity and Safety Rules

- This instruction is global across the repository and applies to all work by default.
- The active code areas are Leaf-Doctor/ (main app) and Model/ (CNN model/training assets).
- Treat old-ui/ as legacy reference code; do not modify old-ui files unless explicitly requested.
- Prioritize safe improvements that do not break existing behavior.
- Before finalizing any change, run available checks or a minimal execution path relevant to modified files.
- Keep implementations simple and straightforward; avoid over-engineered patterns.
- Prefer built-in language features and standard library utilities first.
- When a well-supported library can solve the task clearly, use it instead of manual custom logic.
- For ML and data tasks, prefer ecosystem tools such as NumPy, Pandas, TensorFlow, and scikit-learn pipelines over hand-rolled implementations.
- Preserve the current project layout; do not reorganize folders or rename files unless explicitly requested.
- Avoid creating many markdown documents.
- Record completed improvements only in _features.md, using short plain-language explanations with no code snippets.
- If a new dependency is required, add it to Leaf-Doctor/requirements.txt and keep usage minimal.
