---
description: "Use when implementing or modifying Leaf Doctor features. Enforce simple readable code, library-first choices, non-deprecated APIs, and minimal high-value edits."
name: "Leaf Doctor Coding Guidelines"
---
# Leaf Doctor - Coding Guidelines

## Code Quality Principles

### Simplicity And Readability First
- Keep code simple, readable, and logically ordered.
- Avoid premature abstraction or over-engineering.
- Do not create unused functions or unnecessary intermediate layers.
- Ensure each line has a clear, immediate purpose.

### Library-First Approach
- Prefer established libraries and framework features over custom implementations.
- Check official library or framework documentation before implementing a feature.
- Use built-in utilities, helpers, and standard patterns whenever available.
- Add custom logic only when a library cannot solve the need.

### Deprecation And Best Practices
- Do not use deprecated functions or APIs.
- Verify current best practices before implementation.
- Follow current dependency conventions and changelogs.
- When uncertain, consult official docs before coding.

### Files And Changes
- Make focused, intentional changes only.
- Do not modify files unnecessarily.
- Keep changesets clean and purpose-driven.
- If a change does not add value or solve a real problem, do not include it.

## Before Implementing

1. Search official docs for the library or framework first.
2. Confirm the API or function is current and not deprecated.
3. Check for existing patterns in the codebase and follow them.
4. Prefer built-in or prebuilt library features over custom functions.

## Additional Constraints
- Do not create unnecessary markdown files.
- Keep implementations minimal and avoid overcomplicating solutions.
- Avoid large, unnecessary patch blocks; include only essential edits.
