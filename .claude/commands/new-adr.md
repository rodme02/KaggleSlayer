---
description: Scaffold the next-numbered ADR from the template and add it to the index
---

Create a new ADR for: $ARGUMENTS

1. Find the highest NNNN among `docs/adr/[0-9]*.md` and use NNNN+1 (zero-padded to 4 digits).
2. Copy `docs/adr/template.md` to `docs/adr/NNNN-<kebab-case-title>.md`; fill in Status (Accepted), today's date, and the Context / Decision / Consequences for the topic above. Cite the implementing code and any related spec section.
3. Add a row to the table in `docs/adr/README.md`.
4. Commit both files with message `docs: ADR NNNN — <title>`.
