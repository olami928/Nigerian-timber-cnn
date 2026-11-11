# Timber ID project: instructions for coding agents

The full instructions are in @AGENT_BRIEF_timber_id.md. Read that file completely before doing anything.

Short rules (the brief has the detail):

- Work in phases P0 to P6 and STOP after each one. Wait for the owner to say "go".
- Never invent citations, species names, uses, or results. If something is missing, ask.
- Never modify or delete the image data. Keep a backup before changing anything.
- Models: ShuffleNetV2 (primary) and ResNet18 (comparison). MobileNetV2 must never appear.
- Splits are by specimen, not by image. Run the leak audit for every fold.
- Return and show only valid documented uses from `species_enduse.json`.
- Start with Phase 0 only: summarise the notebook, list conflicts and open questions, then stop.
