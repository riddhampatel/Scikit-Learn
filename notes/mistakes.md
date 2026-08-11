# ⚠️ Scikit-Learn Mistakes & Lessons Learned Log

This document records real mistakes made during daily coding, why they occurred, and how to fix them.

---

### Day 01 — Python for ML
- *Mistake*: Forgetting that `dict.get(key, default)` avoids `KeyError` when accessing missing keys.
- *Correction*: Use `person.get('age', 0)` instead of `person['age']` when key existence isn't guaranteed.
