# Catalog numbering rules

Enforced by `scripts/catalog.py`.

- Album = one per major version (`line@1.x`; a whole `0.x` series is one album). Exception:
  lines with `album_scheme: minor` (the 3.x-lockstep pipelines, senter, vectors, NER corpus) get
  one album per minor (`3.9`, `3.10`). Everything else is a pressing.
- New number only for a new album, assigned at release (= max existing + 1), chronological.
- A prerelease (`rc`, `a`, `b`, `dev`) on an existing album is a flagged pressing. A prerelease
  that would open a new album gets no number; it sits in `data/unnumbered.yaml` and moves onto
  the album when its first public pressing ships.
- Numbers in `data/catalog.lock.json` are permanent.

