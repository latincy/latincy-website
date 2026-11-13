# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pyyaml"]
# ///
"""LatinCy label catalog tooling.

Usage:
    uv run scripts/catalog.py harvest            # refresh data/harvest/*.json from HF, PyPI, GitHub
    uv run scripts/catalog.py propose            # group evidence into albums -> scratch/numbering-proposal.md
    uv run scripts/catalog.py check              # validate entries, lockfile, schedule (CI preflight)
    uv run scripts/catalog.py release LINE VER [--date D] [--formats ..] [--source URL] [--write]
    uv run scripts/catalog.py refresh [--write]  # re-derive scores + install hints from the harvest
    uv run scripts/catalog.py audit [--cached] [--links]   # public pressings missing from the catalog
    uv run scripts/catalog.py schedule next|roll [--write]
    uv run scripts/catalog.py lock               # freeze numbers (first deploy)

Write commands are dry runs unless --write is given.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HARVEST = DATA / "harvest"
LINES_FILE = DATA / "lines.yaml"

HF_API = "https://huggingface.co/api"
PYPI_API = "https://pypi.org/pypi"


def load_lines() -> dict:
    return yaml.safe_load(LINES_FILE.read_text())["lines"]


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dump_yaml(obj) -> str:
    return yaml.dump(obj, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True, width=1000)


def write_json(path: Path, obj) -> None:
    """Atomic write (tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --- harvest -----------------------------------------------------------------


def hf_repo_ids(lines: dict) -> list[str]:
    ids = []
    for line in lines.values():
        ids += list((line.get("formats") or {}).values())
        if line.get("hf"):
            ids.append(line["hf"])
        ids += line.get("legacy_hf") or []
    return sorted(set(ids))


def harvest_hf(client: httpx.Client, lines: dict) -> dict:
    out = {}
    for repo in hf_repo_ids(lines):
        info = client.get(f"{HF_API}/models/{repo}")
        if info.status_code != 200:
            out[repo] = {"status": info.status_code}
            print(f"  hf {repo}: HTTP {info.status_code}", file=sys.stderr)
            continue
        meta = info.json()
        refs = client.get(f"{HF_API}/models/{repo}/refs").json()
        commits, page = [], 0
        while True:
            r = client.get(f"{HF_API}/models/{repo}/commits/main", params={"p": page})
            batch = r.json() if r.status_code == 200 else []
            if not batch:
                break
            commits += [
                {"id": c["id"][:12], "date": c["date"], "title": c["title"]} for c in batch
            ]
            if len(batch) < 50:
                break
            page += 1
        files = [s["rfilename"] for s in meta.get("siblings", [])]
        evaluation = None
        for name in ("eval.json", "meta.json"):
            if name in files and evaluation is None:
                e = client.get(f"https://huggingface.co/{repo}/resolve/main/{name}")
                if e.status_code == 200:
                    j = e.json()
                    evaluation = j.get("performance") if name == "meta.json" else j
                    if evaluation is not None:
                        evaluation = {"_file": name, **evaluation}
        out[repo] = {
            "status": 200,
            "eval": evaluation,
            "createdAt": meta.get("createdAt"),
            "lastModified": meta.get("lastModified"),
            "private": meta.get("private"),
            "files": files,
            "tags": [t["name"] for t in refs.get("tags", [])],
            "branches": [b["name"] for b in refs.get("branches", [])],
            "commits": commits,
        }
        print(f"  hf {repo}: {len(commits)} commits", file=sys.stderr)
    return out


def harvest_pypi(client: httpx.Client, lines: dict) -> dict:
    out = {}
    for line in lines.values():
        name = line.get("pypi")
        if not name:
            continue
        r = client.get(f"{PYPI_API}/{name}/json")
        if r.status_code != 200:
            out[name] = {"status": r.status_code}
            continue
        releases = {}
        for ver, files in r.json()["releases"].items():
            if not files:
                continue
            releases[ver] = {
                "date": min(f["upload_time_iso_8601"] for f in files),
                "yanked": any(f.get("yanked") for f in files),
            }
        out[name] = {"status": 200, "releases": releases}
        print(f"  pypi {name}: {len(releases)} versions", file=sys.stderr)
    return out


def gh_api(path: str):
    r = subprocess.run(
        ["gh", "api", "--paginate", path], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        return None
    # --paginate concatenates JSON arrays as "][": normalise
    return json.loads(r.stdout.replace("]\n[", ",").replace("][", ","))


def harvest_github(lines: dict) -> dict:
    out = {}
    for line in lines.values():
        repo = line.get("github")
        if not repo or repo in out:
            continue
        meta = gh_api(f"repos/{repo}")
        if meta is None:
            out[repo] = {"status": "unavailable"}
            print(f"  gh {repo}: unavailable", file=sys.stderr)
            continue
        tags = []
        for t in gh_api(f"repos/{repo}/tags?per_page=100") or []:
            c = gh_api(f"repos/{repo}/commits/{t['commit']['sha']}") or {}
            date = (c.get("commit") or {}).get("committer", {}).get("date")
            tags.append({"name": t["name"], "date": date})
        releases = [
            {
                "tag": r["tag_name"],
                "published_at": r["published_at"],
                "prerelease": r["prerelease"],
                "draft": r["draft"],
            }
            for r in gh_api(f"repos/{repo}/releases?per_page=100") or []
        ]
        out[repo] = {
            "status": 200,
            "private": meta.get("private"),
            "created_at": meta.get("created_at"),
            "pushed_at": meta.get("pushed_at"),
            "tags": tags,
            "releases": releases,
        }
        print(f"  gh {repo}: {len(tags)} tags, {len(releases)} releases", file=sys.stderr)
    return out


def cmd_harvest(args) -> None:
    lines = load_lines()
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if args.source in ("all", "hf"):
            write_json(HARVEST / "hf.json", {"harvested": stamp, "repos": harvest_hf(client, lines)})
        if args.source in ("all", "pypi"):
            write_json(HARVEST / "pypi.json", {"harvested": stamp, "packages": harvest_pypi(client, lines)})
    if args.source in ("all", "github"):
        write_json(HARVEST / "github.json", {"harvested": stamp, "repos": harvest_github(lines)})


# --- propose -----------------------------------------------------------------

import re

VER = r"\d+\.\d+(?:\.\d+)?(?:-?rc\d+)?"
HF_VERSION_PATTERNS = [
    re.compile(rf"-({VER})-py3"),  # wheel filename
    re.compile(rf"^(?:release:?\s*|hot patch\s+|fix\()?v?({VER})\b(?!\.)", re.I),
    re.compile(rf"\b(?:Release|Add|Initial release:|Initial upload:)\s+(?:\S+\s+)?v?({VER})\b(?!\.)"),
    re.compile(rf"\(v({VER})[):]"),
]
NON_MODEL_FILES = ("README", "LICENSE", "CHANGELOG", "meta.json", "eval.json", ".gitattributes")
PRERELEASE = re.compile(r"(rc|a|b|dev)\d*$")
SIZE_ORDER = {"sm": 0, "md": 1, "lg": 2, "trf": 3}


def hf_title_version(title: str) -> str | None:
    if title.lower().startswith("fix(deps)"):
        return None
    for pat in HF_VERSION_PATTERNS:
        m = pat.search(title)
        if m:
            return m.group(1)
    return None


def is_model_upload(title: str) -> bool:
    t = title.lower()
    return (t.startswith("upload") or t.startswith("add installable")) and not any(
        f.lower() in t for f in NON_MODEL_FILES
    )


def album_of(line: dict, version: str | None, override: str | None = None) -> str:
    if override is not None:
        return override
    if "album" in line:
        return line["album"]
    if version is None:
        return "0.x"
    major, minor = version.split(".")[:2]
    minor = re.sub(r"\D.*", "", minor)
    if major == "0":
        return "0.x"
    # default: one album per major; 3.x-lockstep lines (album_scheme: minor) get one per minor
    return f"{major}.{minor}" if line.get("album_scheme") == "minor" else f"{major}.x"


def collect_pressings(name: str, line: dict, hf: dict, pypi: dict, gh: dict) -> tuple[list, list]:
    """Return (pressings, notes) for one line. A pressing is one public version (all formats)."""
    found: dict[tuple, dict] = {}
    notes: list[str] = []

    def add(version, date, fmt, source, flags=(), album=None, review=None, repo=None, evidence=None):
        key = (album_of(line, version, album), version, date if version is None else None)
        p = found.setdefault(
            key,
            {"album": key[0], "version": version, "date": date, "formats": set(),
             "source": source, "flags": set(), "review": [], "repos": {}, "evidence": evidence},
        )
        if date < p["date"]:
            p["date"], p["source"] = date, source
        if fmt:
            p["formats"].add(fmt)
        if repo:
            p["repos"].setdefault(fmt or "", repo)
        p["flags"].update(flags)
        if review:
            p["review"].append(review)

    repos = dict((line.get("formats") or {}).items())
    if line.get("hf"):
        repos[""] = line["hf"]
    for fmt, repo in repos.items():
        data = hf.get(repo, {})
        if data.get("status") != 200:
            notes.append(f"HF {repo}: HTTP {data.get('status')} (private or missing)")
            continue
        seen_versions = set()
        for c in sorted(data["commits"], key=lambda c: c["date"]):
            date = c["date"][:10]
            src = f"https://huggingface.co/{repo}/commit/{c['id']}"
            v = hf_title_version(c["title"])
            if v:
                flags = {"hotfix"} if re.search(r"hot ?(patch|fix)", c["title"], re.I) else set()
                if PRERELEASE.search(v):
                    flags.add("prerelease")
                add(v, date, fmt, src, flags, repo=repo)
                seen_versions.add(v)
            elif line.get("upload_pressings") and is_model_upload(c["title"]):
                add(None, date, fmt, src, repo=repo)
        for tag in data.get("tags", []):
            v = tag.lstrip("v")
            if v not in seen_versions and not any(m.get("version") == v for m in line.get("manual") or []):
                notes.append(f"HF {repo}: tag {tag} has no dated upload commit")
        if data.get("branches") and set(data["branches"]) - {"main"}:
            notes.append(f"HF {repo}: extra branches {sorted(set(data['branches']) - {'main'})}")

    if line.get("pypi"):
        pkg = pypi.get(line["pypi"], {})
        for v, r in (pkg.get("releases") or {}).items():
            flags = {"yanked"} if r["yanked"] else set()
            if PRERELEASE.search(v):
                flags.add("prerelease")
            add(v, r["date"][:10], None, f"https://pypi.org/project/{line['pypi']}/{v}/", flags)

    if line.get("github"):
        g = gh.get(line["github"], {})
        if g.get("private"):
            notes.append(f"GitHub {line['github']}: private repo")
        tags = {t["name"].lstrip("v"): t for t in g.get("tags", [])}
        if line.get("pypi"):
            on_pypi = set((pypi.get(line["pypi"], {}).get("releases") or {}).keys())
            extra = sorted(set(tags) - on_pypi, key=lambda s: [int(x) if x.isdigit() else 0 for x in re.split(r"[.-]", s)])
            if extra:
                notes.append(f"GitHub tags not on PyPI (not counted): {', '.join(extra)}")
        elif not line.get("hf") and not line.get("formats") and not g.get("private"):
            for v, t in tags.items():
                if t.get("date"):
                    add(v, t["date"][:10], None, f"https://github.com/{line['github']}/releases/tag/{t['name']}")
            if not tags:
                notes.append(f"GitHub {line['github']}: no tags")

    for m in line.get("manual") or []:
        date = m["date"].isoformat() if hasattr(m["date"], "isoformat") else str(m["date"])
        v = m.get("version")
        flags = {"prerelease"} if v and PRERELEASE.search(v) else set()
        fmts = m.get("formats") or [None]
        for f in fmts:
            hf_repo = None
            if m["source"].startswith("https://huggingface.co/"):
                hf_repo = "/".join(m["source"].split("/")[3:5])
            add(v, date, f, m["source"], flags, m.get("album"), m.get("review"),
                repo=hf_repo if f is None or hf_repo in (line.get("formats") or {}).values()
                or hf_repo in (line.get("legacy_hf") or []) else None,
                evidence=m.get("evidence"))

    pressings = sorted(found.values(), key=lambda p: (p["date"], p["version"] or ""))
    # unversioned uploads inside an album that has a same-day versioned pressing are redundant
    versioned_days = {(p["album"], p["date"]) for p in pressings if p["version"]}
    pressings = [p for p in pressings if p["version"] or (p["album"], p["date"]) not in versioned_days]
    return pressings, notes


def build_albums() -> tuple[list[dict], dict]:
    lines = load_lines()
    hf = json.loads((HARVEST / "hf.json").read_text())["repos"]
    pypi = json.loads((HARVEST / "pypi.json").read_text())["packages"]
    gh = json.loads((HARVEST / "github.json").read_text())["repos"]
    albums, line_notes = [], {}
    for name, line in lines.items():
        pressings, notes = collect_pressings(name, line, hf, pypi, gh)
        line_notes[name] = notes
        by_album: dict[str, list] = {}
        for p in pressings:
            by_album.setdefault(p["album"], []).append(p)
        for album, ps in by_album.items():
            public = [p for p in ps if not ({"yanked", "prerelease"} & p["flags"])]
            status = "released" if public else (
                "withdrawn" if all("yanked" in p["flags"] for p in ps) else "advance")
            albums.append({
                "line": name,
                "album": album,
                "title": f"{name} {album}".strip(),
                "kind": line["kind"],
                "language": line["language"],
                "include": line.get("include", True),
                "status": status,
                "date": min(p["date"] for p in (public or ps)),
                "pressings": ps,
                "review": sorted({r for p in ps for r in p["review"]}),
            })
    return albums, line_notes


def number_albums(albums: list[dict]) -> None:
    numbered = [a for a in albums if a["include"] is True and a["status"] == "released"]
    numbered.sort(key=lambda a: (a["date"], a["line"], a["album"]))
    for i, a in enumerate(numbered, 1):
        a["cat"] = i


def fmt_pressing(p: dict) -> str:
    v = p["version"] or "(unversioned upload)"
    bits = [f"[{v}]({p['source']}) {p['date']}"]
    if p["formats"] - {None}:
        bits.append("/".join(sorted(p["formats"] - {None}, key=lambda f: (SIZE_ORDER.get(f, 9), f))))
    if p["flags"]:
        bits.append("`" + ",".join(sorted(p["flags"])) + "`")
    return " ".join(bits)


def cmd_propose(args) -> None:
    albums, line_notes = build_albums()
    number_albums(albums)
    out = ROOT / "scratch" / "numbering-proposal.md"
    out.parent.mkdir(exist_ok=True)
    harvested = json.loads((HARVEST / "hf.json").read_text())["harvested"]
    L = [
        "# Catalog numbering proposal",
        "",
        f"Generated by `scripts/catalog.py propose` from evidence harvested {harvested}.",
        "Apparatus only. ⚠ = needs Patrick's decision. Numbers are provisional until the lockfile is committed.",
        "",
        "## Numbered albums (chronological)",
        "",
        "| Cat. | Album | Lang | Kind | First public | Pressings | ⚠ |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in sorted((a for a in albums if "cat" in a), key=lambda a: a["cat"]):
        L.append(
            f"| LAT-{a['cat']:03d} | `{a['title']}` | {a['language']} | {a['kind']} | {a['date']} | "
            + "<br>".join(fmt_pressing(p) for p in a["pressings"])
            + f" | {'; '.join(a['review'])} |"
        )
    pending = [a for a in albums if "cat" not in a]
    L += ["", "## Not numbered (decision needed or not yet released)", "",
          "| Album | Lang | Kind | Reason | First seen | Pressings |", "|---|---|---|---|---|---|"]
    for a in sorted(pending, key=lambda a: a["date"]):
        reason = a["status"] if a["status"] != "released" else f"include: {a['include']}"
        L.append(f"| `{a['title']}` | {a['language']} | {a['kind']} | {reason} | {a['date']} | "
                 + "<br>".join(fmt_pressing(p) for p in a["pressings"]) + " |")
    crosswalk = yaml.safe_load((DATA / "legacy_crosswalk.yaml").read_text())
    by_key = {f"{a['line']}@{a['album']}": a for a in albums}
    L += ["", "## Crosswalk: old site entries -> proposed numbers", "",
          "Old body text is carried verbatim into `legacy_notes` (not rendered) pending Patrick's review.", "",
          "| Old | Old title | -> | Album | Format | Cover |", "|---|---|---|---|---|---|"]
    for old, x in crosswalk.items():
        f = ROOT / "content" / "catalog" / f"{old.lower()}.md"
        fm = yaml.safe_load(f.read_text().split("---")[1]) if f.exists() else {}
        a = by_key.get(x["album"])
        new = f"LAT-{a['cat']:03d}" if a and "cat" in a else "unnumbered"
        L.append(f"| {old} | {fm.get('title', '?')} | {new} | `{x['album']}` | {x.get('format', '')} | {fm.get('cover', '')} |")
    L += ["", "## Evidence notes", ""]
    for name, notes in line_notes.items():
        for n in notes:
            L.append(f"- `{name}`: {n}")
    out.write_text("\n".join(L) + "\n")
    print(f"{sum('cat' in a for a in albums)} numbered, {len(pending)} pending -> {out.relative_to(ROOT)}")


# --- migrate -----------------------------------------------------------------

CATALOG = ROOT / "content" / "catalog"
UNNUMBERED_FILE = DATA / "unnumbered.yaml"
SCORE_KEYS = ["pos_acc", "tag_acc", "morph_acc", "lemma_acc", "dep_uas", "dep_las", "ents_f", "sents_f"]


def version_key(v: str | None) -> list:
    return [int(x) if x.isdigit() else -1 for x in re.split(r"[.-]", v or "")]


def pressing_out(p: dict) -> dict:
    out = {"version": p["version"], "date": dt.date.fromisoformat(p["date"]), "source": p["source"]}
    fmts = sorted(p["formats"] - {None}, key=lambda f: (SIZE_ORDER.get(f, 9), f))
    if fmts:
        out["formats"] = fmts
    if p["flags"]:
        out["flags"] = sorted(p["flags"])
    if p.get("evidence"):
        out["evidence"] = p["evidence"]
    return out


def album_scores(a: dict, albums: list[dict], hf: dict) -> dict:
    """Scores from a repo's eval.json/meta.json, if this album holds that repo's latest pressing."""
    scores = {}
    line_albums = [x for x in albums if x["line"] == a["line"]]
    for fmt, repo in sorted({(f, r) for p in a["pressings"] for f, r in p["repos"].items()}):
        latest = max(
            (x for x in line_albums if any(repo in p["repos"].values() for p in x["pressings"])),
            key=lambda x: x["date"],
        )
        ev = hf.get(repo, {}).get("eval")
        if latest is a and ev:
            vals = {k: round(ev[k] * 100, 2) for k in SCORE_KEYS if isinstance(ev.get(k), (int, float))}
            if vals:
                scores[fmt or "model"] = {"source": f"https://huggingface.co/{repo}/blob/main/{ev['_file']}", **vals}
    return scores


def album_entry(a: dict, line: dict, albums: list[dict], hf: dict) -> dict:
    fm = {
        "title": a["title"],
        "catalog_number": f"LAT-{a['cat']:03d}",
        "cat": a["cat"],
        "date": dt.date.fromisoformat(a["date"]),
        "line": a["line"],
        "lines": [a["line"]],
        "album": a["album"],
        "languages": [a["language"]],
        "kinds": [a["kind"]],
        "status": a["status"],
    }
    repos = {}
    for p in a["pressings"]:
        for f, r in p["repos"].items():
            repos.setdefault(f, r)
    if any(repos):
        fm["formats"] = [
            {"name": f, "huggingface": repos[f]}
            for f in sorted((f for f in repos if f), key=lambda f: (SIZE_ORDER.get(f, 9), f))
        ]
    if "" in repos:
        fm["huggingface"] = repos[""]
    for key in ("pypi", "github"):
        if line.get(key) and not (key == "github" and a["kind"] == "spaCy Pipeline"):
            fm[key] = line[key]
    fm["pressings"] = [pressing_out(p) for p in sorted(a["pressings"], key=lambda p: (p["date"], version_key(p["version"])))]
    scores = album_scores(a, albums, hf)
    if scores:
        fm["scores"] = scores
    return fm


def write_entry(path: Path, fm: dict, body: str = "") -> None:
    text = "---\n" + dump_yaml(fm) + "---\n" + body
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def cmd_migrate(args) -> None:
    if LOCK_FILE.exists():
        sys.exit("numbers are locked (data/catalog.lock.json); renumbering is no longer allowed")
    lines = load_lines()
    hf = json.loads((HARVEST / "hf.json").read_text())["repos"]
    albums, _ = build_albums()
    number_albums(albums)
    by_key = {f"{a['line']}@{a['album']}": a for a in albums}

    # carry covers, legacy notes and body text forward from the current entries
    carried: dict[str, dict] = {}
    for _, (f, old_fm, body) in load_entries().items():
        line = lines[old_fm["line"]]
        versions = [p["version"] for p in old_fm["pressings"] if p.get("version")]
        has_version_album = bool(re.search(r"\d", str(old_fm["album"])))
        new_album = album_of(line, versions[0]) if versions and has_version_album else old_fm["album"]
        c = carried.setdefault(f"{old_fm['line']}@{new_album}", {"notes": [], "covers": {}, "body": []})
        c["notes"] += old_fm.get("legacy_notes", [])
        if old_fm.get("cover"):
            c["covers"][""] = old_fm["cover"]
        for entry in old_fm.get("formats", []):
            if entry.get("cover"):
                c["covers"][entry["name"]] = entry["cover"]
        if body.strip():
            c["body"].append(body.strip())

    plan = []
    for key, a in by_key.items():
        if "cat" not in a:
            continue
        fm = album_entry(a, lines[a["line"]], albums, hf)
        c = carried.get(key, {})
        covers = c.get("covers", {})
        renames = {}
        for fmt, old in covers.items():
            new = f"images/covers/lat-{a['cat']:03d}" + (f"-{fmt}" if fmt else "") + ".png"
            renames[old] = new
            if fmt:
                for entry in fm.get("formats", []):
                    if entry["name"] == fmt:
                        entry["cover"] = new
            else:
                fm["cover"] = new
        if c.get("notes"):
            fm["legacy_notes"] = c["notes"]
        plan.append((a["cat"], fm, renames, "\n\n".join(c.get("body", []))))

    if not args.write:
        for cat, fm, renames, _ in sorted(plan, key=lambda t: t[0]):
            print(f"LAT-{cat:03d} {fm['title']:<28} {len(fm['pressings'])} pressings"
                  + (f"  covers {renames}" if renames else "")
                  + (f"  legacy_notes {len(fm['legacy_notes'])}" if fm.get("legacy_notes") else ""))
        print("dry run; pass --write to replace content/catalog/lat-*.md")
        return

    static = ROOT / "static"
    staged = {}
    for _, _, renames, _ in plan:
        for old, new in renames.items():
            staged[new] = (static / old).read_bytes()
    for f in CATALOG.glob("lat-*.md"):
        f.unlink()
    for f in (static / "images" / "covers").glob("lat-*.png"):
        f.unlink()
    for new, data in staged.items():
        (static / new).write_bytes(data)
    for cat, fm, _, body in plan:
        write_entry(CATALOG / f"lat-{cat:03d}.md", fm, body + "\n" if body else "")

    unnumbered = [
        {"title": a["title"], "line": a["line"], "album": a["album"], "status": a["status"],
         "language": a["language"], "kind": a["kind"],
         "pressings": [pressing_out(p) for p in a["pressings"]]}
        for a in albums if "cat" not in a and a["include"] is True
    ]
    UNNUMBERED_FILE.write_text(dump_yaml({"albums": unnumbered}))
    print(f"wrote {len(plan)} entries, {len(staged)} covers, {len(unnumbered)} unnumbered albums")


# --- entries, lockfile ---------------------------------------------------------

LOCK_FILE = DATA / "catalog.lock.json"
SCHEDULE_FILE = DATA / "schedule.yaml"
# internal blocker notes, keyed "<line> <version>"; gitignored, so optional (absent in CI)
SCHEDULE_NOTES_FILE = ROOT / "private" / "schedule-notes.yaml"
ITEM_STATUSES = {"slated", "ready", "shipped", "slipped", "blocked"}
PRESSING_FLAGS = {"yanked", "prerelease", "hotfix", "skipped"}


def read_entry(path: Path) -> tuple[dict, str]:
    m = re.match(r"---\n(.*?)\n---\n?(.*)\Z", path.read_text(), re.S)
    if not m:
        sys.exit(f"{path}: no front matter")
    return yaml.safe_load(m.group(1)), m.group(2).lstrip("\n")


def load_entries() -> dict[int, tuple[Path, dict, str]]:
    out = {}
    for f in sorted(CATALOG.glob("lat-*.md")):
        fm, body = read_entry(f)
        out[fm["cat"]] = (f, fm, body)
    return out


def album_key(fm: dict) -> str:
    return f"{fm['line']}@{fm['album']}"


def load_lock() -> dict[str, str]:
    return json.loads(LOCK_FILE.read_text())["numbers"] if LOCK_FILE.exists() else {}


def as_date(d) -> dt.date:
    return d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))


# --- check ---------------------------------------------------------------------


def run_check() -> list[str]:
    errors = []
    entries = load_entries()
    lines = load_lines()
    seen_keys = {}
    prev_date = None
    for cat in sorted(entries):
        f, fm, _ = entries[cat]
        where = f.name
        if f.name != f"lat-{cat:03d}.md" or fm.get("catalog_number") != f"LAT-{cat:03d}":
            errors.append(f"{where}: filename / catalog_number / cat disagree")
        for k in ("title", "date", "line", "album", "languages", "kinds", "status", "pressings"):
            if k not in fm:
                errors.append(f"{where}: missing {k}")
        if fm.get("line") not in lines:
            errors.append(f"{where}: line {fm.get('line')!r} not in data/lines.yaml")
        key = album_key(fm)
        if key in seen_keys:
            errors.append(f"{where}: album {key} also numbered {seen_keys[key]}")
        seen_keys[key] = fm["catalog_number"]
        ps = fm.get("pressings") or []
        if not ps:
            errors.append(f"{where}: no pressings")
        for p in ps:
            if not p.get("source") or not p.get("date"):
                errors.append(f"{where}: pressing {p.get('version')} lacks date or source")
            if set(p.get("flags", [])) - PRESSING_FLAGS:
                errors.append(f"{where}: unknown flags {p['flags']}")
        public = [as_date(p["date"]) for p in ps if not {"yanked", "prerelease"} & set(p.get("flags", []))]
        if public and as_date(fm["date"]) != min(public):
            errors.append(f"{where}: date {fm['date']} != first public pressing {min(public)}")
        if prev_date and as_date(fm["date"]) < prev_date:
            errors.append(f"{where}: dated {fm['date']}, earlier than LAT-{cat - 1:03d} ({prev_date})")
        prev_date = as_date(fm["date"])
    for num, key in load_lock().items():
        cat = int(num.split("-")[1])
        if cat not in entries:
            errors.append(f"lockfile: {num} ({key}) has no entry")
        elif album_key(entries[cat][1]) != key:
            errors.append(f"lockfile: {num} is locked to {key}, entry says {album_key(entries[cat][1])}")
    sched = yaml.safe_load(SCHEDULE_FILE.read_text())
    for slot in sched.get("slots", []) + [{"items": sched.get("queue", [])}]:
        for item in slot.get("items", []):
            if item.get("status") not in ITEM_STATUSES:
                errors.append(f"schedule: {item.get('line')} {item.get('version')}: bad status {item.get('status')!r}")
    return errors


def cmd_check(args) -> None:
    errors = run_check()
    for e in errors:
        print(f"ERROR {e}")
    print(f"check: {len(load_entries())} entries, {len(errors)} errors")
    sys.exit(1 if errors else 0)


def cmd_lock(args) -> None:
    """Freeze numbers: add every current entry to the lockfile. Existing mappings never change."""
    lock = load_lock()
    entries = load_entries()
    for cat, (_, fm, _) in sorted(entries.items()):
        num = fm["catalog_number"]
        if num in lock and lock[num] != album_key(fm):
            sys.exit(f"refusing: {num} is locked to {lock[num]}, entry says {album_key(fm)}")
        lock[num] = album_key(fm)
    write_json(LOCK_FILE, {"note": "LAT numbers are permanent once listed here; never edit by hand.",
                           "numbers": dict(sorted(lock.items()))})
    print(f"locked {len(lock)} numbers -> {LOCK_FILE.relative_to(ROOT)}")


# --- refresh (derived fields) ------------------------------------------------------


def install_for(fm: dict, a: dict | None, line: dict, hf: dict, pypi: dict) -> None:
    """Install hints pinned to the album's newest public pressing, only where that wheel/release exists."""
    if a is None:
        return
    for entry in fm.get("formats", []):
        entry.pop("install", None)
        repo = entry.get("huggingface")
        files = hf.get(repo, {}).get("files", [])
        mine = [p for p in a["pressings"] if repo in p["repos"].values() and p["version"]
                and not {"yanked", "prerelease"} & p["flags"]]
        if not mine:
            continue
        latest = max(mine, key=lambda p: version_key(p["version"]))["version"]
        wheel = f"{repo.split('/')[1]}-{latest}-py3-none-any.whl"
        if wheel in files:
            entry["install"] = f"pip install https://huggingface.co/{repo}/resolve/main/{wheel}"
    fm.pop("install", None)
    if line.get("pypi"):
        versions = [p["version"] for p in a["pressings"] if p["version"] and not {"yanked", "prerelease"} & p["flags"]]
        all_public = [v for v, r in (pypi.get(line["pypi"], {}).get("releases") or {}).items()
                      if not r["yanked"] and not PRERELEASE.search(v)]
        if versions:
            newest = max(versions, key=version_key)
            fm["install"] = (f"pip install {line['pypi']}" if newest == max(all_public, key=version_key)
                             else f'pip install "{line["pypi"]}=={newest}"')


def cmd_refresh(args) -> None:
    lines = load_lines()
    hf = json.loads((HARVEST / "hf.json").read_text())["repos"]
    pypi = json.loads((HARVEST / "pypi.json").read_text())["packages"]
    albums, _ = build_albums()
    by_key = {f"{a['line']}@{a['album']}": a for a in albums}
    changed = 0
    for cat, (f, fm, body) in load_entries().items():
        before = yaml.safe_dump(fm)
        a = by_key.get(album_key(fm))
        if a:
            scores = album_scores(a, albums, hf)
            if scores:
                fm["scores"] = scores
            else:
                fm.pop("scores", None)
        install_for(fm, a, lines[fm["line"]], hf, pypi)
        if yaml.safe_dump(fm) != before:
            changed += 1
            if args.write:
                write_entry(f, fm, body)
            else:
                print(f"would update {f.name}")
    print(f"refresh: {changed} entries {'updated' if args.write else 'would change'}")


# --- release (the publish -> label hook) ----------------------------------------------


def update_schedule(line: str, version: str, catalog_number: str | None, write: bool) -> str | None:
    sched_text = SCHEDULE_FILE.read_text()
    sched = yaml.safe_load(sched_text)
    for slot in sched.get("slots", []) + [{"date": "queue", "items": sched.get("queue", [])}]:
        for item in slot.get("items", []):
            v = str(item.get("version", ""))
            if item.get("line") == line and item.get("status") != "shipped" and (
                v == version or (v.endswith(".x") and version.startswith(v[:-1]))
            ):
                if write:
                    # targeted text edit keeps the file's comments and layout
                    old = re.compile(
                        rf"(\{{line: {re.escape(line)}, version: \"{re.escape(v)}\".*?status: )(\w+)")
                    shipped = f"shipped, shipped_as: \"{catalog_number or version}\""
                    sched_text = old.sub(lambda m: m.group(1) + shipped, sched_text, count=1)
                    SCHEDULE_FILE.write_text(sched_text)
                return f"schedule: {line} {v} ({slot.get('date')}) -> shipped"
    return None


def cmd_release(args) -> None:
    lines = load_lines()
    if args.line not in lines:
        sys.exit(f"unknown line {args.line!r}: add it to data/lines.yaml first")
    line = lines[args.line]
    version, date = args.version, dt.date.fromisoformat(args.date)
    album = album_of(line, version)
    flags = set(args.flags or [])
    if PRERELEASE.search(version):
        flags.add("prerelease")
    source = args.source or (f"https://pypi.org/project/{line['pypi']}/{version}/" if line.get("pypi") else None)
    if not source:
        sys.exit("--source is required for non-PyPI lines (HF commit or tag URL)")
    pressing = {"version": version, "date": date, "source": source}
    if args.formats:
        pressing["formats"] = args.formats
    if flags:
        pressing["flags"] = sorted(flags)

    entries = load_entries()
    existing = [(cat, e) for cat, e in entries.items() if album_key(e[1]) == f"{args.line}@{album}"]
    unnumbered = yaml.safe_load(UNNUMBERED_FILE.read_text())
    advance = [u for u in unnumbered["albums"] if f"{u['line']}@{u['album']}" == f"{args.line}@{album}"]
    actions = []
    if existing:
        cat, (f, fm, body) = existing[0]
        if any(p.get("version") == version for p in fm["pressings"]):
            sys.exit(f"{fm['catalog_number']} already has pressing {version}")
        if date < as_date(fm["date"]):
            sys.exit(f"pressing dated {date} precedes {fm['catalog_number']}'s first pressing ({fm['date']})")
        fm["pressings"].append(pressing)
        actions.append(f"new pressing {version} on {fm['catalog_number']} {fm['title']}")
        number = fm["catalog_number"]
    elif "prerelease" in flags or (advance and advance[0]["status"] == "withdrawn" and "yanked" in flags):
        if advance:
            advance[0]["pressings"].append(pressing)
        else:
            unnumbered["albums"].append({"title": f"{args.line} {album}".strip(), "line": args.line, "album": album,
                                         "status": "advance", "language": line["language"], "kind": line["kind"],
                                         "pressings": [pressing]})
        actions.append(f"unnumbered pressing {version} (no catalog number until a public pressing)")
        number, f = None, None
    else:
        cat = max([*entries, *[int(n.split("-")[1]) for n in load_lock()]], default=0) + 1
        last_date = max((as_date(e[1]["date"]) for e in entries.values()), default=date)
        if date < last_date:
            sys.exit(f"new album dated {date} precedes the newest album ({last_date}); numbers must stay chronological")
        number = f"LAT-{cat:03d}"
        fm = {
            "title": f"{args.line} {album}".strip(), "catalog_number": number, "cat": cat, "date": date,
            "line": args.line, "lines": [args.line], "album": album,
            "languages": [line["language"]], "kinds": [line["kind"]], "status": "released",
        }
        if line.get("formats"):
            fm["formats"] = [{"name": n, "huggingface": r} for n, r in line["formats"].items()]
        if line.get("hf"):
            fm["huggingface"] = line["hf"]
        for key in ("pypi", "github"):
            if line.get(key) and not (key == "github" and line["kind"] == "spaCy Pipeline"):
                fm[key] = line[key]
        fm["pressings"] = [pressing]
        if advance:  # earlier rc/yanked pressings move onto the newly numbered album
            fm["pressings"] = advance[0]["pressings"] + [pressing]
            unnumbered["albums"].remove(advance[0])
            actions.append(f"carried {len(fm['pressings']) - 1} unnumbered pressing(s) onto the album")
        f, body = CATALOG / f"lat-{cat:03d}.md", ""
        actions.append(f"new album {number} {fm['title']} ({date})")
    sched = update_schedule(args.line, version, number, args.write)
    if sched:
        actions.append(sched)

    for a in actions:
        print(a)
    if not args.write:
        print("dry run; pass --write to apply")
        return
    if f is not None:
        write_entry(f, fm, body)
    if f is None or advance:
        UNNUMBERED_FILE.write_text(dump_yaml(unnumbered))
    if number and LOCK_FILE.exists() and not existing:
        lock = load_lock()
        lock[number] = f"{args.line}@{album}"
        write_json(LOCK_FILE, {"note": "LAT numbers are permanent once listed here; never edit by hand.",
                               "numbers": dict(sorted(lock.items()))})
    errors = run_check()
    for e in errors:
        print(f"ERROR {e}")
    sys.exit(1 if errors else 0)


# --- audit (drift detection) -----------------------------------------------------------


def cmd_audit(args) -> None:
    if not args.cached:
        cmd_harvest(argparse.Namespace(source="all"))
    albums, _ = build_albums()
    catalogued = set()
    for _, fm, _ in load_entries().values():
        for p in fm["pressings"]:
            catalogued.add((album_key(fm), p.get("version"), str(p["date"]) if p.get("version") is None else None))
    for a in yaml.safe_load(UNNUMBERED_FILE.read_text())["albums"]:
        for p in a["pressings"]:
            catalogued.add((f"{a['line']}@{a['album']}", p.get("version"), None))
    missing = []
    for a in albums:
        if a["include"] is not True:
            continue
        for p in a["pressings"]:
            k = (f"{a['line']}@{a['album']}", p["version"], p["date"] if p["version"] is None else None)
            if k not in catalogued:
                missing.append(f"{k[0]} {p['version'] or '(unversioned)'} {p['date']} {p['source']}")
    for m in missing:
        print(f"UNCATALOGUED {m}")
    bad_links = []
    if args.links:
        urls = set()
        for _, fm, _ in load_entries().values():
            urls |= {p["source"] for p in fm["pressings"] if p["source"].startswith("http")}
            urls |= {f"https://huggingface.co/{x['huggingface']}" for x in fm.get("formats", [])}
            if fm.get("huggingface"):
                urls.add(f"https://huggingface.co/{fm['huggingface']}")
            if fm.get("pypi"):
                urls.add(f"https://pypi.org/project/{fm['pypi']}/")
            if fm.get("github"):
                urls.add(f"https://github.com/{fm['github']}")
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for u in sorted(urls):
                try:
                    code = client.head(u).status_code
                    if code == 405:
                        code = client.get(u).status_code
                except httpx.HTTPError as e:
                    code = type(e).__name__
                if code != 200:
                    bad_links.append(f"{code} {u}")
                    print(f"LINK {code} {u}")
        print(f"links: {len(urls)} checked, {len(bad_links)} not OK")
    print(f"audit: {len(missing)} uncatalogued pressings")
    sys.exit(1 if missing or bad_links else 0)


# --- schedule ----------------------------------------------------------------------


def nth_weekday(year: int, month: int, nth: int, weekday: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta((weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=nth - 1)


def cmd_schedule(args) -> None:
    sched = yaml.safe_load(SCHEDULE_FILE.read_text())
    today = dt.date.today()
    rule = sched["release_day"]
    slots = sorted(sched.get("slots", []), key=lambda s: s["date"])
    if args.action == "next":
        upcoming = [s for s in slots if as_date(s["date"]) >= today]
        if not upcoming:
            d = nth_weekday(today.year, today.month, rule["nth"], rule["weekday"])
            if d < today:
                m = today.month % 12 + 1
                d = nth_weekday(today.year + (m == 1), m, rule["nth"], rule["weekday"])
            print(f"next release day {d}: no slot defined")
            return
        s = upcoming[0]
        days = (as_date(s["date"]) - today).days
        print(f"next release day {s['date']} (in {days} days){'' if s.get('approved') else ' [proposed]'}")
        if days <= 7:
            print("  slate freeze: T-7 has passed — only 'ready' items should ship")
        notes = yaml.safe_load(SCHEDULE_NOTES_FILE.read_text()) if SCHEDULE_NOTES_FILE.exists() else {}
        for it in s["items"]:
            note = notes.get(f"{it['line']} {it['version']}")
            print(f"  {it['status']:<8} {it['line']} {it['version']}" + (f"  — {note}" if note else ""))
    elif args.action == "roll":
        past = [s for s in slots if as_date(s["date"]) < today]
        upcoming = [s for s in slots if as_date(s["date"]) >= today]
        moved = [(s["date"], it) for s in past for it in s["items"] if it["status"] != "shipped"]
        if not moved:
            print("roll: nothing to move")
            return
        if not upcoming:
            sys.exit("roll: no upcoming slot to roll into; add one to data/schedule.yaml")
        for d, it in moved:
            print(f"roll: {it['line']} {it['version']} {d} -> {upcoming[0]['date']} (slipped)")
        if args.write:
            for s in past:
                s["items"] = [it for it in s["items"] if it["status"] == "shipped"]
            upcoming[0]["items"] += [dict(it, status="slipped") for _, it in moved]
            sched["slots"] = slots
            SCHEDULE_FILE.write_text(dump_yaml(sched))
        else:
            print("dry run; pass --write to apply (note: rewrites schedule.yaml without comments)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="refresh evidence cache")
    h.add_argument("--source", choices=["all", "hf", "pypi", "github"], default="all")
    h.set_defaults(func=cmd_harvest)
    sub.add_parser("propose", help="write scratch/numbering-proposal.md").set_defaults(func=cmd_propose)
    m = sub.add_parser("migrate", help="rewrite content/catalog under the current numbering rules (pre-lock only)")
    m.add_argument("--write", action="store_true")
    m.set_defaults(func=cmd_migrate)
    sub.add_parser("check", help="validate entries, lockfile, schedule").set_defaults(func=cmd_check)
    sub.add_parser("lock", help="freeze current numbers into data/catalog.lock.json").set_defaults(func=cmd_lock)
    r = sub.add_parser("refresh", help="re-derive scores/install hints from the harvest")
    r.add_argument("--write", action="store_true")
    r.set_defaults(func=cmd_refresh)
    rel = sub.add_parser("release", help="record a publish: new pressing or new album")
    rel.add_argument("line")
    rel.add_argument("version")
    rel.add_argument("--date", default=dt.date.today().isoformat())
    rel.add_argument("--formats", nargs="+")
    rel.add_argument("--source", help="evidence URL (defaults to the PyPI release page for PyPI lines)")
    rel.add_argument("--flags", nargs="+", choices=sorted(PRESSING_FLAGS))
    rel.add_argument("--write", action="store_true")
    rel.set_defaults(func=cmd_release)
    au = sub.add_parser("audit", help="list public pressings missing from the catalog")
    au.add_argument("--cached", action="store_true", help="use data/harvest instead of re-harvesting")
    au.add_argument("--links", action="store_true", help="also check every link")
    au.set_defaults(func=cmd_audit)
    s = sub.add_parser("schedule", help="release calendar")
    s.add_argument("action", choices=["next", "roll"])
    s.add_argument("--write", action="store_true")
    s.set_defaults(func=cmd_schedule)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
