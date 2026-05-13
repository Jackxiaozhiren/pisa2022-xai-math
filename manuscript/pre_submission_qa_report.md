# Pre-Submission QA Report

Date: 2026-05-11

## Current Status

Status: technically ready for official-template proof review, with only
institution-specific ethics wording still requiring human action.

The manuscript package, result artifacts, references, reproducibility notes, and
public-release boundary files have been checked in the local workspace. The
author metadata and official Springer template have been integrated. The
remaining blockers are not code or analysis failures; they require institution
ethics wording confirmation or Zotero Desktop local API authorization.

## Validation Commands

All commands were run from `/Users/jackson/论文/pisa2022-xai-math/workspace`
unless noted otherwise.

```bash
.venv/bin/python -m py_compile src/pisa_xai/*.py scripts/*.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python scripts/00_check_inputs.py
.venv/bin/python scripts/05_build_tables.py
python3 - <<'PY'
from pathlib import Path
import re
text = Path('manuscript/manuscript.md').read_text()
bib = Path('manuscript/references.bib').read_text()
md_keys = {m.group(1).rstrip('.,;:)\\]') for m in re.finditer(r'@([A-Za-z0-9_:-]+)', text)}
bib_keys = set(re.findall(r'@\\w+\\s*\\{\\s*([^,]+)', bib))
assert not (md_keys - bib_keys)
assert not (bib_keys - md_keys)
PY
sips -g pixelWidth -g pixelHeight reports/figures/*.png
rg -n --hidden --glob '!/.venv/**' --glob '!data/raw/**' --glob '!data/processed/**' --glob '!docs/sources/**' --glob '!manuscript/build/**' --glob '!manuscript/public_release_audit.md' --glob '!manuscript/reference_audit.md' --glob '!manuscript/pre_submission_qa_report.md' --glob '!**/*.pdf' --glob '!**/*.png' '(sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|SUPABASE_(SERVICE_ROLE|ANON)_KEY|NEON_[A-Z_]*KEY|password\\s*=|passwd\\s*=|api[_-]?key\\s*=|secret\\s*=|token\\s*=)' .
```

The proof PDF was compiled from `manuscript/`:

```bash
/Users/jackson/.codex/plugins/cache/openai-bundled/latex-tectonic/0.1.1/bin/tectonic --keep-logs --keep-intermediates --outdir build springer_submission.tex
```

## Results

- Python compilation: pass.
- Unit tests: pass, `8` tests.
- Input check: pass for the student file; school file detected and available.
- Table/figure index rebuild: pass, `20` CSV tables and `3` PNG figures listed.
- Citation-key consistency: pass, `26` manuscript keys and `26` BibTeX entries.
- Figure source dimensions: pass; all three PNG figures are at least 2,000 px
  wide.
- Official-template PDF: pass; `manuscript/build/springer_submission.pdf`, `14`
  pages.
- BibTeX proof log: pass; no undefined citation warnings and no BibTeX metadata
  warnings in the final kept-log run.
- Sensitive-text scan: pass for release-relevant text surfaces after excluding
  `.venv`, raw/processed data, source downloads, build outputs, and audit files.
- Public-release package: pass; cleaned package prepared at
  `/Users/jackson/论文/pisa2022-xai-math/public_release/pisa2022-xai-math`.

## Remaining Human/External Blockers

- Confirm institution-specific ethics/public-data exemption wording.
- Allow Zotero Desktop local API access or manually export BibTeX if a
  Zotero-export reconciliation is required.
- Exclude OECD raw data and row-level or fitted-state derived artifacts from any
  public repository unless release policy explicitly permits them.
