# Contributing Guide

**Project:** HalluScan — PhD Research Implementation  
**Author:** Chalani Dinitha (20211032)

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code only |
| `feat/xxx` | New features |
| `test/xxx` | Test additions |
| `research/xxx` | Experiments and evaluation |
| `docs/xxx` | Documentation only |
| `fix/xxx` | Bug fixes |

---

## Commit Conventions

Format: `type(scope): description`

| Type | When |
|------|------|
| `feat` | New feature |
| `test` | Adding tests |
| `fix` | Bug fix |
| `docs` | Documentation |
| `research` | Experiment/evaluation |
| `chore` | Build scripts, config |
| `refactor` | Code restructure |

**Examples:**
```
feat(module-a): implement EAT detection using spaCy NER
test(pipeline): add integration tests covering NFR1, NFR4
fix(tsv): resolve shape mismatch (64,) vs (2048,)
docs: update README with setup instructions
research(ablation): run 4-condition ablation study
```

---

## Pull Request Process

1. Create branch from `main`
2. Make changes + write tests
3. Run `pytest backend/tests/ -v`
4. Create PR with descriptive title
5. Merge after tests pass

---

## Test Requirements

- All new code must have unit tests
- Pass rate must remain ≥ 100%
- Run before every PR: `pytest backend/tests/ -v`
