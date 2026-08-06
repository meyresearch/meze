# Contributing to `meze`

Thanks for your interest in contributing! This page covers how to set up a development environment, the conventions used in this repo, and how to submit changes.

## Development environment setup

```bash
git clone https://github.com/meyresearch/meze.git
cd meze
mamba env create -f dev-environment.yml
mamba activate meze-dev
```

`dev-environment.yml` installs `meze` itself in editable mode (`pip install -e .`) as part of environment creation, along with `pytest`, `pytest-cov`, `flake8`, and `ambertools` for local testing/linting — no separate install step needed.

> [!IMPORTANT]
> Run `git pull` after pulling in new updates, and re-run `mamba env update -f dev-environment.yml --prune` if `dev-environment.yml` itself changed.

## Running tests

```bash
pytest tests/ -m "not slow and not gpu" --cov=meze --cov-report=term-missing
```

This is the same command CI runs. Tests marked `@pytest.mark.slow` or `@pytest.mark.gpu` require real MD/BSS runs or a GPU + `$AMBERHOME`, and are skipped by default — drop the `-m` filter to run them manually if you're working in that area.

## Linting

```bash
flake8 meze/ --select=E9,F63,F7,F82 --show-source
```

This is the check CI treats as build-breaking: syntax errors, illegal control flow, and undefined names — genuine bugs, not style. CI also runs a second, non-blocking pass reporting style and complexity issues; worth a glance in the Actions log, but it won't fail your PR.

## Branch naming and commits

Branches follow `<type>/<short-description>`, e.g. `feat/analysis-statistics`, `fix/license-mismatch`, `docs/usage-guide`. Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `build`, `examples`.

Commit messages follow the same `type: description` convention, e.g. `fix: correct restraint mask indexing`.

See [this](https://medium.com/@abhay.pixolo/naming-conventions-for-git-branches-a-cheatsheet-8549feca2534) great blog post for examples of good names for the types of branches and commits. 

## Submitting changes

1. Branch off `main`.
2. Keep PRs scoped to one logical change — easier to review, easier to bisect later.
3. Add or update tests for anything you change in `meze/`.
4. Make sure `pytest` and the build-breaking `flake8` pass locally before opening the PR — CI runs both automatically, but catching issues locally saves a round trip.
5. Open the PR against `main` and fill in the PR template.

## Reporting bugs / requesting features

Use the issue templates: [bug report](.github/ISSUE_TEMPLATE/bug_report.md), [feature request](.github/ISSUE_TEMPLATE/feature_request.md), or [custom](.github/ISSUE_TEMPLATE/custom.md).

## Questions

Open a GitHub issue — this is currently the primary support channel.
