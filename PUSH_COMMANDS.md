# Git commands to push the fixed workflow

Run these commands from the root of your local GitHub repository.

## Option A: if you copied the fixed files into the repository root

```bash
git status

git add .github/workflows/python-package.yml requirements.txt main.py mec_side.py user_side.py README_GITHUB_ACTIONS_VISIBLE_OUTPUT.md

git commit -m "Fix GitHub Actions simulation output visibility"

git pull --rebase origin main

git push origin main
```

## Option B: if you only want to replace the workflow file

```bash
mkdir -p .github/workflows
cp ~/Downloads/python-package-visible-output.yml .github/workflows/python-package.yml

git status

git add .github/workflows/python-package.yml

git commit -m "Show simulation output in GitHub Actions summary and artifact"

git pull --rebase origin main

git push origin main
```

## If your active branch is not main

Check the current branch:

```bash
git branch --show-current
```

Then push to that branch:

```bash
git push origin HEAD
```

Note: the workflow above is configured to run automatically on push to `main`. It can still be run manually from the Actions tab because `workflow_dispatch` is enabled.
