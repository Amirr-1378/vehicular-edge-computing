# GitHub Actions: visible simulation output

This version keeps the original Python execution model but makes the output easier to inspect in GitHub Actions.

## What changed

The workflow now shows the simulation result in three places:

1. **Job log**  
   Open the workflow run, then open the `run-main` job and the step named `Run main simulation and save visible output`.

2. **GitHub Actions Summary**  
   Open the completed workflow run and check the Summary page. The workflow appends the full console output to `$GITHUB_STEP_SUMMARY`.

3. **Downloadable artifact**  
   The run uploads `reports/run-output.txt` as an artifact named `simulation-run-output`.

## Important file path

The workflow must be committed at:

```text
.github/workflows/python-package.yml
```

If the workflow file is placed in the repository root, GitHub Actions will not detect it as a workflow.

## Manual run

After pushing to GitHub:

1. Go to the repository on GitHub.
2. Open the **Actions** tab.
3. Select **Main Procedure Run**.
4. Click **Run workflow**.
5. After completion, inspect Summary, job logs, or the artifact.
