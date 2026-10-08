# Appearance and task recovery

The interface defaults to a dark workspace. Select an appearance for the current tab:

- Dark: `http://127.0.0.1:7788/?appearance=dark`
- Classic: `http://127.0.0.1:7788/?appearance=classic`

Use your deployment's hostname when appropriate. The tab remembers an explicit
choice through navigation and refresh. Appearance changes do not alter molecular
coordinates, scientific settings, checkpoints or task storage.

Refreshing restores the tab's saved task through the server resume API and checks
the available checkpoints. Without a saved task, a workflow URL opens its first
step. From another tab or device, enter the Task ID on the home page. Unsubmitted
edits are not checkpoints; an interrupted preparation may require Check again.

Administrators can change the installation default from its source directory:

```bash
python scripts/set_frontend_appearance.py classic
```

Use `dark` to restore the default. A page refresh loads the new appearance; tabs
with an explicit preference retain it. No scientific-worker restart is needed.
