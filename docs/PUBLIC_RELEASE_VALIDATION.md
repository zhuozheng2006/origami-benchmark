# Public release validation — 2026-11-02

## Independently built physical runtime

Built the bundled native source in a new directory, without the original project's compiled backend or private assets. Used a dedicated Python venv (base interpreter/site packages inherited, requested requirements installed separately; not a fresh OS/container). Linux x86_64, kernel 5.4.0, glibc 2.31, Python 3.10.22. NumPy 1.26.4, SciPy 1.15.3, Pillow 11.3.0, Matplotlib 3.10.9, ipctk 1.6.1.

The compiled JGSL SHA256 is `db57b67af7e7c1a4a86ed83ece8f5d2d5a6a268a6ec6617481b12a11d37fd7a8`, matching the historical directional backend binary. This is evidence for this build environment only, not a guarantee of binary-identical builds on other systems.

All three core tasks were prepared and physically replayed using the new local subprocess transport and the same accounted harness as user agents. No saved target was teleported into the simulator. No paid model calls were made.

| Task | Result | Attempts incl. finish | RMS mm | Max mm | Seconds incl. prep |
|---|---|---:|---:|---:|---:|
| O01 | passed_scoped | 5 | 0.127598 | 0.357626 | 129.6 |
| O02 | passed_scoped | 7 | 0.191667 | 0.503448 | 323.2 |
| O03 | passed_scoped | 7 | 0.507036 | 1.232354 | 453.0 |

Raw reports: [O01](validation/O01-reference.json), [O02](validation/O02-reference.json), [O03](validation/O03-reference.json). These are oracle checks, not model scores. The executed candidate manifest is retained as [executed-rc-manifest.json](validation/executed-rc-manifest.json). Final packaging adds error-report persistence, manifest metadata, documentation and tests; physical solver/controller/grader and task assets are unchanged from these runs.

## Negative control and failures

The generic finish-only agent was dispatched through the plugin interface. It used one attempt and was correctly scored `not_passed`: [report](validation/O01-finish-only.json). No API or answer lookup was involved.

The initial O01 launch occurred before native linking had finished and failed with `ImportError: file too short`. Its log is preserved at [initial-build-race.log](validation/initial-build-race.log). The second run used a new output directory after the build completed. No old attempt was overwritten. This was an environment startup failure, not a model failure or task failure. The documented installation command runs the build in the foreground before evaluation.

## Offline checks

Four tests cover three historical reference rescores, missing finish, missing physical validity, immutable failure/undo budget history, and task asset completeness. All passed locally. The bundled self-contained HTML replay was opened in the browser and its slider/geometry verified; the camera framing was corrected. The source archive and regular files were scanned for configured credentials and common secret signatures.

## Limits

This does not verify arbitrary Linux distributions, macOS native simulation, Windows native simulation, or arbitrary provider APIs. O04 is diagnostic only. The collection contains three related development calibration tasks, not a large independently held-out benchmark. Existing historical pose/path grading limitations are documented in the README. Re-scoring records is not a substitute for trusted physical replay when submissions may be tampered with.
