# Native backend source provenance

`codim-directional-source.tar.gz` contains the source tree used by the project's directional fixture backend, captured on 2026-11-02. It contains no compiled extension, environment, credentials or solver output. The archive includes upstream LICENSE files. Project modifications include plastic crease memory, rollback/controls, soft directional fixtures and solver tolerance controls.

Upstream: https://github.com/ipc-sim/Codim-IPC ; historical project source pin: `9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a`. The captured modified tree was not itself an independent Git checkout; the archive SHA256 in `release-manifest.json` is the authoritative identity for this release, not a claimed clean upstream commit. Do not interpret the parent server directory's Git commit as backend provenance.

Dependency pins recorded in the source tree:

- amgcl: `1fbc24e687b404962c31bc4db598f6c73c0f4333`
- Kokkos: `7933c2e4b9c83a4d99101acbf2ae06a1115f50d5`
- Cabana: `c33c8a00b42560129e9953eac0d0f4109624c90d`
- Kokkos kernels: `07a60bcc136d4db46f1a05102762a0a17c07049f`

These sources and their licenses are bundled, including the upstream Python drivers and pybind11 components. They are covered by their respective licenses, not an invented blanket project license. C-IPC publication: https://ipc-sim.github.io/C-IPC/ . Published solver capabilities do not certify this project's tasks or controllers; local validation must be checked separately.
