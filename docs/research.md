# Public release portability research (2026-11-02)

Primary reference: https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html . CMake defines this as a semicolon-separated list of installation prefixes searched by find_package and related commands.

Local observation: the first base backend configure failed to find Eigen3 because an isolated Python venv prefix omitted its parent Conda dependency prefix. A second attempt exposed a shell/Python quoting mistake. Both failed build logs are preserved. The installers now supply sys.prefix and sys.base_prefix using a shell-safe Python expression; this only changes dependency discovery, not solver mathematics or acceptance thresholds. The third attempt reached native compilation and linking. Final validation status is recorded separately in PUBLIC_RELEASE_V2_VALIDATION.md.

Both backend source archive hashes pin modified source trees and bundled dependency sources. An upstream capability claim does not constitute a local task validation. The legacy and layered backends remain separate and their scores must not be pooled.
