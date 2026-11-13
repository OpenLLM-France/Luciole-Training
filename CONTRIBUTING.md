# Contributing to Luciole-Training

Thank you for your interest in contributing to the Luciole training pipeline,
part of the [OpenLLM-France](https://github.com/OpenLLM-France) initiative.

## Ways to contribute

- Report bugs or suggest improvements by opening an issue.
- Improve the documentation.
- Submit code changes through pull requests.

## Reporting issues

Before opening an issue, please search the existing issues to avoid
duplicates. For a bug report, include the affected pipeline stage, the command
you ran, the expected and actual behavior, and any relevant logs or
environment details (cluster, Python environment, package versions).

For security issues, please do not open a public issue; follow
[SECURITY.md](SECURITY.md) instead.

## Development workflow

1. Fork the repository and create a topic branch from `main`.
2. Keep each change focused and self-contained.
3. Run the linters and formatter before committing:

   ```bash
   pre-commit run --all-files
   # or directly:
   ruff check . --fix
   ruff format .
   ```

   Note that `pretrain/train/patched/` is excluded from formatting.
4. Write clear, neutral commit messages describing what changed and why.
5. Open a pull request against `main`, describing your change and linking any
   related issue.

## License of contributions

This project is licensed under the GNU General Public License v3.0
(see [LICENSE](LICENSE)). By contributing, you agree that your contributions
will be licensed under the same terms. See [NOTICE](NOTICE) for copyright and
attribution.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.
