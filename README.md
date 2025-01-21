[![pre-commit](https://github.com/daniel-gallo/ssm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/daniel-gallo/ssm/actions/workflows/pre-commit.yml)
# Dev instructions
As linter / formatter, we can use [Ruff](https://docs.astral.sh/ruff/).
## Zed instructions
Add this to `~/.config/zed/settings.json`

```json
{
    "languages": {
        "Python": {
            "format_on_save": "on",
            "formatter": [
                {
                    "code_actions": {
                        "source.organizeImports.ruff": true,
                        "source.fixAll.ruff": true
                    }
                },
                {
                    "language_server": {
                        "name": "ruff"
                    }
                }
            ]
        }
    }
}

```
## Pre-commit hook
This ensures that your commits will stick to a pre-defined coding style.
1. Install `pre-commit` (using pip for example)
1. Run `pre-commit install`
1. (Optional) run `pre-commit run --all-files`

After installing the hook, some checks will be performed automatically before you commit your files.
