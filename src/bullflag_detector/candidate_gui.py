"""Compatibility shim.

The interactive matplotlib GUI has been removed.

Use the non-interactive batch exporter instead:
	bullflag-export-labelstudio
"""


def main() -> None:
	raise SystemExit(
		"The GUI labeler has been removed. Use 'bullflag-export-labelstudio' to export candidates."
	)


__all__ = ["main"]
