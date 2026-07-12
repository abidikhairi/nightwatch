from pathlib import Path

from nightwatch.app import NightwatchApp


async def test_app_starts(tmp_path: Path) -> None:
    app = NightwatchApp(db_path=tmp_path / "db.sqlite3")
    async with app.run_test():
        assert app.title == "nightwatch"
