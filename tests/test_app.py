from nightwatch.app import NightwatchApp


async def test_app_starts() -> None:
    app = NightwatchApp()
    async with app.run_test():
        assert app.title == "nightwatch"
