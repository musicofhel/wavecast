"""Playwright E2E tests for the WaveCast Streamlit dashboard.

Launches Streamlit, navigates to each page, and verifies content renders.
Run: .venv/bin/pytest tests/test_dashboard_pages.py -v
"""

from __future__ import annotations

import subprocess
import time
import urllib.request

import pytest
from playwright.sync_api import Page, expect, sync_playwright

STREAMLIT_PORT = 8502
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
STARTUP_TIMEOUT = 15

# Streamlit multipage URL routes (numeric prefix stripped from filenames)
PAGE_ROUTES = {
    "Forward Test": "/forward_test",
    "Production System": "/production",
    "Ticker Drilldown": "/drilldown",
    "Experiment Archive": "/experiments",
}


@pytest.fixture(scope="module")
def streamlit_server():
    """Launch Streamlit server for the test session."""
    proc = subprocess.Popen(
        [
            "/home/musicofhel/wavecast/.venv/bin/streamlit",
            "run",
            "/home/musicofhel/wavecast/dashboard/app.py",
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.address",
            "localhost",
            "--server.headless",
            "true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    for _ in range(STARTUP_TIMEOUT * 2):
        try:
            resp = urllib.request.urlopen(
                f"{STREAMLIT_URL}/_stcore/health", timeout=2
            )
            if resp.read() == b"ok":
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Streamlit did not start within timeout")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser_page(streamlit_server):
    """Create a Playwright browser page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


def _goto_page(page: Page, page_name: str):
    """Navigate to a page by its URL route.

    Streamlit renders via WebSocket after initial page load, so we must
    wait for the script to finish running (indicated by the status widget
    disappearing) rather than relying solely on networkidle.
    """
    route = PAGE_ROUTES[page_name]
    page.goto(f"{STREAMLIT_URL}{route}")
    page.wait_for_load_state("networkidle")
    # Wait for Streamlit to finish running (status indicator disappears)
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=15000)
    time.sleep(1)


def _assert_no_errors(page: Page, page_name: str):
    """Assert no Streamlit exception banners on the page."""
    errors = page.locator('[data-testid="stException"]')
    assert errors.count() == 0, (
        f"Error on page '{page_name}': "
        f"{errors.first.text_content() if errors.count() > 0 else ''}"
    )


# ── Page 1: Forward Test ─────────────────────────────────────────────────────


class TestForwardTestPage:
    def test_loads_without_errors(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        _assert_no_errors(browser_page, "Forward Test")

    def test_title(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(browser_page.locator("h1").filter(has_text="Forward Test")).to_be_visible()

    def test_metric_cards(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        for label in ["Total Predictions", "Resolved", "Pending"]:
            expect(browser_page.get_by_text(label).first).to_be_visible()

    def test_equity_curve(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(browser_page.get_by_text("Equity Curve").first).to_be_visible()

    def test_rolling_accuracy(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(browser_page.get_by_text("Rolling Accuracy").first).to_be_visible()

    def test_recent_predictions(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(browser_page.get_by_text("Recent Predictions").first).to_be_visible()

    def test_refresh_button(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(
            browser_page.get_by_role("button", name="Refresh")
        ).to_be_visible()

    def test_run_log_expander(self, browser_page):
        _goto_page(browser_page, "Forward Test")
        expect(browser_page.get_by_text("Run Log").first).to_be_visible()


# ── Page 2: Production System ────────────────────────────────────────────────


class TestProductionPage:
    def test_loads_without_errors(self, browser_page):
        _goto_page(browser_page, "Production System")
        _assert_no_errors(browser_page, "Production System")

    def test_title(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.locator("h1").filter(has_text="Production System")
        ).to_be_visible()

    def test_model_caption(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(browser_page.get_by_text("d1_augmented_v1").first).to_be_visible()

    def test_2026_vs_2025(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.get_by_text("2026 OOS vs 2025 Test").first
        ).to_be_visible()

    def test_config_bars(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.get_by_text("8-Config Sharpe Comparison").first
        ).to_be_visible()

    def test_head_to_head(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.get_by_text("A2i vs B2i").first
        ).to_be_visible()

    def test_reversal_warning(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.get_by_text("Reversal Filter Failure").first
        ).to_be_visible()

    def test_crosstab(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(browser_page.get_by_text("Cross-Tab").first).to_be_visible()

    def test_system_spec_expander(self, browser_page):
        _goto_page(browser_page, "Production System")
        expect(
            browser_page.get_by_text("System Specification").first
        ).to_be_visible()


# ── Page 3: Ticker Drilldown ─────────────────────────────────────────────────


class TestDrilldownPage:
    def test_loads_without_errors(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        _assert_no_errors(browser_page, "Ticker Drilldown")

    def test_title(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        expect(
            browser_page.locator("h1").filter(has_text="Ticker Drilldown")
        ).to_be_visible()

    def test_ticker_selector(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        # Default selection is AAPL, sector label should appear
        expect(browser_page.get_by_text("tech").first).to_be_visible()

    def test_metric_cards(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        for label in ["Accuracy", "Sharpe"]:
            expect(browser_page.get_by_text(label).first).to_be_visible()

    def test_latest_forecast(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        expect(
            browser_page.get_by_text("Latest Forecast").first
        ).to_be_visible()

    def test_ranking_chart(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        expect(
            browser_page.get_by_text("Ranked by Sharpe").first
        ).to_be_visible()

    def test_all_predictions_table(self, browser_page):
        _goto_page(browser_page, "Ticker Drilldown")
        expect(
            browser_page.get_by_text("All Predictions").first
        ).to_be_visible()


# ── Page 4: Experiment Archive ───────────────────────────────────────────────


class TestExperimentsPage:
    def test_loads_without_errors(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        _assert_no_errors(browser_page, "Experiment Archive")

    def test_title(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        expect(
            browser_page.locator("h1").filter(has_text="Experiment Archive")
        ).to_be_visible()

    def test_summary_stats(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        # Summary caption with experiment counts
        expect(browser_page.get_by_text("experiments").first).to_be_visible()
        expect(browser_page.get_by_text("PASS").first).to_be_visible()

    def test_results_table(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        expect(browser_page.get_by_text("Results").first).to_be_visible()

    def test_scatter_chart(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        expect(
            browser_page.get_by_text("Econ Dir vs Transition").first
        ).to_be_visible()

    def test_detail_expander(self, browser_page):
        _goto_page(browser_page, "Experiment Archive")
        expect(
            browser_page.get_by_text("Experiment Detail").first
        ).to_be_visible()


# ── Landing page & navigation ────────────────────────────────────────────────


class TestLandingPage:
    def test_loads_without_errors(self, browser_page):
        browser_page.goto(STREAMLIT_URL)
        browser_page.wait_for_load_state("networkidle")
        time.sleep(2)
        _assert_no_errors(browser_page, "Landing")

    def test_title(self, browser_page):
        browser_page.goto(STREAMLIT_URL)
        browser_page.wait_for_load_state("networkidle")
        time.sleep(2)
        expect(
            browser_page.locator("h1").filter(has_text="WaveCast Dashboard")
        ).to_be_visible()


class TestNavigation:
    def test_sidebar_has_all_pages(self, browser_page):
        browser_page.goto(STREAMLIT_URL)
        browser_page.wait_for_load_state("networkidle")
        time.sleep(2)
        sidebar = browser_page.locator('[data-testid="stSidebarNav"]')
        for name in [
            "forward test",
            "production",
            "drilldown",
            "experiments",
        ]:
            expect(sidebar.get_by_text(name, exact=False)).to_be_visible()

    def test_cycle_all_pages_no_errors(self, browser_page):
        """Navigate to every page by URL and verify no exceptions render."""
        for name in [
            "Forward Test",
            "Production System",
            "Ticker Drilldown",
            "Experiment Archive",
        ]:
            _goto_page(browser_page, name)
            _assert_no_errors(browser_page, name)
