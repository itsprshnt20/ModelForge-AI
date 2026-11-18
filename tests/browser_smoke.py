"""Run the demo, threshold and export workflow using Chromium."""
import sys

from playwright.sync_api import expect, sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501").rstrip("/")


def metric(page, label):
    return page.get_by_test_id("stMetric").filter(has_text=label).get_by_test_id("stMetricValue")


def main():
    expect.set_options(timeout=45000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 1000})
        outside = []

        def local_only(route):
            if route.request.url.startswith(URL + "/"):
                route.continue_()
            else:
                outside.append(route.request.url)
                route.abort()

        context.route("**/*", local_only)
        page = context.new_page()
        page.goto(URL)
        expect(page.get_by_role("heading", name="ModelForge AI")).to_be_visible()
        expect(page.locator(".stApp")).to_have_css("background-color", "rgb(0, 0, 0)")
        page.get_by_role("button", name="Load demo").click()
        expect(page.get_by_role("button", name="Train models")).to_be_disabled()
        page.get_by_test_id("stSelectbox").filter(has_text="Positive class").get_by_role("combobox").click()
        page.get_by_role("option", name="Yes", exact=True).click()
        page.get_by_role("button", name="Train models").click()
        expect(page.get_by_role("heading", name="Model comparison")).to_be_visible()
        expect(page.get_by_text("Positive class: Yes", exact=False)).to_be_visible()
        expect(page.get_by_role("heading", name="Threshold explorer")).to_be_visible()
        print("PASS: synthetic demo, explicit positive class, all models trained, metrics rendered")

        slider = page.get_by_role("slider")
        track = page.locator('[data-baseweb="slider"]')
        track.click(position={"x": 1, "y": 16})
        expect(slider).to_have_attribute("aria-valuenow", "0")
        expect(metric(page, "False negatives (FN)")).to_have_text("0")
        expect(metric(page, "True negatives (TN)")).to_have_text("0")
        low_fp = int(metric(page, "False positives (FP)").inner_text())
        track.click(position={"x": track.bounding_box()["width"] - 2, "y": 16})
        expect(slider).to_have_attribute("aria-valuenow", "1")
        expect(metric(page, "False positives (FP)")).to_have_text("0")
        expect(metric(page, "True positives (TP)")).to_have_text("0")
        high_fn = int(metric(page, "False negatives (FN)").inner_text())
        assert low_fp > 0 and high_fn > 0
        expect(page.get_by_text("Running...", exact=True)).to_be_hidden()
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download model card").click()
        card = downloaded.value.path().read_text(encoding="utf-8")
        assert "- Model: Logistic regression" in card
        assert "- Threshold: 1.00" in card
        assert "FP 0" in card and f"FN {high_fn}" in card
        assert "SYN-0001" not in card
        print(f"PASS: threshold 0 to 1, FP {low_fp} to 0, FN 0 to {high_fn}, matching export")

        page.set_viewport_size({"width": 390, "height": 850})
        expect(page.get_by_role("button", name="Train models")).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        upload = page.locator('input[type="file"]')
        upload.set_input_files({"name": "example.csv", "mimeType": "text/csv",
                                "buffer": b"age,churn\n10,Yes\n20,No\n30,Yes\n40,No"})
        page.get_by_role("button", name="Load uploaded CSV").click()
        expect(page.get_by_text("User-uploaded CSV (example.csv)", exact=True)).to_be_visible()
        expect(page.get_by_text("Select the target column", exact=False)).to_be_visible()
        upload.set_input_files({
            "name": "bad.csv", "mimeType": "text/csv", "buffer": b'x,y\n1,"truncated',
        })
        page.get_by_role("button", name="Load uploaded CSV").click()
        expect(page.get_by_text("Malformed CSV", exact=False)).to_be_visible()
        assert not outside, outside
        print("PASS: mobile layout, valid and malformed CSV uploads, no external requests")
        browser.close()


if __name__ == "__main__":
    main()
