"""
SHL Catalog Scraper
Scrapes https://www.shl.com/solutions/products/product-catalog/ (Individual Test Solutions only)
and produces catalog.json.

Usage:
    pip install requests beautifulsoup4 lxml
    python scraper.py

Note: The SHL catalog uses JavaScript rendering. This script attempts to paginate
through the catalog API. If it fails due to JS rendering, use Playwright:
    pip install playwright && playwright install chromium
    python scraper_playwright.py
"""

import json
import time
import requests
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# The SHL catalog paginates with ?start=N&type=1 for Individual Test Solutions
# type=1 filters to Individual Test Solutions (type=2 is Pre-packaged Job Solutions)

def scrape_catalog():
    """
    Attempt to scrape the SHL catalog. The catalog is paginated.
    Each page has 12 items. We iterate until we get no more results.
    """
    all_items = []
    start = 0
    page_size = 12

    session = requests.Session()
    session.headers.update(HEADERS)

    while True:
        url = f"{CATALOG_URL}?start={start}&type=1&action_doFilteringForm=Search"
        print(f"Fetching: {url}")
        
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            break

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        # Find product cards - SHL uses custom-select__item or product-catalogue__item
        cards = soup.select(".custom-select__item, .product-catalogue__item, [data-course-id]")
        
        if not cards:
            # Try alternative selectors
            cards = soup.select("table tbody tr, .catalogue-item")

        if not cards:
            print(f"No items found at start={start}, stopping.")
            break

        for card in cards:
            item = parse_card(card, soup)
            if item:
                all_items.append(item)

        print(f"  Found {len(cards)} items (total so far: {len(all_items)})")

        if len(cards) < page_size:
            break  # Last page

        start += page_size
        time.sleep(1)  # Polite delay

    return all_items


def parse_card(card, soup) -> dict | None:
    """Extract assessment data from a catalog card element."""
    try:
        # Name and URL
        link = card.find("a")
        if not link:
            return None
        
        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = urljoin(BASE_URL, href) if href else ""

        if not name or not url:
            return None

        # Test type indicators (SHL uses colored icons A/B/K/P/S)
        test_types = []
        type_icons = card.select(".catalogue__type-icon, [class*='type-']")
        for icon in type_icons:
            t = icon.get_text(strip=True).upper()
            if t in ("A", "B", "K", "P", "S"):
                test_types.append(t)

        # Remote testing checkbox
        remote = bool(card.select(".catalogue__remote, [class*='remote']"))

        return {
            "name": name,
            "url": url,
            "test_type": test_types[0] if test_types else "A",
            "remote_testing": remote,
            "description": "",
            "job_levels": [],
            "competencies": [],
            "languages": ["English"],
            "adaptive": False,
            "duration_minutes": None,
        }
    except Exception as e:
        print(f"  Error parsing card: {e}")
        return None


def enrich_item(item: dict, session: requests.Session) -> dict:
    """Visit each product page to extract richer data."""
    try:
        resp = session.get(item["url"], timeout=15)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        # Description
        desc_el = soup.select_one(".product-catalogue__description, .product__description, [class*='description']")
        if desc_el:
            item["description"] = desc_el.get_text(strip=True)

        # Duration
        dur_el = soup.find(string=lambda t: t and "minute" in t.lower())
        if dur_el:
            import re
            m = re.search(r"(\d+)\s*minute", dur_el, re.IGNORECASE)
            if m:
                item["duration_minutes"] = int(m.group(1))

    except Exception as e:
        print(f"  Failed to enrich {item['name']}: {e}")

    return item


if __name__ == "__main__":
    print("Starting SHL catalog scrape...")
    items = scrape_catalog()
    
    if not items:
        print("No items scraped. The catalog may require JavaScript rendering.")
        print("Try using Playwright: pip install playwright && playwright install chromium")
        print("Falling back to bundled catalog.json")
    else:
        print(f"\nScraped {len(items)} assessments.")
        
        session = requests.Session()
        session.headers.update(HEADERS)
        
        print("Enriching items with detail pages...")
        enriched = []
        for i, item in enumerate(items):
            print(f"  {i+1}/{len(items)}: {item['name']}")
            enriched.append(enrich_item(item, session))
            time.sleep(0.5)
        
        with open("catalog.json", "w") as f:
            json.dump(enriched, f, indent=2)
        
        print(f"\nSaved {len(enriched)} items to catalog.json")
