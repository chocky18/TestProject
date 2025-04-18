import requests
from bs4 import BeautifulSoup

BASE_URL = "https://beminimalist.co"
COLLECTION_URLS = [
    "https://beminimalist.co/collections/skin-body-1?page=1",
    "https://beminimalist.co/collections/skin-body-1?page=2"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

product_links = []

for url in COLLECTION_URLS:
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    product_items = soup.find_all("div", class_="product-item")

    for item in product_items:
        a_tag = item.find("a", class_="product-item__image")
        if a_tag and a_tag.get("href"):
            full_url = BASE_URL + a_tag["href"].split("?")[0]
            if full_url not in product_links:
                product_links.append(full_url)

print(f"✅ Found {len(product_links)} unique product URLs:\n")
for link in product_links:
    print(link)
