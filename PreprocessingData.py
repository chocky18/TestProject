import requests
from bs4 import BeautifulSoup
import json

def extract_product_info(url):
    try:
        # Step 1: Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()

        # Step 2: Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize the data dictionary
        product_info = {
            "product_title": "",
            "product_highlights": [],
            "detailed_description": [],
            "Goes_Well_With": []
        }

        print(f"\n🔍 Extracted content from: {url}\n")

        # 1. Extract product title
        title = soup.find('h1', class_='product__title heading-size--page-title')
        if title:
            product_info["product_title"] = title.get_text(strip=True)
        else:
            print("⚠️ No product title found.")

        print("\n" + "-" * 60 + "\n")

        # 2. Extract product highlights (using class 'metafield-rich_text_field' for these)
        rich_texts = soup.find_all('div', class_='metafield-rich_text_field')
        if rich_texts:
            print("📝 Product Highlights:")
            for div in rich_texts:
                product_info["product_highlights"].append(div.get_text(strip=True))
        else:
            print("⚠️ No product highlights found.")

        print("\n" + "-" * 60 + "\n")

        # 3. Extract detailed description (these are typically in 'span' or similar tags with specific classes)
        detailed_desc = soup.find_all('span', class_='metafield-multi_line_text_field')
        if detailed_desc:
            print("📄 Detailed Description:")
            for span in detailed_desc:
                product_info["detailed_description"].append(span.get_text(strip=True))
        else:
            print("⚠️ No detailed description found.")

        print("\n" + "-" * 60 + "\n")

        # 4. Extract featured ingredients or tags (using class 'text-animation--underline-thin text-weight--bold' for example)
        featured_ingredients = soup.find_all('span', class_='text-animation--underline-thin text-weight--bold')
        if featured_ingredients:
            print("🎯 Featured Ingredients or Tags:")
            for span in featured_ingredients:
                product_info["Goes_Well_With"].append(span.get_text(strip=True))
        else:
            print("⚠️ No featured ingredients or tags found.")

        # Output the structured data as JSON format
        return product_info

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching the URL: {e}")
        return None

# List of sample product URLs
# List of product URLs to scrape
product_urls = [
    'https://beminimalist.co/collections/skin-body-1/products/salicylic-lha-2-cleanser',
    'https://beminimalist.co/collections/skin-body-1/products/multi-vitamin-spf-50',
    'https://beminimalist.co/collections/skin-body-1/products/salicylic-acid-2',
    'https://beminimalist.co/collections/skin-body-1/products/vitamin-c-ethyl-ascorbic-acid-10-acetyl-glucosamine-1',
    'https://beminimalist.co/collections/skin-body-1/products/vitamin-b5-10-moisturizer',
    'https://beminimalist.co/collections/skin-body-1/products/niacinamide-10-with-matmarine',
    'https://beminimalist.co/collections/skin-body-1/products/alpha-arbutin-2',
    'https://beminimalist.co/collections/skin-body-1/products/niacinamide-5-hyaluronic-acid-1',
    'https://beminimalist.co/collections/skin-body-1/products/pha-3-biotic-toner',
    'https://beminimalist.co/collections/skin-body-1/products/marula-05-moisturizer',
    'https://beminimalist.co/collections/skin-body-1/products/vitamin-c-e-ferulic-16',
    'https://beminimalist.co/collections/skin-body-1/products/2-hyaluronic-acid',
    'https://beminimalist.co/collections/skin-body-1/products/retinol-0-3-q10',
    'https://beminimalist.co/collections/skin-body-1/products/aha-25-pha-5-bha-2',
    'https://beminimalist.co/collections/skin-body-1/products/spf-60-silymarin',
    'https://beminimalist.co/collections/skin-body-1/products/glycolic-acid-08-exfoliating-liquid',
    'https://beminimalist.co/collections/skin-body-1/products/oat-extract-06-gentle-cleanser',
    'https://beminimalist.co/collections/skin-body-1/products/tranexamic-3-hpa',
    'https://beminimalist.co/collections/skin-body-1/products/spf-50-sunscreen-stick',
    'https://beminimalist.co/collections/skin-body-1/products/aquaporin-booster-05-cleanser',
    'https://beminimalist.co/collections/skin-body-1/products/l-ascorbic-acid-08-lip-treatment-balm',
    'https://beminimalist.co/collections/skin-body-1/products/niacinamide-05-body-lotion',
    'https://beminimalist.co/collections/skin-body-1/products/alpha-lipoic-glycolic-07-cleanser',
    'https://beminimalist.co/collections/skin-body-1/products/anti-acne-kit',
    'https://beminimalist.co/collections/skin-body-1/products/retinoid-2',
    'https://beminimalist.co/collections/skin-body-1/products/aha-bha-10',
    'https://beminimalist.co/collections/skin-body-1/products/nonapeptide-aha-06-underarm-roll-on',
    'https://beminimalist.co/collections/skin-body-1/products/vitamin-k-retinal-01-eye-cream',
    'https://beminimalist.co/collections/skin-body-1/products/lip-balm-spf-30',
    'https://beminimalist.co/collections/skin-body-1/products/multi-peptide-serum-7-matrixyl-3000-3-bio-placenta',
    'https://beminimalist.co/collections/skin-body-1/products/vitamin-b5-10-moisturizer-30g',
    'https://beminimalist.co/collections/skin-body-1/products/salicylic-acid-lha-02-body-wash',
    'https://beminimalist.co/collections/skin-body-1/products/oily-skincare-kit',
    'https://beminimalist.co/collections/skin-body-1/products/glycolic-tranexamic-11-body-exfoliator',
    'https://beminimalist.co/collections/skin-body-1/products/light-fluid-spf-50-sunscreen',
    'https://beminimalist.co/collections/skin-body-1/products/retinol-0-6',
    'https://beminimalist.co/collections/skin-body-1/products/ceramides-0-3-madecassoside',
    'https://beminimalist.co/collections/skin-body-1/products/spf-30-body-lotion',
    'https://beminimalist.co/collections/skin-body-1/products/squalane-100',
    'https://beminimalist.co/collections/skin-body-1/products/dry-skincare-kit',
    'https://beminimalist.co/collections/skin-body-1/products/anti-pigmentation-kit',
    'https://beminimalist.co/collections/skin-body-1/products/hocl-skin-relief-spray-150-ppm',
    'https://beminimalist.co/collections/skin-body-1/products/anti-aging-kit',
    'https://beminimalist.co/collections/skin-body-1/products/the-daily-radiance-ritual-3x-kit-50ml',
    'https://beminimalist.co/collections/skin-body-1/products/retinal-0-1-face-serum',
    'https://beminimalist.co/collections/skin-body-1/products/body-care-kit',
    'https://beminimalist.co/collections/skin-body-1/products/glow-and-protection-kit',
    'https://beminimalist.co/collections/skin-body-1/products/sun-protection-kit',
    'https://beminimalist.co/collections/skin-body-1/products/brightening-spf-skincare-gift-set',
    'https://beminimalist.co/collections/skin-body-1/products/retinal-0-2-liposomal-cream',
    'https://beminimalist.co/collections/skin-body-1/products/hydrating-repairing-skincare-gift-set'
]

# Dictionary to hold all product data
all_product_info = []

# Loop through each URL and scrape data
for url in product_urls:
    print(f"Scraping data for: {url}")
    product_data = extract_product_info(url)
    if product_data:
        all_product_info.append(product_data)

# Save the extracted data to a local JSON file
with open('product_data.json', 'w') as json_file:
    json.dump(all_product_info, json_file, indent=4)

print("\n✅ Data saved to product_data.json")


