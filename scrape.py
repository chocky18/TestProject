import requests
from bs4 import BeautifulSoup

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
            "Goes Well With": []
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
                product_info["Goes Well With"].append(span.get_text(strip=True))
        else:
            print("⚠️ No Goes Well With found.")

        # Output the structured data as JSON format
        print("\n🔧 Structured Data (JSON Format):")
        print(product_info)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching the URL: {e}")

# 🟡 Replace this with your actual product page URL
url = "https://beminimalist.co/collections/skin-body-1/products/multi-vitamin-spf-50?_pos=2&_fid=46c6b3422&_ss=c"
extract_product_info(url)
