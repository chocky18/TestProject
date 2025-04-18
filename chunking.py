import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your JSON file
with open('D:/Multi-Agents/skin_care_product_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

# Store all structured chunks
structured_chunks = []

for idx, product in enumerate(data):
    product_id = str(idx + 1)
    title = product.get('product_title', '')

    sections = {
        "Highlights": '\n'.join(product.get('product_highlights', [])) if isinstance(product.get('product_highlights'), list) else product.get('product_highlights', ''),
        "Description": product.get('detailed_description', ''),
        "Ingredients": product.get('ingredients', '')  # If present in your JSON
    }

    for section_name, section_text in sections.items():
        if section_text:  # Avoid empty fields
            # Convert lists to strings
            if isinstance(section_text, list):
                section_text = '\n'.join(str(item) for item in section_text)
            else:
                section_text = str(section_text)

            chunks = text_splitter.split_text(section_text)
            for chunk in chunks:
                structured_chunks.append({
                    "product_id": product_id,
                    "title": title,
                    "section": section_name,
                    "content": chunk
                })


# Optional: Save to file
with open("rag_structured_chunks.json", "w", encoding="utf-8") as f:
    json.dump(structured_chunks, f, ensure_ascii=False, indent=2)
