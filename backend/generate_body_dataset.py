import os
import requests
from tqdm import tqdm

# Body types
categories = {
    "pear": "pear body type person full body",
    "rectangle": "rectangle body shape person full body",
    "hourglass": "hourglass body type person full body",
    "inverted_triangle": "inverted triangle body type person full body"
}

BASE_DIR = "dataset_body"
os.makedirs(BASE_DIR, exist_ok=True)

def download_images(query, folder, num_images=30):
    os.makedirs(folder, exist_ok=True)
    saved = 0
    failed = 0

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FashionPagluDatasetBot/1.0)"
    }

    for i in tqdm(range(num_images)):
        try:
            url = f"https://source.unsplash.com/600x800/?{query},{i}"
            response = requests.get(url, headers=headers, timeout=12, allow_redirects=True)

            if response.status_code != 200:
                failed += 1
                continue

            content_type = response.headers.get("Content-Type", "").lower()
            if "image" not in content_type and not response.content.startswith(b"\xff\xd8"):
                failed += 1
                continue

            with open(os.path.join(folder, f"{i}.jpg"), "wb") as f:
                f.write(response.content)
            saved += 1

        except Exception as e:
            failed += 1
            # Keep script running even when network/proxy blocks some requests.
            print(f"Error for {query} #{i}: {e}")

    return saved, failed

# Generate dataset
total_saved = 0
total_failed = 0
for category, query in categories.items():
    print(f"\nDownloading {category} images...")
    folder_path = os.path.join(BASE_DIR, category)
    saved, failed = download_images(query, folder_path, num_images=40)
    total_saved += saved
    total_failed += failed
    print(f"{category}: saved={saved}, failed={failed}")

print("\n✅ Dataset generation finished!")
print(f"Total images saved: {total_saved}")
print(f"Total failed downloads: {total_failed}")
if total_saved == 0:
    print("⚠️ No images were saved. Check network/proxy access to source.unsplash.com")