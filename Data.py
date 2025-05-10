import os
import hashlib
import logging
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fetch_stack_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face authentication
HF_TOKEN = "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"
try:
    login(token=HF_TOKEN)
    logger.info("Successfully authenticated with Hugging Face")
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {e}")
    exit(1)

# Define desired languages
DESIRED_LANGUAGES = [
    "Swift", "Python", "Lua", "C", "C++", "Objective-C", "C#",
    "Ruby", "JavaScript", "TypeScript", "Luau"
]

# Output directory for code files
OUTPUT_BASE_DIR = "code_by_language"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Create subdirectories for each language
for lang in DESIRED_LANGUAGES:
    lang_dir = os.path.join(OUTPUT_BASE_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    logger.info(f"Created directory: {lang_dir}")

# Limit number of files per language
MAX_FILES_PER_LANGUAGE = 10000

# Track number of files saved per language
files_saved = {lang: 0 for lang in DESIRED_LANGUAGES}

def get_safe_filename(path, content, lang):
    """Generate a unique, safe filename based on the file path and content."""
    try:
        unique_str = f"{path}_{content[:100]}"  # Use first 100 chars of content
        hash_object = hashlib.md5(unique_str.encode())
        base_name = hash_object.hexdigest()
        # Use appropriate extension based on language
        ext_map = {
            "Swift": ".swift", "Python": ".py", "Lua": ".lua", "C": ".c",
            "C++": ".cpp", "Objective-C": ".m", "C#": ".cs", "Ruby": ".rb",
            "JavaScript": ".js", "TypeScript": ".ts", "Luau": ".luau"
        }
        ext = ext_map.get(lang, os.path.splitext(path)[1] if "." in path else ".txt")
        return f"{base_name}{ext}"
    except Exception as e:
        logger.error(f"Error generating safe filename for {path}: {e}")
        return f"{hashlib.md5(content.encode()).hexdigest()}.txt"

# Verify dataset availability
try:
    logger.info("Checking dataset availability...")
    # Attempt to download a small file from the dataset to verify access
    hf_hub_download(
        repo_id="bigcode/the-stack-v2",
        filename="README.md",
        repo_type="dataset",
        token=HF_TOKEN
    )
    logger.info("Dataset access verified: Successfully accessed bigcode/the-stack-v2")
except Exception as e:
    logger.error(f"Cannot access dataset: {e}")
    exit(1)

# Check network connectivity
try:
    response = requests.get("https://huggingface.co", timeout=5)
    if response.status_code != 200:
        logger.error("Network issue: Cannot reach Hugging Face servers")
        exit(1)
    logger.info("Network check passed: Hugging Face servers reachable")
except requests.RequestException as e:
    logger.error(f"Network check failed: {e}")
    exit(1)

# Load dataset in streaming mode
try:
    ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True, token=HF_TOKEN)
    logger.info("Successfully loaded dataset in streaming mode")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    exit(1)

# Iterate over dataset and filter by language
try:
    for file in tqdm(ds, desc="Processing files", unit="file"):
        # Skip if all languages have reached their file limit
        if all(files_saved[lang] >= MAX_FILES_PER_LANGUAGE for lang in DESIRED_LANGUAGES):
            logger.info("All languages reached file limit; stopping")
            break

        language = file.get("lang")
        content = file.get("content")
        path = file.get("path", "unknown")

        # Handle Luau fallback for Roblox-related Lua files
        if language == "Lua" and any(keyword in path.lower() for keyword in ["roblox", ".luau"]):
            language = "Luau"

        # Normalize language names to match DESIRED_LANGUAGES
        language = {
            "Cpp": "C++", "JavaScript": "JavaScript", "TypeScript": "TypeScript",
            "ObjectiveC": "Objective-C", "CSharp": "C#"
        }.get(language, language)

        if language in DESIRED_LANGUAGES and files_saved[language] < MAX_FILES_PER_LANGUAGE:
            if not content or not path or not content.strip():
                logger.warning(f"Skipping file: missing or empty content/path")
                continue

            # Generate a safe filename
            safe_filename = get_safe_filename(path, content, language)
            output_path = os.path.join(OUTPUT_BASE_DIR, language, safe_filename)

            # Check for duplicate content
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if os.path.exists(output_path):
                logger.warning(f"Skipping duplicate file: {safe_filename}")
                continue

            # Save file
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                files_saved[language] += 1
                logger.info(f"Saved {safe_filename} to {language}/ (Total: {files_saved[language]})")
            except (OSError, UnicodeEncodeError) as e:
                logger.warning(f"Failed to save {safe_filename} to {language}/: {e}")

        # Log progress periodically
        if sum(files_saved.values()) % 100 == 0:
            logger.info(f"Files saved per language: {files_saved}")

except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    exit(1)

# Final summary
logger.info("\nFinal count of files saved per language:")
for lang, count in files_saved.items():
    logger.info(f"{lang}: {count} files")
