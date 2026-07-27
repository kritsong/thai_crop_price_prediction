import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_DIR = PROJECT_ROOT / 'paper'
PAPER_PATH = PAPER_DIR / 'final_paper.md'


def resolve_image_path(img_path):
    candidate = Path(img_path)
    if candidate.is_absolute():
        return candidate

    normalized = img_path.replace('\\', '/')
    if normalized.startswith('paper/'):
        return PROJECT_ROOT / candidate
    if normalized.startswith('images/'):
        return PAPER_DIR / candidate
    return PAPER_DIR / candidate

def verify():
    with open(PAPER_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    forbidden = ['weather', 'rain', 'temperature', 'climate', 'precipitation']
    allowed_substrings = ['train', 'training', 'constrained', 'strain', 'unbounded', 'grain', 'grains', 'brain', 'brains']
    
    # We want to find occurrences of the forbidden words.
    # But we want to exclude cases where the match is part of an allowed substring.
    # To do this robustly, we can find all matches of the forbidden word, and for each, 
    # check if the surrounding context matches one of the allowed substrings.
    
    failures = []
    for word in forbidden:
        for match in re.finditer(word, content, re.IGNORECASE):
            start_idx = match.start()
            end_idx = match.end()
            
            # Check context around the match
            is_allowed = False
            for allowed in allowed_substrings:
                # Find occurrences of the allowed word in the content
                for allowed_match in re.finditer(allowed, content, re.IGNORECASE):
                    if allowed_match.start() <= start_idx and end_idx <= allowed_match.end():
                        is_allowed = True
                        break
                if is_allowed:
                    break
            
            if not is_allowed:
                line_no = content[:start_idx].count('\n') + 1
                snippet = content[max(0, start_idx-40):min(len(content), end_idx+40)].replace('\n', ' ')
                failures.append((word, line_no, snippet))
                
    if failures:
        print(f"FAILED: Found forbidden words in final_paper.md:")
        for word, line_no, snippet in failures:
            print(f"  - '{word}' at line {line_no}: ... {snippet} ...")
        sys.exit(1)
    else:
        print("PASSED: No forbidden words (weather, rain, temperature, climate, precipitation) found in final_paper.md (excluding allowed substrings).")
        
    # 2. Check embedded images
    print("\nChecking embedded images in final_paper.md...")
    image_pattern = re.compile(r'!\[.*?\]\((file:///)?(.*?)\)')
    matches = image_pattern.findall(content)
    
    if len(matches) != 12:
        print(f"FAILED: Expected exactly 12 embedded images in final_paper.md, but found {len(matches)}.")
        sys.exit(1)
        
    image_failures = []
    for index, (protocol, img_path) in enumerate(matches, 1):
        resolved_path = resolve_image_path(img_path)

        if not resolved_path.exists():
            image_failures.append(f"Image {index}: path '{resolved_path}' does not exist (from markdown link '{img_path}').")
        else:
            print(f"Image {index} exists: {resolved_path}")
            
    if image_failures:
        print("FAILED: Some embedded images are missing:")
        for f in image_failures:
            print(" ", f)
        sys.exit(1)
    else:
        print("PASSED: All 12 embedded images exist at their exact paths.")
        sys.exit(0)

if __name__ == '__main__':
    verify()
