import sys
import json

def find_error_position(file_path, pos=4281):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File length: {len(content)}")
    print(f"Error position: {pos}")
    
    # Find line and column
    lines = content[:pos].split('\n')
    line_num = len(lines)
    col = pos - len(content[:pos].rsplit('\n', 1)[0] - 1 if '\n' in content[:pos] else pos
    
    print(f"Approximate line: {line_num}, column: {col}")
    print("\nContext around position (lines {}-{}):".format(line_num-2, line_num+2))
    
    for i, line in enumerate(lines[-3:], line_num-2):
        print(f"{i}: {repr(line)}")  # repr to show quotes/escapes
    
    # Try to find unterminated string
    snippet = content[max(0, pos-50):pos+50]
    print(f"\nSnippet around pos {pos}: {repr(snippet)}")
    
    # Attempt partial parse
    try:
        # Find last complete object before error
        bracket_count = 0
        last_valid = 0
        for i, c in enumerate(content):
            if c == '{': bracket_count += 1
            elif c == '}': bracket_count -= 1
            if bracket_count == 0 and i > 0:
                last_valid = i + 1
        partial = content[:last_valid]
        data = json.loads(partial)
        print(f"\nPartial parse up to {last_valid} chars successful. Cells found: {len(data.get('cells', []))}")
    except json.JSONDecodeError as e:
        print(f"Partial parse failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_error_position(sys.argv[1])
    else:
        print("Usage: python diagnose_json.py <file>")
