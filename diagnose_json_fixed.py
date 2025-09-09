import sys
import json

def find_error_position(file_path, pos=4281):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"File length: {len(content)}")
    print(f"Error position: {pos}")
    
    # Find line and column more carefully
    prefix = content[:pos]
    lines = prefix.split('\n')
    line_num = len(lines)
    
    # Calculate column: length of last line + 1 for newline
    if len(lines) > 1:
        col = len(lines[-1]) + 1
    else:
        col = pos
    
    print(f"Approximate line: {line_num}, column: {col}")
    
    # Show context lines
    all_lines = content.split('\n')
    start_line = max(0, line_num - 3)
    end_line = min(len(all_lines), line_num + 2)
    
    print(f"\nContext around position (lines {start_line}-{end_line}):")
    for i in range(start_line, end_line + 1):
        if i < len(all_lines):
            line = all_lines[i]
            print(f"{i+1}: {repr(line)}")  # repr to show quotes/escapes
    
    # Snippet around position
    snippet_start = max(0, pos - 50)
    snippet_end = min(len(content), pos + 50)
    snippet = content[snippet_start:snippet_end]
    print(f"\nSnippet around pos {pos} (chars {snippet_start}-{snippet_end}): {repr(snippet)}")
    
    # Attempt partial parse to find last valid position
    try:
        bracket_count = 0
        quote_count = 0
        escape_next = False
        last_valid = 0
        
        for i, c in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            
            if c == '\\':
                escape_next = True
                continue
            
            if c == '"' and not escape_next:
                quote_count += 1
            
            if c == '{' and quote_count % 2 == 0:
                bracket_count += 1
            elif c == '}' and quote_count % 2 == 0:
                bracket_count -= 1
            
            if bracket_count == 0 and i > 0:
                last_valid = i + 1
        
        partial = content[:last_valid]
        data = json.loads(partial)
        cells = data.get('cells', [])
        print(f"\n✅ Partial parse up to {last_valid} chars successful.")
        print(f"📊 Cells found: {len(cells)}")
        if cells:
            print(f"🔍 Last cell type: {cells[-1].get('cell_type', 'unknown')}")
            print(f"📝 Last cell source preview: {cells[-1].get('source', [''])[-1][:100]}...")
    except json.JSONDecodeError as e:
        print(f"\n❌ Partial parse failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_error_position(sys.argv[1])
    else:
        print("Usage: python diagnose_json_fixed.py <file>")
