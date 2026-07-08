import re
from typing import List

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from a table cell string."""
    text = text.replace(',', '')
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    return [float(m) for m in matches]

def evaluate_program(program: str, table: List[List[str]]) -> float:
    """
    Evaluate a FinQA-style program.
    E.g. "subtract(108.50, 100), divide(#0, 100)"
    or "table_max(chênh lệch tỷ giá hối đoái, none)"
    """
    if not program:
        raise ValueError("Empty program")
        
    # FinQA format uses commas to separate steps, but arguments can be strings containing commas or parentheses.
    steps = []
    current_op = ""
    current_args = ""
    in_args = False
    paren_depth = 0
    i = 0
    while i < len(program):
        c = program[i]
        if not in_args:
            if c == '(':
                in_args = True
                paren_depth = 1
            elif c.isalnum() or c == '_':
                current_op += c
        else:
            if c == '(':
                paren_depth += 1
                current_args += c
            elif c == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    steps.append((current_op.strip(), current_args.strip()))
                    current_op = ""
                    current_args = ""
                    in_args = False
                    while i + 1 < len(program) and program[i+1] in [',', ' ']:
                        i += 1
                else:
                    current_args += c
            else:
                current_args += c
        i += 1
    results = []
    
    for op, args_str in steps:
        op = op.lower().strip()
        
        if op.startswith("table_"):
            # Table ops take 2 arguments, often strings with commas.
            # Usually format is "row_name, none" or "none, col_name"
            if args_str.lower().endswith(", none"):
                arg1 = args_str[:-6].strip()
                arg2 = "none"
            elif args_str.lower().startswith("none,"):
                arg1 = "none"
                arg2 = args_str[5:].strip()
            else:
                parts = args_str.rsplit(",", 1)
                arg1 = parts[0].strip() if len(parts) > 0 else ""
                arg2 = parts[1].strip() if len(parts) > 1 else ""
            args = [arg1, arg2]
        else:
            args = [arg.strip() for arg in args_str.split(",")]
        
        resolved_args = []
        for arg in args:
            if arg.startswith("#"):
                try:
                    idx = int(arg[1:])
                    resolved_args.append(results[idx])
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid reference {arg}")
            elif arg.lower() == "none":
                resolved_args.append(None)
            else:
                # Try float
                clean_arg = arg.replace("%", "").replace(",", "")
                try:
                    resolved_args.append(float(clean_arg))
                except ValueError:
                    resolved_args.append(arg)
                    
        # Evaluate operation
        try:
            if op == "add":
                res = resolved_args[0] + resolved_args[1]
            elif op == "subtract":
                res = resolved_args[0] - resolved_args[1]
            elif op == "multiply":
                res = resolved_args[0] * resolved_args[1]
            elif op == "divide":
                res = resolved_args[0] / resolved_args[1]
            elif op == "exp":
                res = resolved_args[0] ** resolved_args[1]
            elif op == "greater":
                res = resolved_args[0] if resolved_args[0] > resolved_args[1] else resolved_args[1]
            elif op == "abs":
                res = abs(resolved_args[0])
            elif op.startswith("table_"):
                row_name = resolved_args[0] if isinstance(resolved_args[0], str) else None
                col_name = resolved_args[1] if len(resolved_args) > 1 and isinstance(resolved_args[1], str) else None
                
                numbers = []
                if row_name and row_name.lower() != "none" and table:
                    row_idx = -1
                    for i, row in enumerate(table):
                        if row and row_name.lower() in row[0].lower():
                            row_idx = i
                            break
                    if row_idx != -1:
                        for cell in table[row_idx][1:]:
                            numbers.extend(extract_numbers(cell))
                elif col_name and col_name.lower() != "none" and table:
                    if table and table[0]:
                        col_idx = -1
                        for j, header in enumerate(table[0]):
                            if col_name.lower() in header.lower():
                                col_idx = j
                                break
                        if col_idx != -1:
                            for i in range(1, len(table)):
                                if col_idx < len(table[i]):
                                    numbers.extend(extract_numbers(table[i][col_idx]))
                                    
                if not numbers:
                    raise ValueError(f"No numbers found for table operation {op}")
                    
                if op == "table_max":
                    res = max(numbers)
                elif op == "table_min":
                    res = min(numbers)
                elif op == "table_sum":
                    res = sum(numbers)
                elif op == "table_average":
                    res = sum(numbers) / len(numbers)
                else:
                    raise ValueError(f"Unknown table op {op}")
            else:
                raise ValueError(f"Unknown op {op}")
        except Exception as e:
            raise ValueError(f"Error evaluating {op} with args {resolved_args}: {e}")
            
        results.append(res)
        
    if not results:
        raise ValueError("No valid steps parsed from program")
        
    return float(results[-1])

if __name__ == "__main__":
    # Test cases
    print(evaluate_program("subtract(108.50, 100), divide(#0, 100)", [])) # 0.085
    table = [
        ['', '2015', '2014'],
        ['chênh lệch tỷ giá', '$ -61.1 ( 61.1 )', '$ -16.6 ( 16.6 )']
    ]
    print(evaluate_program("table_max(chênh lệch tỷ giá, none)", table)) # -16.6
