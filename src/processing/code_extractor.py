import re
from typing import List, Optional, Tuple

def extract_code_from_text(text: str) -> str:
    """
    Extract code blocks from LLM generated text.
    
    This function handles various code block formats including markdown-style
    triple backticks with language identifiers, and attempts to handle cases where
    the model might output explanations or multiple code blocks.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        String containing only the extracted code
    """
    # Pattern 1: Code blocks with triple backticks, with or without language identifier
    pattern1 = r'```(?:python|(?:py))?(.+?)```'
    # Pattern 2: Code blocks with ```python format
    pattern2 = r'```python\s*(.+?)```'
    # Pattern 3: Backup if the model doesn't properly close code blocks 
    pattern3 = r'```(?:python|(?:py))?(.*?)(?:```|$)'
    
    # Try the most restrictive pattern first, then fall back to more lenient ones
    matches = re.findall(pattern2, text, re.DOTALL)
    if not matches:
        matches = re.findall(pattern1, text, re.DOTALL)
    if not matches:
        matches = re.findall(pattern3, text, re.DOTALL)
    
    if matches:
        # If we have multiple code blocks, join them with newlines
        return "\n\n".join(match.strip() for match in matches)
    
    # Fallback: If no code blocks are found, try to extract what looks like Python code
    # We look for common Python patterns like function or class definitions, imports, etc.
    lines = text.split('\n')
    code_lines = []
    in_code_section = False
    
    for line in lines:
        stripped = line.strip()
        # Heuristics to detect Python code
        if (stripped.startswith('def ') or 
            stripped.startswith('class ') or 
            stripped.startswith('import ') or 
            stripped.startswith('from ') or 
            stripped.endswith(':') or
            any(op in stripped for op in ['=', '+=', '-=', '*=', '/=', '==', '>=', '<=', '!='])):
            in_code_section = True
            code_lines.append(line)
        # Continue adding lines if we're in a code section and the line is indented or not empty
        elif in_code_section and (line.startswith(' ') or line.startswith('\t') or stripped):
            code_lines.append(line)
        # Empty lines within code sections are kept
        elif in_code_section and not stripped:
            code_lines.append(line)
        # Reset when encountering likely non-code text after a code section
        elif in_code_section and len(stripped) > 0 and not any(char in stripped for char in '(){}[]+=*/&|<>!%^'):
            in_code_section = False
    
    # If we found code lines using the heuristic approach, join them
    if code_lines:
        return '\n'.join(code_lines)
    
    # Last resort: if all else fails, return the original text
    return text

def clean_extracted_code(code: str) -> str:
    """
    Clean up extracted code by removing non-code elements.
    
    Args:
        code: Extracted code string
        
    Returns:
        Cleaned code string
    """
    # Remove lines that are likely explanations
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that are likely explanations or comments that start with natural language
        if (line.startswith('This ') or 
            line.startswith('Here ') or
            line.startswith('The ') or
            line.startswith('In this ')):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def is_valid_python(code: str) -> bool:
    """
    Check if a string is valid Python code.
    
    Args:
        code: String of Python code to check
        
    Returns:
        True if the code is valid Python, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def extract_and_validate_code(text: str) -> Tuple[str, bool]:
    """
    Extract code from text and validate it as valid Python.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Tuple of (extracted_code, is_valid)
    """
    extracted = extract_code_from_text(text)
    cleaned = clean_extracted_code(extracted)
    is_valid = is_valid_python(cleaned)
    return cleaned, is_valid
