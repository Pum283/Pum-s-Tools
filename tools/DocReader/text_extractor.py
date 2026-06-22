import os
import docx

def read_txt_file(file_path: str) -> str:
    """Read a text file with fallback encodings to support Vietnamese."""
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-8-sig', 'latin-1', 'cp1258']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Fallback to ignore errors
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_docx_file(file_path: str) -> str:
    """Read paragraphs and tables from a DOCX file."""
    doc = docx.Document(file_path)
    
    # Extract paragraphs
    paragraphs = [p.text for p in doc.paragraphs]
    
    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text and cell_text not in row_text:
                    row_text.append(cell_text)
            if row_text:
                paragraphs.append(" | ".join(row_text))
                
    return "\n".join(paragraphs)

def extract_text_from_file(file_path: str) -> str:
    """Detect file extension and extract text content."""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        return read_txt_file(file_path)
    elif ext == ".docx":
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Định dạng file {ext} không được hỗ trợ. Chỉ hỗ trợ .txt và .docx.")
