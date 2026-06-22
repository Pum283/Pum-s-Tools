import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
from pypdf import PdfReader
import docx

LANGUAGES = {
    "Tự động phát hiện": "auto",
    "Tiếng Việt": "vi",
    "Tiếng Anh": "en",
    "Tiếng Nhật": "ja",
    "Tiếng Hàn": "ko",
    "Tiếng Trung (Giản thể)": "zh-CN",
    "Tiếng Trung (Phồn thể)": "zh-TW",
    "Tiếng Pháp": "fr",
    "Tiếng Đức": "de",
    "Tiếng Tây Ban Nha": "es",
    "Tiếng Nga": "ru"
}

STYLES = {
    "Mặc định": "Dịch chính xác, tự nhiên.",
    "Trang trọng, lịch sự": "Dịch theo văn phong trang trọng, lịch sự, sử dụng từ ngữ kính trọng (ví dụ: dùng 'quý khách', 'chúng tôi', xưng hô lịch sự, từ ngữ trang nhã).",
    "Thân mật, gần gũi": "Dịch theo văn phong thân mật, gần gũi, tự nhiên như giao tiếp hàng ngày (ví dụ: dùng 'bạn', 'mình', 'tớ', câu văn tự nhiên, thoải mái).",
    "Học thuật, chuyên ngành": "Dịch theo phong cách học thuật, khoa học chuyên ngành, sử dụng chính xác các thuật ngữ chuyên môn.",
    "Văn học, nghệ thuật": "Dịch theo phong cách văn thơ bay bổng, uyển chuyển, giàu tính nhạc và hình ảnh nghệ thuật."
}

# Cache for loaded local models
_LOCAL_MODEL_CACHE = {}

def get_local_model_and_tokenizer(model_name: str):
    """Load and cache local translation model from Hugging Face."""
    global _LOCAL_MODEL_CACHE
    if model_name in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_name]
        
    print(f"📦 Loading local translation model: {model_name}...")
    
    if "m2m100" in model_name:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    else:
        try:
            if "opus-mt" in model_name or "helsinki" in model_name.lower():
                from transformers import MarianTokenizer
                tokenizer = MarianTokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️ Cảnh báo: AutoTokenizer thất bại, chuyển sang MarianTokenizer cho {model_name}: {e}")
            try:
                from transformers import MarianTokenizer
                tokenizer = MarianTokenizer.from_pretrained(model_name)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    _LOCAL_MODEL_CACHE[model_name] = (model, tokenizer, device)
    return model, tokenizer, device

def apply_vietnamese_style(text: str, style: str) -> str:
    """Adjust pronouns and word choices in Vietnamese text to match the requested style."""
    if not text or style == "Mặc định":
        return text
    
    if style == "Trang trọng, lịch sự":
        replacements = {
            " tôi ": " chúng tôi ",
            "Tôi ": "Chúng tôi ",
            " tớ ": " chúng tôi ",
            "Tớ ": "Chúng tôi ",
            " mình ": " chúng tôi ",
            "Mình ": "Chúng tôi ",
            " cậu ": " quý khách ",
            "Cậu ": "Quý khách ",
            " mày ": " quý khách ",
            "Mày ": "Quý khách ",
            " bạn ": " quý khách ",
            "Bạn ": "Quý khách ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
    elif style == "Thân mật, gần gũi":
        replacements = {
            " tôi ": " mình ",
            "Tôi ": "Mình ",
            " chúng tôi ": " chúng mình ",
            "Chúng tôi ": "Chúng mình ",
            " quý khách ": " bạn ",
            "Quý khách ": "Bạn ",
            " ngài ": " bạn ",
            "Ngài ": "Bạn ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
    elif style == "Văn học, nghệ thuật":
        replacements = {
            " nhanh chóng ": " mau chóng ",
            " rất ": " vô cùng ",
            " đẹp ": " thơ mộng ",
            " nói ": " bộc bạch ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
            
    return text

def translate_text(text: str, target_lang: str, source_lang: str = "auto", style: str = "Mặc định") -> str:
    """Translate plain text locally in small chunks of 500 characters."""
    if not text or not text.strip():
        return ""
    
    src = LANGUAGES.get(source_lang, "auto")
    tgt = LANGUAGES.get(target_lang, "vi")
    
    if src == "auto":
        src = "en" # Fallback auto to English source
        
    # Determine model name
    if src == "en" and tgt == "vi":
        model_name = "Helsinki-NLP/opus-mt-en-vi"
    elif src == "vi" and tgt == "en":
        model_name = "Helsinki-NLP/opus-mt-vi-en"
    elif src == "zh-CN" and tgt == "vi":
        model_name = "Helsinki-NLP/opus-mt-zh-vi"
    elif src == "fr" and tgt == "vi":
        model_name = "Helsinki-NLP/opus-mt-fr-vi"
    elif src == "de" and tgt == "vi":
        model_name = "Helsinki-NLP/opus-mt-de-vi"
    else:
        model_name = "facebook/m2m100_418M"
        
    model, tokenizer, device = get_local_model_and_tokenizer(model_name)
    
    # Split text into sentences or small chunks (approx 500 characters) to avoid generation cutoff
    chunk_size = 500
    paragraphs = text.split("\n")
    translated_paragraphs = []
    
    for para in paragraphs:
        if not para.strip():
            translated_paragraphs.append("")
            continue
            
        # Split paragraph into chunks of 500 chars
        chunks = [para[i:i+chunk_size] for i in range(0, len(para), chunk_size)]
        translated_chunks = []
        
        for chunk in chunks:
            if not chunk.strip():
                translated_chunks.append(chunk)
                continue
                
            try:
                if "m2m100" in model_name:
                    src_m2m = src.split("-")[0]
                    tgt_m2m = tgt.split("-")[0]
                    tokenizer.src_lang = src_m2m
                    inputs = tokenizer(chunk, return_tensors="pt").to(device)
                    with torch.no_grad():
                        generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_m2m), max_length=512)
                    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                else:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_length=512)
                    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Apply Vietnamese style styling if target is Vietnamese
                if tgt == "vi":
                    translated = apply_vietnamese_style(translated, style)
                    
                translated_chunks.append(translated)
            except Exception as e:
                print(f"Error local translating chunk: {e}")
                translated_chunks.append(chunk)
                
        translated_paragraphs.append("".join(translated_chunks))
        
    return "\n".join(translated_paragraphs)

def translate_docx(file_path: str, target_lang: str, source_lang: str = "auto", style: str = "Mặc định") -> str:
    """Translate a Word (.docx) document locally paragraph by paragraph."""
    doc = docx.Document(file_path)
    
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            try:
                p.text = translate_text(p.text, target_lang, source_lang, style)
            except Exception as e:
                print(f"Error translating docx paragraph {i}: {e}")
                
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if p.text.strip():
                        try:
                            p.text = translate_text(p.text, target_lang, source_lang, style)
                        except Exception as e:
                            print(f"Error translating table paragraph: {e}")

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    out_name = f"translated_{base_name}"
    out_path = os.path.join(dir_name, out_name)
    doc.save(out_path)
    return out_path

def translate_pdf(file_path: str, target_lang: str, source_lang: str = "auto", style: str = "Mặc định") -> tuple[str, str]:
    """Extract text from PDF, translate locally, and save it as a new .docx file."""
    reader = PdfReader(file_path)
    text_content = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
            
    full_text = "\n\n".join(text_content)
    translated_text = translate_text(full_text, target_lang, source_lang, style)
    
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    out_docx_name = f"translated_{os.path.splitext(base_name)[0]}.docx"
    out_docx_path = os.path.join(dir_name, out_docx_name)
    
    doc = docx.Document()
    for paragraph in translated_text.split("\n\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph)
    doc.save(out_docx_path)
        
    return out_docx_path, translated_text

def translate_document(file_path: str, target_lang: str, source_lang: str = "auto", style: str = "Mặc định") -> tuple[str, str]:
    """Main function to parse extension and translate."""
    if not file_path or not os.path.exists(file_path):
        return None, "❌ File không tồn tại."
        
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    
    try:
        if ext == ".docx":
            out_path = translate_docx(file_path, target_lang, source_lang, style)
            doc = docx.Document(out_path)
            preview_text = "\n".join([p.text for p in doc.paragraphs[:10]])
            return out_path, preview_text
            
        elif ext == ".pdf":
            out_path, translated_text = translate_pdf(file_path, target_lang, source_lang, style)
            return out_path, translated_text[:1000]
            
        elif ext in [".txt", ".md", ".json"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            translated_text = translate_text(content, target_lang, source_lang, style)
            
            out_name = f"translated_{base_name}"
            out_path = os.path.join(dir_name, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
                
            return out_path, translated_text[:1000]
            
        else:
            return None, f"❌ Định dạng file {ext} không được hỗ trợ."
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Lỗi trong quá trình dịch: {str(e)}"
