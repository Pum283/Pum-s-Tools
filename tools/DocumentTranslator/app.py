import gradio as gr
import os
from tools.DocumentTranslator.translator import translate_document, LANGUAGES, STYLES

def handle_translation(file, source_lang, target_lang, style):
    if file is None:
        return None, "⚠️ Vui lòng tải lên tài liệu!", ""
        
    source_code = LANGUAGES[source_lang]
    target_code = LANGUAGES[target_lang]
    
    status_msg = f"⏳ Đang tải mô hình & dịch tài liệu cục bộ (Offline)... Phong cách: {style}."
    yield None, status_msg, ""
    
    out_path, preview = translate_document(file.name, target_code, source_code, style)
    
    if out_path is None:
        yield None, preview, ""
    else:
        yield out_path, f"✅ Dịch thành công cục bộ trên máy! (Phong cách: {style})", preview

with gr.Blocks() as demo:
    gr.HTML("<div style='margin-bottom: 20px;'><h2>📄 Dịch tài liệu cục bộ (On-Device Neural Translation)</h2><p style='color: #64748b; margin-top: 5px;'>Chạy hoàn toàn cục bộ (offline) trên CPU/GPU của bạn, không cần API Key, không gửi dữ liệu ra internet. Hỗ trợ các định dạng file .txt, .md, .json, .pdf, .docx.</p></div>")
    
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                label="Tải lên tài liệu",
                file_types=[".txt", ".docx", ".pdf", ".md", ".json"]
            )
            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Tự động phát hiện",
                    label="Ngôn ngữ nguồn"
                )
                target_lang = gr.Dropdown(
                    choices=[k for k in LANGUAGES.keys() if k != "Tự động phát hiện"],
                    value="Tiếng Việt",
                    label="Ngôn ngữ đích"
                )
                style_select = gr.Dropdown(
                    choices=list(STYLES.keys()),
                    value="Mặc định",
                    label="Phong cách dịch"
                )
            btn_translate = gr.Button("🚀 Bắt đầu dịch", variant="primary")
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="Trạng thái",
                value="⏳ Chờ tải tài liệu...",
                interactive=False
            )
            file_output = gr.File(
                label="Tải về tài liệu đã dịch",
                interactive=False
            )
            preview_output = gr.Textbox(
                label="Xem trước nội dung (1000 ký tự đầu)",
                lines=10,
                interactive=False,
                placeholder="Nội dung dịch sẽ hiển thị xem trước ở đây..."
            )

    btn_translate.click(
        fn=handle_translation,
        inputs=[file_input, source_lang, target_lang, style_select],
        outputs=[file_output, status_output, preview_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7862)
