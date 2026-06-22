import os
import sys
import gradio as gr
from tools.DocReader.text_extractor import extract_text_from_file

# Heuristic offline language detection
def detect_language(text: str) -> str:
    if not text:
        return "vi"
    vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                   "ÀÁẢÃẠĂẰẮẰẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỲĐ")
    vi_count = sum(1 for c in text if c in vi_chars)
    vi_words = {"và", "là", "của", "được", "người", "có", "không", "trong", "một", "cho", "đến", "với", "những", "này", "theo", "đã", "để"}
    text_words = set(text.lower().split())
    vi_word_count = len(text_words.intersection(vi_words))
    en_words = {"the", "and", "of", "to", "in", "is", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "at"}
    en_word_count = len(text_words.intersection(en_words))
    if vi_count > 0 or vi_word_count > en_word_count:
        return "vi"
    else:
        return "en"

def get_synthesis_language(text, language_choice):
    if language_choice == "Tự động phát hiện":
        lang_code = detect_language(text)
        return lang_code, f"Tự động phát hiện ({'Tiếng Việt' if lang_code == 'vi' else 'Tiếng Anh'})"
    else:
        lang_code = "vi" if language_choice == "Tiếng Việt" else "en"
        return lang_code, language_choice

def handle_file_upload(file):
    if file is None:
        return "", gr.update(visible=False)
    try:
        text = extract_text_from_file(file.name)
        lang_code = detect_language(text)
        lang_name = "Tiếng Việt" if lang_code == "vi" else "Tiếng Anh"
        detect_msg = f"ℹ️ **Đã tự động phát hiện ngôn ngữ:** {lang_name}"
        return text, gr.update(value=detect_msg, visible=True)
    except Exception as e:
        return f"Lỗi đọc file: {str(e)}", gr.update(value=f"❌ Lỗi: {str(e)}", visible=True)

def handle_single_read(text, voice, language, session_id):
    from tools.TextToSpeech.app import synthesize_speech, model_loaded
    if not model_loaded:
        yield None, "⚠️ Vui lòng quay lại tab 'Text to Speech' để nạp Model trước khi đọc tài liệu!"
        return
    lang_code, lang_name = get_synthesis_language(text, language)
    yield None, f"⏳ Đang chuẩn bị đọc bằng {lang_name}..."
    for audio_path, status in synthesize_speech(
        text, voice, None, None,
        "preset_mode", "Standard (Một lần)", True, 32,
        0.8, 250, session_id
    ):
        yield audio_path, status

def handle_conversation_read(text, language, silence_dur, session_id, *speaker_args):
    from tools.TextToSpeech.app import synthesize_conversation, model_loaded
    if not model_loaded:
        yield None, "⚠️ Vui lòng quay lại tab 'Text to Speech' để nạp Model trước khi đọc tài liệu!"
        return
    # First 8 args are speaker names, next 8 are voices
    speaker_names = list(speaker_args[:8])
    speaker_voices = list(speaker_args[8:16])
    
    yield None, "⏳ Đang chuẩn bị đọc hội thoại..."
    for audio_path, status in synthesize_conversation(
        text,
        *speaker_names,
        *speaker_voices,
        silence_dur, 0.8, 250, session_id
    ):
        yield audio_path, status

def handle_read(mode, text, voice, language, silence_dur, session_id, *speaker_args):
    if not text or not text.strip():
        yield None, "⚠️ Vui lòng nhập hoặc tải lên văn bản!"
        return
    if mode == "Đọc truyện (Đơn ca)":
        yield from handle_single_read(text, voice, language, session_id)
    else:
        yield from handle_conversation_read(text, language, silence_dur, session_id, *speaker_args)

def get_voices_update_func():
    from tools.TextToSpeech.app import PRESET_VOICES_CACHE, CONV_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return [gr.update()] * (1 + 8)
    default_v = PRESET_VOICES_CACHE[0][1] if isinstance(PRESET_VOICES_CACHE[0], tuple) else PRESET_VOICES_CACHE[0]
    conv_default = CONV_VOICES_CACHE[0][1] if (CONV_VOICES_CACHE and isinstance(CONV_VOICES_CACHE[0], tuple)) else (CONV_VOICES_CACHE[0] if CONV_VOICES_CACHE else None)
    return [
        gr.update(choices=PRESET_VOICES_CACHE, value=default_v),
        *[gr.update(choices=CONV_VOICES_CACHE, value=conv_default) for _ in range(8)]
    ]

def on_mode_change(mode):
    is_conv = mode == "Hội thoại (Đa ca)"
    return gr.update(visible=not is_conv), gr.update(visible=is_conv)

def request_stop():
    from tools.TextToSpeech.app import _STOP_EVENT
    print("🛑 STOP REQUESTED via DocReader button click.")
    _STOP_EVENT.set()
    return None, "⏹️ Đã dừng đọc tài liệu.", gr.update(interactive=False)

# Setup layout
MAX_SPEAKERS = 8

with gr.Blocks() as demo:
    session_id_state = gr.State("")
    gr.HTML("<div style='margin-bottom: 20px;'><h2>📖 Đọc tài liệu & Kịch bản hội thoại</h2><p style='color: #64748b; margin-top: 5px;'>Trích xuất tài liệu .txt, .docx để đọc hoặc phát kịch bản hội thoại đa nhân vật sử dụng mô hình VieNeu-TTS chạy cục bộ.</p></div>")
    
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                label="Tải lên tài liệu (.txt, .docx)",
                file_types=[".txt", ".docx"]
            )
            detected_lang_label = gr.Markdown(value="", visible=False)
            
            text_input = gr.Textbox(
                lines=12,
                label="Nội dung văn bản (Có thể chỉnh sửa)",
                placeholder="Nội dung tài liệu sẽ hiển thị ở đây sau khi tải lên, hoặc bạn tự nhập nội dung..."
            )
            
            with gr.Row():
                language_select = gr.Dropdown(
                    choices=["Tự động phát hiện", "Tiếng Việt", "Tiếng Anh"],
                    value="Tự động phát hiện",
                    label="Ngôn ngữ đọc"
                )
                mode_select = gr.Radio(
                    choices=["Đọc truyện (Đơn ca)", "Hội thoại (Đa ca)"],
                    value="Đọc truyện (Đơn ca)",
                    label="Chế độ đọc"
                )
                
            # Single Speaker Voice settings
            with gr.Group(visible=True) as single_voice_group:
                voice_select = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="🎤 Giọng đọc truyện",
                    interactive=True,
                    allow_custom_value=True
                )
                
            # Conversation Settings
            with gr.Group(visible=False) as multi_voice_group:
                with gr.Row():
                    btn_detect_speakers = gr.Button("🔍 Quét nhân vật", variant="secondary")
                    silence_slider = gr.Slider(minimum=0, maximum=3, value=0.5, step=0.1, label="⏱️ Khoảng lặng giữa các câu (giây)")
                
                gr.Markdown("### 🎭 Cấu hình giọng nhân vật")
                speaker_name_boxes = []
                speaker_voice_dds = []
                speaker_slot_rows = []
                
                for i in range(MAX_SPEAKERS):
                    row_visible = i < 3
                    default_name = ""
                    if i == 0: default_name = "Phương"
                    elif i == 1: default_name = "Dũng"
                    elif i == 2: default_name = "Hùng"
                    
                    with gr.Row(visible=row_visible) as row:
                        name = gr.Textbox(
                            value=default_name,
                            label="👤 Nhân vật",
                            interactive=False,
                            scale=1,
                            min_width=120
                        )
                        dd = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="🎤 Giọng đọc",
                            interactive=True,
                            scale=3,
                            allow_custom_value=True
                        )
                    speaker_slot_rows.append(row)
                    speaker_name_boxes.append(name)
                    speaker_voice_dds.append(dd)
                    
            btn_generate = gr.Button("🚀 Bắt đầu đọc tài liệu", variant="primary")
            btn_stop = gr.Button("🛑 Dừng", variant="stop", interactive=False)
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="Trạng thái",
                value="⏳ Chờ tải tài liệu hoặc nhập văn bản...",
                interactive=False
            )
            audio_output = gr.Audio(
                label="Âm thanh đầu ra",
                interactive=False
            )

    file_input.change(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[text_input, detected_lang_label]
    )
    
    mode_select.change(
        fn=on_mode_change,
        inputs=[mode_select],
        outputs=[single_voice_group, multi_voice_group]
    )
    
    from tools.TextToSpeech.app import extract_speakers_from_script
    btn_detect_speakers.click(
        fn=extract_speakers_from_script,
        inputs=[text_input],
        outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
    )
    
    # Read generation
    gen_inputs = [
        mode_select,
        text_input,
        voice_select,
        language_select,
        silence_slider,
        session_id_state,
        *speaker_name_boxes,
        *speaker_voice_dds
    ]
    
    gen_event = btn_generate.click(
        fn=handle_read,
        inputs=gen_inputs,
        outputs=[audio_output, status_output]
    )
    
    btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
    gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)
    
    btn_stop.click(fn=request_stop, outputs=[audio_output, status_output, btn_stop])

# Expose components and update function
voice_components = [voice_select] + speaker_voice_dds
update_voices_fn = get_voices_update_func
