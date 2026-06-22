import sys
import os
import gradio as gr

# Add root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.TextToSpeech.app import demo as tts_demo
from tools.DocumentTranslator.app import demo as trans_demo
from tools.TextToSpeech.ui_constants import theme, css, head_html

# Create unified block interface
with gr.Blocks(theme=theme, css=css, title="Pum's Tools", head=head_html) as demo:
    gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">Pum's Tools</span>
    </h1>
    <p class="header-subtitle">Bộ sưu tập công cụ tiện ích đa năng</p>
</div>
    """)
    
    with gr.Tabs():
        with gr.Tab("🗣️ Text to Speech"):
            tts_demo.render()
            
        with gr.Tab("📄 Dịch tài liệu"):
            trans_demo.render()

def main():
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if ("SPACE_ID" in os.environ or "PORT" in os.environ) else "127.0.0.1")
    default_port = "7860" if "SPACE_ID" in os.environ else ("7861" if "PORT" not in os.environ else os.getenv("PORT"))
    server_port = int(os.getenv("GRADIO_SERVER_PORT", default_port))

    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None
    from vieneu_utils.core_utils import env_bool
    share = env_bool("GRADIO_SHARE", default=is_on_colab)

    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)

if __name__ == "__main__":
    main()
