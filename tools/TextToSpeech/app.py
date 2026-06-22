import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr
print("⏳ Đang khởi động Pum's Tools — TextToSpeech... Vui lòng chờ...")
import soundfile as sf
import tempfile
from vieneu import Vieneu
import os
import time
import numpy as np
import queue
import threading
import yaml
import gc

from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks, env_bool, get_silence_duration_v2
from vieneu_utils.phonemize_text import phonemize_to_chunks
from vieneu_utils.phonemize_text import PuncNormalizer as Normalizer

from tools.TextToSpeech.ui_utils import (
    _format_duration,
    _split_estimate_status,
    wrap_with_estimate,
    cleanup_gpu_memory,
    get_ref_text_cached,
    on_codec_change,
    validate_audio_duration,
    on_custom_id_change
)
from tools.TextToSpeech.ui_constants import (
    theme,
    css,
    head_html,
    DEFAULT_TEXT_GPU,
    DEFAULT_TEXT_TURBO,
    DEFAULT_TEXT_V3,
    DEFAULT_CONV_SCRIPT,
)

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})

# Filter by GPU availability
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}
filtered_backbones["VieNeu-TTS-v3-Turbo (Thử nghiệm)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "supports_streaming": False,
    "description": "🆕 v3 Turbo (early access) — 48kHz. Giọng mặc định dùng speaker token; Voice Cloning clone từ audio mẫu. Hỗ trợ tag cảm xúc [cười]/[hắng giọng]/[thở dài] (thử nghiệm)."
}

if HAS_GPU:
    filtered_backbones["VieNeu-TTS-v2 (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 2 - hỗ trợ song ngữ (Anh-Việt) và chế độ podcast"
    }
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 1 - ổn định, production-ready"
    }

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "NeuCodec (ONNX)": {
        "repo": "neuphonic/neucodec-onnx-decoder-int8",
        "description": "Codec siêu nhẹ, tối ưu cho CPU (ONNX)",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
tts = None
current_backbone = None
current_codec = None
model_loaded = False
model_loading = False
using_lmdeploy = False
PRESET_VOICES_CACHE = []
CONV_VOICES_CACHE = []
MAX_SPEAKERS = 8

_text_normalizer = Normalizer()


def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append("MPS")
        elif torch.cuda.is_available():
            devices.append("CUDA")
    except ImportError:
        pass
    return devices


def _supports_cloning(backbone_choice: str) -> bool:
    """Voice Cloning availability by model."""
    c = (backbone_choice or "").lower()
    return "v3" in c or c == "vieneu-tts-v2 (gpu)"


def get_model_status_message() -> str:
    """Reconstruct status message from global state."""
    global model_loaded, tts
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    return "✅ Model đã tự động tải thành công và sẵn sàng sử dụng!"


def restore_ui_state():
    """Update UI components based on persistence."""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg,
        gr.update(interactive=model_loaded),
        gr.update(interactive=model_loaded),
        gr.update(interactive=False)
    )


def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    if sys.platform == "darwin":
        return False

    bc = backbone_choice.lower()
    if "gguf" in bc or "v2-turbo" in bc or "v3" in bc:
        return False

    try:
        import torch
        if device_choice == "Auto":
            has_gpu = torch.cuda.is_available()
        elif device_choice == "CUDA":
            has_gpu = torch.cuda.is_available()
        else:
            has_gpu = False
        return has_gpu
    except ImportError:
        return False


def load_model(backbone_choice: str, codec_choice: str, device_choice: str,
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "",
               custom_hf_token: str = ""):
    """Load model with optimizations."""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy, model_loading, PRESET_VOICES_CACHE, CONV_VOICES_CACHE
    
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    if model_loading:
        while model_loading:
            yield (
                "⏳ Model đang được tải ở một phiên khác... Vui lòng đợi.",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(),
                *slot_no_updates
            )
            time.sleep(1)
        
        if model_loaded:
            try:
                voices = tts.list_preset_voices()
            except Exception:
                voices = []
            has_voices = len(voices) > 0
            if has_voices:
                default_v = tts._default_voice
                is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
                voice_values = [v[1] for v in voices] if is_tuple else voices
                if not default_v and voice_values:
                    default_v = voice_values[0]
                if default_v and default_v not in voice_values:
                    if is_tuple:
                        voices.append((default_v, default_v))
                    else:
                        voices.append(default_v)
                if is_tuple:
                    voices.sort(key=lambda x: str(x[0]))
                else:
                    voices.sort()
                voice_update = gr.update(choices=voices, value=default_v, interactive=True)
                PRESET_VOICES_CACHE = voices
                def _check_podcast(v_id):
                    val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                    if isinstance(val, str):
                        return val.strip().lower() == "true"
                    return bool(val)
                CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]
                slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
                tab_p = gr.update(visible=True)
                tab_c = gr.update(visible=_supports_cloning(backbone_choice))
                tab_sel = gr.update(selected="preset_mode")
                mode_state = "preset_mode"
            else:
                msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
                voice_update = gr.update(choices=[msg], value=msg, interactive=False)
                slot_dd_update = gr.update(choices=[])
                tab_p = gr.update(visible=True)
                tab_c = gr.update(visible=_supports_cloning(backbone_choice))
                tab_sel = gr.update(selected="preset_mode")
                mode_state = "preset_mode"
            is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
            is_v3_conv = "v3" in (backbone_choice or "").lower()
            conv_tab_update = gr.update(visible=is_v2 or is_v3_conv)
            slot_updates = [slot_dd_update] * MAX_SPEAKERS
            
            yield (
                get_model_status_message(),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                voice_update,
                tab_p, tab_c, tab_sel, mode_state,
                conv_tab_update,
                *slot_updates
            )
        return

    model_loading = True
    lmdeploy_error_reason = None
    model_loaded = False

    yield (
        f"⏳ Đang tự động tải model {backbone_choice}... Vui lòng kiên nhẫn.",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(),
        *slot_no_updates
    )

    try:
        if tts is not None:
            tts = None
            cleanup_gpu_memory()

        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    *slot_no_updates
                )
                return

            if "lora" in custom_model_id.lower():
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(),
                        *slot_no_updates
                    )
                    return

                base_config = BACKBONE_CONFIGS[custom_base_model]
                backbone_config = {
                    "repo": base_config["repo"],
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False,
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]

        codec_config = CODEC_CONFIGS[codec_choice]
        use_lmdeploy = False

        if custom_loading:
            if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                use_lmdeploy = False
            elif is_merged_lora:
                use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
            else:
                use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice)

        if "v2-Turbo" in backbone_choice or "v3" in backbone_choice.lower():
            should_use_generic_fast = False
        elif custom_loading:
            should_use_generic_fast = False
        else:
            should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)

        if should_use_generic_fast:
            use_lmdeploy = True

        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")

            backbone_device = "cuda"
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                try:
                    import torch
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    codec_device = "cpu"

            target_backbone_repo = backbone_config["repo"]

            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(__file__), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)

                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    yield (
                        f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy...",
                        gr.update(interactive=False), gr.update(interactive=False),
                        gr.update(interactive=False), gr.update(interactive=False),
                        gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(),
                        *slot_no_updates
                    )

                    try:
                        from vieneu.standard import VieNeuTTS
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"

                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device,
                            codec_repo=codec_config["repo"],
                            codec_device="cpu",
                            hf_token=custom_hf_token
                        )
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)

                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()

                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)

                        try:
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not save slow tokenizer files: {e}")

                        import json
                        voices_json_path = os.path.join(cache_dir, "voices.json")
                        voices_content = {
                            "meta": {"note": "Automatically generated during LoRA merge"},
                            "default_voice": temp_tts._default_voice,
                            "presets": temp_tts._preset_voices
                        }
                        with open(voices_json_path, 'w', encoding='utf-8') as f:
                            json.dump(voices_content, f, ensure_ascii=False, indent=2)

                        del temp_tts
                        cleanup_gpu_memory()
                        print("   ✅ Merge & Save successfully!")

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {target_backbone_repo} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")

            try:
                from vieneu.fast import FastVieNeuTTS
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=False,
                    enable_triton=True,
                    hf_token=custom_hf_token
                )
                using_lmdeploy = True

            except Exception as e:
                import traceback
                traceback.print_exc()

                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH."
                else:
                    lmdeploy_error_reason = f"{error_str}"

                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang chuyển về backend mặc định...",
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    *slot_no_updates
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False

        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower

                if is_gguf_backbone:
                    if sys.platform == "darwin":
                        backbone_device = "gpu"
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        codec_device = "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"
            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()
                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                backbone_device = "gpu"

            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")

            if "v3-Turbo" in backbone_choice:
                print("   🆕 Mode: v3 Turbo (CPU=ONNX / GPU=PyTorch)")
                v3_device = "cpu" if str(backbone_device).lower() == "cpu" else "auto"
                tts = Vieneu(
                    mode="v3turbo",
                    backbone_repo=backbone_config["repo"],
                    device=v3_device,
                    hf_token=custom_hf_token,
                )
            elif "v2-Turbo" in backbone_choice:
                print("   ⚡ Mode: Turbo")
                mode = "turbo_gpu" if "GPU" in backbone_choice else "turbo"
                tts = Vieneu(
                    mode=mode,
                    backbone_repo=backbone_config["repo"],
                    decoder_repo=codec_config["repo"],
                    device=backbone_device,
                    backend="lmdeploy" if force_lmdeploy and "GPU" in backbone_choice else "standard",
                    hf_token=custom_hf_token
                )
            else:
                from vieneu.standard import VieNeuTTS
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token,
                    gguf_filename=backbone_config.get("gguf_filename")
                )

            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(),
                    *slot_no_updates
                )
                try:
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        print("   🔄 Merging LoRA into backbone...")
                        tts.backbone = tts.backbone.merge_and_unload()
                        tts._lora_loaded = False
                        tts._current_lora_repo = None
                        print("   ✅ Merged successfully!")
                    else:
                        print("   ⚠️ Warning: Model does not support merge_and_unload.")
                except Exception as e:
                    raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False

        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True

        # Prepare voice update
        try:
            voices = tts.list_preset_voices()
        except Exception:
            voices = []

        has_voices = len(voices) > 0

        if has_voices:
            default_v = tts._default_voice
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices

            if not default_v and voice_values:
                default_v = voice_values[0]

            if default_v and default_v not in voice_values:
                if is_tuple:
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)

            if is_tuple:
                voices.sort(key=lambda x: str(x[0]))
            else:
                voices.sort()

            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            PRESET_VOICES_CACHE = voices

            def _check_podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                if isinstance(val, str):
                    return val.strip().lower() == "true"
                return bool(val)

            CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]

            slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)

            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            slot_dd_update = gr.update(choices=[])

            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
        is_v3_conv = "v3" in (backbone_choice or "").lower()
        conv_tab_update = gr.update(visible=is_v2 or is_v3_conv)

        slot_updates = [slot_dd_update] * MAX_SPEAKERS

        success_msg = get_model_status_message()
        warning_msg = ""
        if lmdeploy_error_reason:
            warning_msg = (
                f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy:\n"
                f"👉 {lmdeploy_error_reason}\n"
                f"💡 Đã tự động chuyển về chế độ Standard."
            )
        if warning_msg:
            success_msg += warning_msg

        model_loading = False
        yield (
            success_msg,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_update,
            tab_p, tab_c, tab_sel, mode_state,
            conv_tab_update,
            *slot_updates
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        model_loading = False
        using_lmdeploy = False

        yield (
            f"❌ Lỗi khi tải model: {str(e)}",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
            *slot_no_updates
        )


def auto_load_model_if_needed(backbone_choice: str, codec_choice: str, device_choice: str,
                              force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "",
                              custom_hf_token: str = ""):
    global model_loaded, tts
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    if model_loaded and tts is not None:
        # Model is already loaded, restore UI state
        success_msg = get_model_status_message()
        try:
            voices = tts.list_preset_voices()
        except Exception:
            voices = []
        has_voices = len(voices) > 0
        if has_voices:
            default_v = tts._default_voice
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices
            if not default_v and voice_values:
                default_v = voice_values[0]
            if default_v and default_v not in voice_values:
                if is_tuple:
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)
            if is_tuple:
                voices.sort(key=lambda x: str(x[0]))
            else:
                voices.sort()
            voice_update = gr.update(choices=voices, value=default_v, interactive=True)
            global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
            PRESET_VOICES_CACHE = voices
            def _check_podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                if isinstance(val, str):
                    return val.strip().lower() == "true"
                return bool(val)
            CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]
            slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"
        else:
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            slot_dd_update = gr.update(choices=[])
            tab_p = gr.update(visible=True)
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
            tab_sel = gr.update(selected="preset_mode")
            mode_state = "preset_mode"

        is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
        is_v3_conv = "v3" in (backbone_choice or "").lower()
        conv_tab_update = gr.update(visible=is_v2 or is_v3_conv)
        slot_updates = [slot_dd_update] * MAX_SPEAKERS

        yield (
            success_msg,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_update,
            tab_p, tab_c, tab_sel, mode_state,
            conv_tab_update,
            *slot_updates
        )
        return

    # Otherwise load it
    yield from load_model(
        backbone_choice, codec_choice, device_choice, force_lmdeploy,
        custom_model_id, custom_base_model, custom_hf_token
    )


def resolve_voice_id(v_id: str) -> str:
    """Robustly resolve voice ID."""
    if not v_id:
        return v_id

    global PRESET_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return v_id

    for item in PRESET_VOICES_CACHE:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, value = item[0], item[1]
            if v_id == value or v_id == label:
                return value
        else:
            if v_id == item:
                return item

    return v_id


# --- 2. SYNTHESIS ---
# Cancellation event
_STOP_EVENT = threading.Event()


def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int, session_id: str = None):
    """Speech synthesis with optimization support."""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy

    _STOP_EVENT.clear()

    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return

    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return

    raw_text = text.strip()
    codec_config = CODEC_CONFIGS[current_codec]

    # Setup Reference
    yield None, "📄 Đang xử lý Reference..."

    try:
        ref_codes = None
        ref_text_raw = ""
        v3_voice_token_id = None

        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng.")

            v_id = resolve_voice_id(voice_choice)
            voice_data = tts.get_preset_voice(v_id)
            ref_codes = voice_data['codes']
            ref_text_raw = voice_data['text']
            v3_voice_token_id = voice_data.get('reserved_id')

        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Vui lòng upload file Audio mẫu!")

            cb_lower = (current_backbone or "").lower()
            needs_ref_text = "v2-turbo" not in cb_lower and "v3" not in cb_lower
            if needs_ref_text and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu!")

            ref_text_raw = custom_text.strip() if custom_text else ""
            ref_codes = tts.encode_reference(custom_audio)

        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return

    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        # v3 Turbo branch
        if "v3" in (current_backbone or "").lower():
            _t0 = time.time()
            yield None, "⏳ Đang tổng hợp (v3 Turbo)..."
            sr_v3 = getattr(tts, "sample_rate", 48000)
            try:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions

                v3_chunks = split_text_into_chunks(raw_text, max_chars=max_chars_chunk) or [raw_text]
                v3_bs = max(1, int(max_batch_size_run)) if use_batch else 1
                v3_engine_dev = getattr(getattr(tts, "engine", None), "device", None)
                v3_can_batch = (
                    v3_bs > 1 and len(v3_chunks) > 1
                    and v3_engine_dev is not None and v3_engine_dev.type == "cuda"
                )

                if v3_can_batch:
                    from vieneu.v3_turbo_serve import V3TurboBatchEngine
                    if getattr(tts, "_v3_batch_engine", None) is None:
                        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)
                    v3_wavs = []
                    for i in range(0, len(v3_chunks), v3_bs):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        group = v3_chunks[i:i + v3_bs]
                        yield None, f"⚡ v3 Turbo: lô {i // v3_bs + 1} ({len(group)} đoạn, batch size {v3_bs})..."
                        reqs = [{"phonemes": phonemize_text_with_emotions(c), "ref_codes": ref_codes,
                                 "voice_token_id": v3_voice_token_id} for c in group]
                        v3_wavs.extend(tts._v3_batch_engine.generate_batch(
                            reqs, temperature=temperature, max_new_frames=300))
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_p=0.15)
                else:
                    total_v3 = len(v3_chunks)
                    stream_kwargs = ({"voice": v_id} if mode_tab == "preset_mode"
                                     else {"ref_codes": ref_codes})
                    v3_wavs = []
                    chunk_durations = []
                    last_t = time.time()
                    yield None, f"⏳ v3 Turbo: Đang xử lý đoạn 1/{total_v3}..."
                    for i, chunk_wav in enumerate(tts.infer_stream(
                            raw_text, temperature=temperature,
                            max_chars=max_chars_chunk, **stream_kwargs)):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        now = time.time()
                        chunk_durations.append(now - last_t)
                        last_t = now
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            v3_wavs.append(chunk_wav)
                        done = i + 1
                        if done < total_v3:
                            avg = sum(chunk_durations) / len(chunk_durations)
                            eta = avg * (total_v3 - done)
                            yield None, (
                                f"⏳ v3 Turbo: Đã xong {done}/{total_v3} đoạn "
                                f"(ước tính còn lại: {_format_duration(eta)})... "
                                f"đang xử lý đoạn {done + 1}/{total_v3}"
                            )
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_p=0.15)
            except Exception as e:
                yield None, f"❌ Lỗi tổng hợp (v3 Turbo): {str(e)}"
                return
            if wav is None or len(wav) == 0:
                yield None, "❌ Không sinh được audio nào."
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, wav, sr_v3)
                out_path_v3 = tmp.name
            _dt = time.time() - _t0
            _spd = f", Tốc độ: {len(wav)/sr_v3/_dt:.2f}x realtime" if _dt > 0 else ""
            yield out_path_v3, f"✅ Hoàn tất! (v3 Turbo, Thời gian: {_dt:.2f}s{_spd})"
            cleanup_gpu_memory()
            return

        # Standard/v1/v2 branch
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")

        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            text_chunks = []
            for raw_chunk in split_text_into_chunks(raw_text, max_chars=max_chars_chunk):
                normalized_chunk = _text_normalizer.normalize(raw_chunk)
                text_chunks.extend(split_text_into_chunks(normalized_chunk, max_chars=max_chars_chunk))

        total_chunks = len(text_chunks)
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name} ({total_chunks} đoạn)..."

        all_wavs = []
        sr = 24000
        start_time = time.time()

        try:
            if is_v2_turbo:
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."

                    chunk_wav = tts.infer(
                        chunk.text,
                        ref_codes=ref_codes,
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True,
                        skip_phonemize=True
                    )

                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
                        if i < total_chunks - 1:
                            sil_dur = get_silence_duration_v2(chunk)
                            sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                            all_wavs.append(sil_wav)

            elif use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run

                for i in range(0, total_chunks, max_batch_size_run):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return

                    batch_idx = i // max_batch_size_run
                    yield None, f"⚡ Đang xử lý batch {batch_idx+1}/{num_batches}..."

                    current_batch = text_chunks[i: i + max_batch_size_run]
                    batch_wavs = tts.infer_batch(
                        current_batch,
                        ref_codes=ref_codes,
                        ref_text=ref_text_raw,
                        max_batch_size=max_batch_size_run,
                        temperature=temperature,
                        skip_normalize=True
                    )
                    for chunk_wav in batch_wavs:
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
            else:
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wav = tts.infer(
                        chunk,
                        ref_codes=ref_codes,
                        ref_text=ref_text_raw,
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)

            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return

            yield None, "💾 Đang ghép file và lưu..."

            silence_p = 0.15 if not is_v2_turbo else 0.0
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name

            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""

            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"

            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            cleanup_gpu_memory()

        except Exception as e:
            if 'torch' in sys.modules:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    cleanup_gpu_memory()
                    yield None, f"❌ GPU hết VRAM! Hãy thử giảm Batch Size hoặc độ dài văn bản."
                    return

            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return

    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3

        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""

        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            text_chunks = []
            for raw_chunk in split_text_into_chunks(raw_text, max_chars=max_chars_chunk):
                normalized_chunk = _text_normalizer.normalize(raw_chunk)
                text_chunks.extend(split_text_into_chunks(normalized_chunk, max_chars=max_chars_chunk))

        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                for i, chunk_text in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        break

                    if is_v2_turbo:
                        stream_gen = tts.infer_stream(
                            chunk_text.text, ref_codes=ref_codes,
                            temperature=temperature, max_chars=max_chars_chunk,
                            skip_normalize=True, skip_phonemize=True, emotion_tag=""
                        )
                    else:
                        stream_gen = tts.infer_stream(
                            chunk_text, ref_codes=ref_codes, ref_text=ref_text_raw,
                            temperature=temperature, max_chars=max_chars_chunk,
                            skip_normalize=True, emotion_tag=""
                        )

                    for part_idx, audio_part in enumerate(stream_gen):
                        if _STOP_EVENT.is_set():
                            break
                        if audio_part is None or len(audio_part) == 0:
                            continue

                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                blended = (audio_part[:overlap] * fade_in + previous_tail[-overlap:] * fade_out)
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])

                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part

                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))

                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))

                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)

        threading.Thread(target=producer_thread, daemon=True).start()
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."

        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break

        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."

        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue

        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"

            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            cleanup_gpu_memory()


synthesize_speech_with_estimate = wrap_with_estimate(synthesize_speech)


def synthesize_conversation_with_empty_estimate(*args):
    for audio_path, status in synthesize_conversation(*args):
        yield audio_path, status, ""


# --- 3. CONVERSATION LOGIC ---

def _synthesize_conversation_v3(lines, mapping, temperature, max_chars_chunk, silence_duration):
    """v3 Turbo conversation: batch the WHOLE conversation at batch size 32."""
    global tts
    from collections import defaultdict
    from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    sr = getattr(tts, "sample_rate", 48000)
    t0 = time.time()

    def _voice_for(spk_name):
        cfg = mapping.get(spk_name.lower())
        v_id = (cfg or {}).get('voice') or tts._default_voice
        try:
            vd = tts.get_preset_voice(v_id)
        except Exception:
            vd = tts.get_preset_voice(tts._default_voice)
        rc = vd['codes']
        if 'torch' in sys.modules:
            import torch
            if isinstance(rc, torch.Tensor):
                rc = rc.cpu().numpy()
        return np.asarray(rc), vd.get('reserved_id')

    dev = getattr(getattr(tts, "engine", None), "device", None)
    is_cuda = dev is not None and getattr(dev, "type", None) == "cuda"
    if not is_cuda:
        all_wavs = []
        for li, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            cfg = mapping.get(line['speaker'].lower())
            v_id = (cfg or {}).get('voice') or tts._default_voice
            yield None, f"⏳ [{li+1}/{len(lines)}] {line['speaker']}: {line['text'][:30]}..."
            try:
                wav = tts.infer(line['text'], voice=v_id, temperature=temperature, max_chars=max_chars_chunk)
            except Exception as e:
                print(f"❌ Lỗi câu {li+1}: {e}")
                continue
            if wav is not None and len(wav):
                all_wavs.append(wav)
                if li < len(lines) - 1 and silence_duration > 0:
                    all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))
        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {time.time()-t0:.1f}s, CPU tuần tự)"
        cleanup_gpu_memory()
        return

    voice_cache = {}
    reqs, req_line = [], []
    for li, line in enumerate(lines):
        key = line['speaker'].lower()
        if key not in voice_cache:
            voice_cache[key] = _voice_for(line['speaker'])
        ref_codes, vtok = voice_cache[key]
        chunks = split_text_into_chunks(line['text'], max_chars=max_chars_chunk) or [line['text']]
        for c in chunks:
            reqs.append({"phonemes": phonemize_text_with_emotions(c),
                         "ref_codes": ref_codes, "voice_token_id": vtok})
            req_line.append(li)

    if not reqs:
        yield None, "❌ Không có lời thoại để tổng hợp."
        return

    if getattr(tts, "_v3_batch_engine", None) is None:
        from vieneu.v3_turbo_serve import V3TurboBatchEngine
        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)

    BS = 32
    total_batches = (len(reqs) + BS - 1) // BS
    wavs_flat = []
    for bi, i in enumerate(range(0, len(reqs), BS)):
        if _STOP_EVENT.is_set():
            yield None, "⏹️ Đã dừng hội thoại."
            return
        group = reqs[i:i + BS]
        yield None, f"⚡ v3 Turbo hội thoại: lô {bi + 1}/{total_batches} ({len(group)} đoạn, batch 32)..."
        wavs_flat.extend(tts._v3_batch_engine.generate_batch(
            group, temperature=temperature, max_new_frames=300))

    by_line = defaultdict(list)
    for w, li in zip(wavs_flat, req_line):
        by_line[li].append(w)

    all_wavs = []
    for li in range(len(lines)):
        lw = join_audio_chunks(by_line[li], sr=sr, silence_p=0.15) if by_line[li] else None
        if lw is None or len(lw) == 0:
            continue
        all_wavs.append(lw)
        if li < len(lines) - 1 and silence_duration > 0:
            all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))

    if not all_wavs:
        yield None, "❌ Không thể tạo được âm thanh nào!"
        return

    yield None, "🪄 Đang ghép nối âm thanh..."
    final_wav = np.concatenate(all_wavs)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, final_wav, sr)
        elapsed = time.time() - t0
        yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {len(reqs)} đoạn, {elapsed:.1f}s, batch 32)"
    cleanup_gpu_memory()


def synthesize_conversation(script_text: str, *args):
    """Synthesizes multi-speaker conversation from a script."""
    speaker_names = list(args[:MAX_SPEAKERS])
    speaker_voices = list(args[MAX_SPEAKERS:MAX_SPEAKERS*2])
    silence_duration = args[MAX_SPEAKERS * 2]
    temperature = args[MAX_SPEAKERS * 2 + 1]
    max_chars_chunk = args[MAX_SPEAKERS * 2 + 2]
    session_id = args[MAX_SPEAKERS * 2 + 3] if len(args) > MAX_SPEAKERS * 2 + 3 else None

    global tts, model_loaded, using_lmdeploy

    _STOP_EVENT.clear()

    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return

    if not script_text or script_text.strip() == "":
        yield None, "⚠️ Vui lòng nhập kịch bản hội thoại!"
        return

    # Parse Script
    lines = []
    for line in script_text.strip().split('\n'):
        if not line.strip():
            continue
        if ':' in line:
            parts = line.split(':', 1)
            lines.append({'speaker': parts[0].strip(), 'text': parts[1].strip()})
        else:
            if lines:
                lines[-1]['text'] += " " + line.strip()
            else:
                lines.append({'speaker': 'Narrator', 'text': line.strip()})

    if not lines:
        yield None, "⚠️ Không tìm thấy lời thoại hợp lệ!"
        return

    # Build Speaker Mapping
    mapping = {}
    for name, voice in zip(speaker_names, speaker_voices):
        name = str(name).strip() if name else ""
        if not name:
            continue
        v_id = resolve_voice_id(str(voice)) if voice else ""
        mapping[name.lower()] = {
            'type': 'Preset',
            'voice': v_id,
            'ref_text': ''
        }

    # v3 Turbo batch
    if "v3" in (current_backbone or "").lower():
        yield from _synthesize_conversation_v3(
            lines, mapping, temperature, max_chars_chunk, silence_duration
        )
        return

    # v1/v2 sequential
    all_wavs = []
    sr = 24000
    total_lines = len(lines)

    yield None, f"🎭 Đang khởi tạo hội thoại ({total_lines} câu)..."
    start_time = time.time()

    try:
        for i, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            spk_name = line['speaker']
            text = line['text']

            yield None, f"⏳ [{i+1}/{total_lines}] {spk_name}: {text[:30]}..."

            ref_codes = None
            ref_text_val = None
            current_voice_obj = None

            config = mapping.get(spk_name.lower())

            if not config:
                try:
                    default_v_id = tts._default_voice
                    if not default_v_id:
                        dv_list = tts.list_preset_voices()
                        if dv_list:
                            first = dv_list[0]
                            default_v_id = first[1] if isinstance(first, tuple) else first

                    if default_v_id:
                        current_voice_obj = tts.get_preset_voice(default_v_id)
                        ref_codes = current_voice_obj['codes']
                        ref_text_val = current_voice_obj['text']
                except Exception as e:
                    print(f"  ❌ Fallback failed: {e}")
            else:
                try:
                    v_id = config['voice']
                    if config['type'] == "Preset":
                        current_voice_obj = tts.get_preset_voice(v_id)
                        if current_voice_obj and 'codes' in current_voice_obj:
                            ref_codes = current_voice_obj['codes']
                            ref_text_val = current_voice_obj['text']
                    else:
                        if v_id and os.path.exists(v_id):
                            ref_codes = tts.encode_reference(v_id)
                            ref_text_val = config.get('ref_text', '')
                            current_voice_obj = {'codes': ref_codes, 'text': ref_text_val}
                except Exception as e:
                    print(f"  ❌ Lỗi nạp giọng cho {spk_name}: {e}")

            if 'torch' in sys.modules:
                import torch
                if isinstance(ref_codes, torch.Tensor):
                    ref_codes = ref_codes.cpu().numpy()

            try:
                wav = tts.infer(
                    text,
                    voice=current_voice_obj,
                    ref_codes=ref_codes,
                    ref_text=ref_text_val,
                    temperature=temperature,
                    max_chars=max_chars_chunk,
                    emotion_tag="<|emotion_0|>"
                )

                all_wavs.append(wav)

                if i < total_lines - 1 and silence_duration > 0:
                    silence_len = int(sr * silence_duration)
                    silence = np.zeros(silence_len)
                    all_wavs.append(silence)

            except Exception as e:
                print(f"❌ Lỗi tổng hợp câu {i+1}: {e}")
                continue

        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return

        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            elapsed = time.time() - start_time
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({total_lines} câu, xử lý trong {elapsed:.1f}s)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ Lỗi hệ thống: {str(e)}"


def extract_speakers_from_script(script):
    """Find unique speakers and return gr.update() lists for the 8 slot components."""
    global CONV_VOICES_CACHE
    if not script:
        name_updates = [gr.update(value="", visible=False)] * MAX_SPEAKERS
        dd_updates = [gr.update(value=None, visible=False)] * MAX_SPEAKERS
        row_updates = [gr.update(visible=False)] * MAX_SPEAKERS
        return name_updates + dd_updates + row_updates

    speakers = []
    seen = set()
    for line in script.strip().split('\n'):
        if ':' in line:
            s = line.split(':', 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                speakers.append(s)

    def _best_match(name):
        if not CONV_VOICES_CACHE:
            return None

        name_l = name.lower()

        overrides = {
            "phương": "Trúc Ly",
            "dũng": "Thanh Bình",
            "hùng": "Thái Sơn"
        }
        if name_l in overrides:
            target = overrides[name_l].lower()
            for v in CONV_VOICES_CACHE:
                label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
                if target in label.lower() or target in value.lower():
                    return value

        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l == label.lower() or name_l == value.lower():
                return value

        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l in label.lower() or name_l in value.lower() or label.lower() in name_l or value.lower() in name_l:
                return value

        first_voice = CONV_VOICES_CACHE[0]
        return first_voice[1] if isinstance(first_voice, tuple) else first_voice

    name_updates, dd_updates, row_updates = [], [], []
    for i in range(MAX_SPEAKERS):
        if i < len(speakers):
            name_updates.append(gr.update(value=speakers[i], visible=True))
            dd_updates.append(gr.update(value=_best_match(speakers[i]), choices=CONV_VOICES_CACHE, visible=True))
            row_updates.append(gr.update(visible=True))
        else:
            name_updates.append(gr.update(value="", visible=False))
            dd_updates.append(gr.update(value=None, choices=CONV_VOICES_CACHE, visible=False))
            row_updates.append(gr.update(visible=False))

    return name_updates + dd_updates + row_updates


# --- 4. GRADIO UI ---

with gr.Blocks(theme=theme, css=css, title="Pum's Tools — TextToSpeech", head=head_html) as demo:
    session_id_state = gr.State("")

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">Pum's Tools</span>
    </h1>
    <p class="header-subtitle">🗣️ Text to Speech</p>
</div>
        """)

        # --- CONFIGURATION (Hidden from UI, but active in backend) ---
        with gr.Group(visible=False):
            with gr.Row():
                default_backbone = "VieNeu-TTS-v3-Turbo (Thử nghiệm)"
                if default_backbone not in BACKBONE_CONFIGS:
                    default_backbone = list(BACKBONE_CONFIGS.keys())[0]

                if "v3" in default_backbone.lower():
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.8
                    default_text = DEFAULT_TEXT_V3
                elif "Turbo" in default_backbone:
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.4
                    default_text = DEFAULT_TEXT_TURBO
                elif "(CPU)" in default_backbone:
                    default_codec = "NeuCodec (ONNX)"
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU
                else:
                    default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU

                default_batch_size = 32 if "v3" in default_backbone.lower() else 4

                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()) + ["Custom Model"],
                    value=default_backbone,
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(
                    list(CODEC_CONFIGS.keys()),
                    value=default_codec,
                    label="🎵 Codec",
                    interactive=False
                )
                device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")

            with gr.Row(visible=False) as custom_model_group:
                custom_backbone_model_id = gr.Textbox(
                    label="📦 Custom Model ID",
                    placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                    info="Nhập HuggingFace Repo ID hoặc đường dẫn local",
                    scale=2
                )
                custom_backbone_hf_token = gr.Textbox(
                    label="🔑 HF Token (nếu private)",
                    placeholder="Để trống nếu repo public",
                    type="password",
                    scale=1
                )
                base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                custom_backbone_base_model = gr.Dropdown(
                    base_model_choices,
                    label="🔗 Base Model (cho LoRA)",
                    value=base_model_choices[0] if base_model_choices else None,
                    visible=False,
                    scale=1
                )

            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=True,
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    visible="v3" not in default_backbone.lower(),
                )

            btn_load = gr.Button("🔄 Tải Model", variant="primary")

        # --- MODEL STATUS DISPLAY (Visible) ---
        with gr.Group(elem_classes="model-status-group"):
            model_status = gr.Markdown("⏳ **Trạng thái model:** Đang tự động tải model... Vui lòng kiên nhẫn.", elem_classes="model-status-markdown")

        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                with gr.Tabs() as main_input_tabs:
                    # --- TAB 1: SINGLE SPEAKER ---
                    with gr.Tab("🦜 Đọc truyện", id="single_tab") as single_tab:
                        text_input = gr.Textbox(
                            label=f"Văn bản",
                            lines=8,
                            value=default_text,
                        )

                        with gr.Tabs() as tabs:
                            with gr.TabItem("👤 Preset", id="preset_mode") as tab_preset:
                                voice_select = gr.Dropdown(choices=[], value=None, label="Giọng mẫu", allow_custom_value=True)

                            with gr.TabItem("🦜 Voice Cloning", id="custom_mode", visible=False) as tab_custom:
                                clone_info_md = gr.Markdown(
                                    "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                                    "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
                                )
                                with gr.Group(visible=True) as cloning_elements_group:
                                    custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                                    cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                                    custom_text = gr.Textbox(label="Nội dung audio mẫu", visible=False)

                                    gr.Markdown("""
                                    **💡 Mẹo nhỏ:** Nếu kết quả Voice Cloning chưa như ý, hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất.
                                    """)

                        generation_mode = gr.Radio(
                            ["Standard (Một lần)"],
                            value="Standard (Một lần)",
                            label="Chế độ sinh"
                        )
                        btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)

                    # --- TAB 2: MULTI-SPEAKER CONVERSATION ---
                    with gr.Tab("🎭 Hội thoại", id="conv_tab", visible=False) as conv_tab:
                        conv_script_input = gr.Textbox(
                            label="Kịch bản hội thoại",
                            placeholder="Phương: Chào mọi người, mình là Phương...",
                            lines=10,
                            elem_classes="script-box",
                            value=DEFAULT_CONV_SCRIPT,
                        )

                        with gr.Row():
                            btn_detect_speakers = gr.Button("🔍 Quét nhân vật", size="sm", variant="secondary")
                            silence_slider = gr.Slider(minimum=0, maximum=3, value=0.1, step=0.1, label="⏱️ Khoảng lặng (giây)")

                        gr.Markdown("### 🎭 Cấu hình giọng đọc")
                        gr.Markdown("*Nhấn **Quét nhân vật** để tự động phát hiện và ánh xạ giọng đọc.*")

                        speaker_name_boxes = []
                        speaker_voice_dds = []
                        speaker_slot_rows = []

                        for _i in range(MAX_SPEAKERS):
                            _default_name = ""
                            _default_voice = None
                            _row_visible = False

                            if _i == 0:
                                _default_name = "Phương"
                                _default_voice = "Ly"
                                _row_visible = True
                            elif _i == 1:
                                _default_name = "Dũng"
                                _default_voice = "Binh"
                                _row_visible = True
                            elif _i == 2:
                                _default_name = "Hùng"
                                _default_voice = "Sơn"
                                _row_visible = True

                            with gr.Row(visible=_row_visible) as _row:
                                _name = gr.Textbox(
                                    value=_default_name,
                                    label="👤 Nhân vật",
                                    interactive=False,
                                    scale=1,
                                    min_width=120
                                )
                                _dd = gr.Dropdown(
                                    choices=PRESET_VOICES_CACHE,
                                    value=_default_voice,
                                    label="🎤 Giọng đọc",
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                            speaker_slot_rows.append(_row)
                            speaker_name_boxes.append(_name)
                            speaker_voice_dds.append(_dd)

                        btn_generate_conv = gr.Button("🎭 Bắt đầu hội thoại", variant="primary", interactive=False)

                # Global Generation Settings
                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=True,
                        label="⚡ Batch Processing",
                        info="Xử lý nhiều đoạn cùng lúc (chỉ áp dụng khi sử dụng GPU)"
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=default_batch_size,
                        step=1,
                        label="📊 Batch Size (Generation)",
                        info="Số đoạn xử lý cùng lúc. Cao = nhanh hơn nhưng tốn VRAM hơn."
                    )

                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=default_temp, step=0.1,
                            label="🌡️ Temperature",
                            info="Cao = đa dạng cảm xúc hơn. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512, value=256, step=32,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý."
                        )

                current_mode_state = gr.State("preset_mode")

                with gr.Row():
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)

            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                with gr.Group():
                    status_output = gr.Textbox(
                        label="Trạng thái",
                        elem_classes="status-box",
                        lines=2,
                        max_lines=10,
                        show_copy_button=True
                    )
                with gr.Group():
                    estimate_output = gr.Textbox(
                        label="Ước tính thời gian",
                        elem_classes="estimate-box",
                        lines=2,
                        max_lines=4,
                        show_copy_button=True
                    )
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Watermarker).</div>")

    # --- EVENT HANDLERS ---
    codec_select.change(
        on_codec_change,
        inputs=[codec_select, current_mode_state],
        outputs=[tab_custom, tabs, current_mode_state]
    )

    tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
    tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)

    custom_audio.change(validate_audio_duration, inputs=[custom_audio], outputs=[cloning_warning_msg])

    def on_backbone_change(choice):
        is_custom = (choice == "Custom Model")
        is_v3 = "v3" in (choice or "").lower()
        is_v2_gpu = (choice == "VieNeu-TTS-v2 (GPU)")
        clone_ok = is_v3 or is_v2_gpu

        is_hw_accel_supported = "(GPU)" in choice or "v2-Turbo" in choice or "v3" in choice.lower() or is_custom

        if is_hw_accel_supported:
            dev_choices = get_available_devices()
            initial_dev = "Auto"
        else:
            dev_choices = ["CPU"]
            initial_dev = "CPU"

        if is_v3:
            codec_update = gr.update(value="VieNeu-Codec", interactive=False)
            text_update = gr.update(value=DEFAULT_TEXT_V3)
            temp_update = gr.update(value=0.8)
        elif "Turbo" in choice:
            codec_update = gr.update(value="VieNeu-Codec", interactive=False)
            text_update = gr.update(value=DEFAULT_TEXT_TURBO)
            temp_update = gr.update(value=0.4)
        elif "(CPU)" in choice:
            codec_update = gr.update(value="NeuCodec (ONNX)", interactive=False)
            text_update = gr.update(value=DEFAULT_TEXT_GPU)
            temp_update = gr.update(value=0.7)
        else:
            codec_update = gr.update(value="NeuCodec (Distill)", interactive=False)
            text_update = gr.update(value=DEFAULT_TEXT_GPU)
            temp_update = gr.update(value=0.7)

        if is_v2_gpu:
            clone_info_update = gr.update(value=(
                "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                "và **nhập đúng nội dung** của audio đó."
            ))
        else:
            clone_info_update = gr.update(value=(
                "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
            ))

        return (
            gr.update(visible=is_custom),
            codec_update,
            text_update,
            temp_update,
            gr.update(choices=dev_choices, value=initial_dev),
            gr.update(visible=clone_ok),
            gr.update(visible=clone_ok),
            gr.update(value=32 if is_v3 else 4),
            gr.update(visible=not is_v3),
            gr.update(visible=is_v2_gpu),
            clone_info_update,
        )

    backbone_select.change(
        on_backbone_change,
        inputs=[backbone_select],
        outputs=[
            custom_model_group,
            codec_select,
            text_input,
            temperature_slider,
            device_choice,
            cloning_elements_group,
            tab_custom,
            max_batch_size_run,
            use_lmdeploy_cb,
            custom_text,
            clone_info_md,
        ]
    )

    custom_backbone_model_id.change(
        on_custom_id_change,
        inputs=[custom_backbone_model_id],
        outputs=[custom_backbone_base_model, custom_audio, custom_text]
    )

    btn_load.click(
        fn=load_model,
        inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
        outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                 tab_preset, tab_custom, tabs, current_mode_state,
                 conv_tab,
                 *speaker_voice_dds]
    )

    # Conversation handlers
    btn_detect_speakers.click(
        fn=extract_speakers_from_script,
        inputs=[conv_script_input],
        outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
    )

    conv_gen_event = btn_generate_conv.click(
        fn=synthesize_conversation_with_empty_estimate,
        inputs=[conv_script_input,
                *speaker_name_boxes,
                *speaker_voice_dds,
                silence_slider, temperature_slider, max_chars_chunk_slider,
                session_id_state],
        outputs=[audio_output, status_output, estimate_output]
    )
    btn_generate_conv.click(lambda: gr.update(interactive=True), outputs=btn_stop)
    conv_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

    # Auto-adjust Temperature on Tab Switch
    conv_tab.select(
        fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else 1.0),
        inputs=backbone_select,
        outputs=temperature_slider
    )
    single_tab.select(
        fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else default_temp),
        inputs=backbone_select,
        outputs=temperature_slider
    )

    # Standard Generation Handlers
    gen_event = btn_generate.click(
        fn=synthesize_speech_with_estimate,
        inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state,
                generation_mode, use_batch, max_batch_size_run,
                temperature_slider, max_chars_chunk_slider, session_id_state],
        outputs=[audio_output, status_output, estimate_output]
    )
    btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
    gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

    # Stop Button
    def request_stop():
        print("🛑 STOP REQUESTED via button click.")
        _STOP_EVENT.set()
        return None, "⏹️ Đã dừng tạo giọng nói.", "", gr.update(interactive=False)

    btn_stop.click(fn=request_stop, outputs=[audio_output, status_output, estimate_output, btn_stop])

    # Automatic Model Load on Start
    demo.load(
        fn=auto_load_model_if_needed,
        inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
        outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                 tab_preset, tab_custom, tabs, current_mode_state,
                 conv_tab,
                 *speaker_voice_dds]
    )


def main():
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))

    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None
    share = env_bool("GRADIO_SHARE", default=is_on_colab)

    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    main()
