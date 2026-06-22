---
title: Pums Tools
emoji: 🦜
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# 🛠️ Pum's Tools

Bộ sưu tập các công cụ tiện ích của Pum.

## 📁 Cấu trúc dự án

```
Pum's Tools/
├── tools/
│   └── TextToSpeech/       # 🗣️ Text to Speech (VieNeu-TTS)
│       ├── app.py           # Entry point — Gradio app
│       ├── config.yaml      # Cấu hình backbone/codec
│       ├── ui_constants.py  # Theme, CSS, default texts
│       ├── ui_utils.py      # Helper functions
│       └── requirements.txt # Dependencies
├── shared/                  # Code dùng chung giữa các tool
├── assets/                  # Tài nguyên (ảnh, icon, ...)
└── README.md
```

## 🗣️ TextToSpeech

Tool chuyển văn bản thành giọng nói dựa trên **VieNeu-TTS SDK**.

### Tính năng

- 🦜 **Đọc truyện** — Chọn giọng Preset hoặc Voice Cloning
- 🎭 **Hội thoại đa người nói** — Tự động quét nhân vật, ánh xạ giọng
- ⚡ **Batch Processing** — Xử lý nhiều đoạn cùng lúc (GPU)
- 🆕 **VieNeu-TTS v3 Turbo** — 48kHz, Voice Cloning, tag cảm xúc

### Chạy tool

```bash
# Từ thư mục gốc Pum's Tools (cần môi trường có vieneu đã cài)
python -m tools.TextToSpeech.app
```

Mặc định chạy ở `http://127.0.0.1:7861`

## 🚀 Hướng dẫn

Mỗi tool sẽ được đặt trong thư mục `tools/` với cấu trúc riêng.

---

*Made by Pum* 🐾
