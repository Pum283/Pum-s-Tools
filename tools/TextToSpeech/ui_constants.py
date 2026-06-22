import gradio as gr

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Outfit'), gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1400px; margin: auto; padding: 15px 0; }
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 30px 20px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.05);
}
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white !important;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}
.gradient-text {
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 8px;
}
.header-icon {
    color: white;
    font-size: 2.6rem;
    margin-right: 8px;
    vertical-align: middle;
}
.status-box {
    font-weight: 500;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    background: rgba(16, 185, 129, 0.03) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.05);
    transition: all 0.3s ease;
}
.status-box:hover {
    border-color: rgba(16, 185, 129, 0.4) !important;
}
.status-box textarea {
    text-align: center;
    font-family: 'Inter', sans-serif;
    background: transparent !important;
    border: none !important;
    resize: none !important;
    box-shadow: none !important;
    font-size: 0.95rem;
    color: #a7f3d0 !important;
}
.estimate-box {
    font-weight: 500;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    background: rgba(56, 189, 248, 0.03) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.05);
    transition: all 0.3s ease;
}
.estimate-box:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
}
.estimate-box textarea {
    text-align: center;
    font-family: 'Inter', sans-serif;
    background: transparent !important;
    border: none !important;
    resize: none !important;
    box-shadow: none !important;
    font-size: 0.95rem;
    color: #bae6fd !important;
}
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 20px;
    font-size: 0.95rem;
    text-align: center;
    color: #94a3b8 !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #cbd5e1 !important;
}
.model-card-item strong {
    color: #f8fafc !important;
}
.model-card-item span {
    color: #475569 !important;
}
.model-card-link {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
    border-bottom: 1px dashed rgba(56, 189, 248, 0.3);
}
.model-card-link:hover {
    color: #60a5fa;
    border-bottom-style: solid;
}
.script-box textarea {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    border-radius: 12px;
}
.speaker-table {
    margin-top: 10px;
}
/* Premium layout improvements */
.block {
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
}
.block:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08) !important;
}
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #0ea5e9 100%) !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
button.primary:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%) !important;
}
button.stop {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border-radius: 10px !important;
}
button.stop:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.35) !important;
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
}
.gr-examples {
    border-radius: 12px !important;
    padding: 10px !important;
}
/* Align vertical heights for the model configuration row */
.btn-align-with-inputs {
    align-self: flex-end !important;
    margin-bottom: 2px !important;
}
.btn-align-with-inputs button {
    height: 42px !important;
    line-height: 1 !important;
}
.model-status-group {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 20px;
    text-align: center;
}
.model-status-markdown p {
    margin: 0 !important;
    font-size: 0.95rem;
    color: #e2e8f0;
}
"""

head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
"""

DEFAULT_TEXT_GPU = "Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà."

DEFAULT_TEXT_TURBO = (
    "Trước đây, hệ thống điện chủ yếu sử dụng direct current, nhưng Tesla đã chứng minh rằng alternating current is more efficient for long-distance transmission. Nhờ đó, điện có thể được truyền đi xa hơn với ít tổn thất năng lượng hơn. Đây là một bước tiến cực kỳ quan trọng trong ngành điện.\n\n"
    "Một trong những phát minh nổi tiếng của ông là Tesla coil, một thiết bị có thể tạo ra điện áp rất cao và những tia sét nhân tạo. This device is still used today in demonstrations và trong một số ứng dụng nghiên cứu. Khi nhìn thấy những tia điện này, nhiều người cảm thấy vừa ấn tượng vừa hơi đáng sợ."
)

DEFAULT_TEXT_V3 = (
    "Xin chào mọi người! [hắng giọng] Như bạn đang nghe thấy đấy, tốc độ xử lý của mình cực kỳ nhanh và mượt mà, giúp phản hồi gần như ngay lập tức theo thời gian thực. Chính vì vậy, mình rất phù hợp để ứng dụng trực tiếp vào các hệ thống Chatbot thông minh, trợ lý ảo, hoặc làm tổng đài viên tự động cho các doanh nghiệp. Tiện lợi quá đúng không ạ? [cười] Hi vọng phiên bản nâng cấp v3 này sẽ mang lại trải nghiệm tuyệt vời cho dự án của bạn."
)

DEFAULT_CONV_SCRIPT = (
    "Phương: Chào mọi người, mình là Phương. Hôm nay team có một announcement cực lớn về VieNeu-TTS Version 2. Đồng hành cùng mình là anh Dũng và Hùng. Hi guys!\n\n"
    "Dũng: Yo, chào cả nhà. Mình sẽ đi thẳng vào technical side của bản nâng cấp này để mọi người có cái nhìn deep hơn nhé.\n\n"
    "Hùng: Chào mọi người. Thật sự V2 là một huge milestone. Nó phá vỡ rào cản của những công cụ đọc văn bản khô khan, hướng tới một sự natural communication đúng nghĩa.\n\n"
    "Phương: Correct! Và bất ngờ nhất là: nãy giờ mọi người đang nghe bản demo được tạo ra 100% bằng VieNeu-TTS V2 đấy. Tụi mình đều là sản phẩm của AI hết. Amazing, right?\n\n"
    "Dũng: Đỉnh thật sự! Tiện đây Hùng share thêm về cái nội công bên trong của model này đi.\n\n"
    "Hùng: Chắc chắn rồi. Model được train trên 10000 hours audio chất lượng cao, nên nó hỗ trợ code-switching Anh Việt cực mượt, tự nhiên như podcast. Đặc biệt, dự án này hoàn toàn open-source để cộng đồng cùng phát triển."
)
