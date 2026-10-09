# Deep Research Agent (Deep Agents + Sandbox)

Lab dựng một **hệ thống deep research đa tác tử**: người dùng chỉ cần nhập một chủ đề (ví dụ `survey about world model`), hệ thống tự lập kế hoạch, giao việc cho nhiều subagent, tìm tài liệu trên arXiv, Hugging Face và web, rồi viết một **báo cáo có trích dẫn**.

Hình thức: **bài thực hành cá nhân**. Ngôn ngữ lập trình: Python 3.11 trở lên.

## Bài nộp - Trần Nhật Minh (2A202602483)

### Cài đặt và chạy

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                  # điền khóa của bạn (xem bảng cấu hình bên dưới)
python research.py "survey about world model"
python self_check.py                  # kiểm tra trước khi nộp, không tốn token
python -m pytest -q tests             # unit test (pip install pytest)
```

Cấu hình đã dùng để sinh `reports/` (khóa nằm trong `.env`, **không** commit):

| Thành phần | Giá trị |
|---|---|
| LLM | NVIDIA NIM (endpoint tương thích OpenAI): `LAB_BASE_URL=https://integrate.api.nvidia.com/v1`, `LAB_MODEL=nvidia/nemotron-3-super-120b-a12b`, `LAB_API_KEY=nvapi-...` |
| Sandbox | `SANDBOX=docker` (container `python:3.12-slim`, `--network none`) |
| Web | Exa MCP với `EXA_API_KEY` |

Cũng chạy được với Gemini (`LAB_MODEL=google_genai:<model>` + `GOOGLE_API_KEY`; gói `langchain-google-genai` đã có trong `requirements.txt`) hoặc Daytona (`DAYTONA_API_KEY`, bỏ `SANDBOX=docker`).

### Đọc thư mục `reports/`

Mỗi chủ đề có ba tệp, tên là slug của chủ đề:

- `<slug>.md`: báo cáo (TL;DR, Background, các phần theo chủ đề, Trends and open problems, References). Mọi `[n]` trỏ tới dòng `[n]` trong References.
- `<slug>.sources.json`: danh sách nguồn `{n, id, url, title, date, source}`; `source` là công cụ đã tìm ra nguồn (`arxiv`, `hf-daily`, `hf-search`, `web`).
- `<slug>.meta.json`: số liệu lần chạy: `model`, `elapsed_s`, `subagent_calls` (số lần lead gọi `task`), `tool_calls`, `tokens` (chỉ tin nhắn của lead), `n_sources`, `source_families`.

Kiểm tra một báo cáo: `python check_citations.py reports/<slug>.md reports/<slug>.sources.json`.

### Kết quả 5 chủ đề

Model `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM), sandbox Docker. `python self_check.py`: **READY to submit**; `check_citations.py` in `OK` cho cả 5 báo cáo.

| Chủ đề | Báo cáo | Nguồn | Họ nguồn | `subagent_calls` | Phút |
|---|---|---|---|---|---|
| survey about world model | [survey-about-world-model.md](reports/survey-about-world-model.md) | 16 | arxiv, hf-daily, hf-search, web | 4 | 20 |
| survey about reinforcement learning for LLM reasoning | [survey-about-reinforcement-learning-for-llm-reasoning.md](reports/survey-about-reinforcement-learning-for-llm-reasoning.md) | 19 | arxiv, hf-daily, hf-search, web | 10 | 52 |
| survey about LLM agents and tool use | [survey-about-llm-agents-and-tool-use.md](reports/survey-about-llm-agents-and-tool-use.md) | 15 | arxiv, hf-daily, hf-search, web | 4 | 10 |
| survey about video and multimodal generation | [survey-about-video-and-multimodal-generation.md](reports/survey-about-video-and-multimodal-generation.md) | 26 | arxiv, hf-daily, hf-search, web | 5 | 15 |
| survey about efficient inference and small language models | [survey-about-efficient-inference-and-small-language-models.md](reports/survey-about-efficient-inference-and-small-language-models.md) | 15 | arxiv, hf-search, web | 4 | 9 |

Ghi chú từ quá trình chạy:

- Gemini (`gemini-3.8-flash`) liên tục trả `503 high demand` nên chuyển sang NVIDIA NIM; từ đó thêm `ModelRetryMiddleware` cho lỗi tạm thời.
- Chạy không có `EXA_API_KEY` thì `web_fetch`/`web_search` bị giới hạn tốc độ liên tục (bước citation-checker kéo dài hàng chục phút); có khóa thì mỗi lời gọi khoảng 1-2 giây.
- Lần chạy đầu của chủ đề world model thiếu mục `## Trends and open problems` và mô hình tưởng "hai năm gần nhất" là 2023-2024: prompt được bổ sung ngày hiện tại, tên mục bắt buộc và bước `grep '^## '` tự kiểm tra, rồi chạy lại chủ đề đó.
- `tokens` trong `meta.json` chỉ đếm tin nhắn của lead; chi phí thật (gồm subagent) cao hơn.

### Những điểm đáng chú ý trong cài đặt

- `tools.py`: `with_retry` (backoff lũy thừa + jitter, tôn trọng `Retry-After`, chặn `cap`, không ngủ sau lần cuối); arXiv giữ khoảng cách 3 giây bằng khóa luồng (các researcher chạy song song) và làm sạch truy vấn; Exa free tier báo giới hạn tốc độ bằng HTTP 200 + cờ `result._meta["ai.exa/rateLimited"]` nên công cụ phát hiện cờ này để retry; khóa Exa bị che khỏi mọi chuỗi `ERROR`.
- `agents.py`: lead lập kế hoạch bằng `write_todos`, giao 3-5 researcher song song, kiểm tra notes, gộp `sources.json`, kiểm tra đủ 3 họ nguồn, chạy finalizer + validator trong sandbox, rồi nhờ `citation-checker` kiểm tra mẫu. Giới hạn `ModelCallLimitMiddleware`/`ToolCallLimitMiddleware` cho lead và từng subagent (kể cả `general-purpose` được thay bằng bản có giới hạn), `recursion_limit=1000`, và `ModelRetryMiddleware` chỉ retry lỗi tạm thời của nhà cung cấp (503 "high demand", 429, timeout).
- `research.py`: sandbox luôn được dọn bằng `open_sandbox`; chạy hỏng thì in `FAILED`, thoát mã 1 và không ghi tệp nào; in một dòng log cho mỗi lần gọi công cụ ra stderr để theo dõi.
- `tests/`: unit test cho `with_retry`, làm sạch truy vấn arXiv, che khóa Exa, `check_citations`, `slugify`, `summarize`, `save_outputs`.

## 1. Mục tiêu học tập

Sau lab, bạn có thể:

1. Dựng agent bằng thư viện Deep Agents (LangChain): công cụ (tool), system prompt, subagent, backend.
2. Dùng **sandbox** (Daytona) làm không gian làm việc và nơi chạy mã cho agent; hiểu vì sao khóa API và công cụ mạng phải nằm ở phía host chứ không nằm trong sandbox.
3. Viết công cụ gọi API ngoài **chịu được giới hạn tốc độ** (retry, backoff, jitter, `Retry-After`).
4. Thiết kế quy trình đa tác tử: lead chia nhỏ câu hỏi, giao cho N researcher chạy song song, tổng hợp và kiểm tra trích dẫn.
5. Tạo báo cáo có thể kiểm chứng: mọi khẳng định có `[n]` trỏ tới một nguồn có thật.

## 2. Hệ thống làm gì

```mermaid
flowchart TD
    U["python research.py &quot;survey about world model&quot;"] --> S["open_sandbox() - Daytona"]
    S --> L["Lead agent: write_todos, chia N câu hỏi con"]
    L -->|task x N, song song| R["researcher subagents"]
    R --> T1["arxiv_search"]
    R --> T2["hf_daily_papers / hf_search_papers"]
    R --> T3["web_search / web_fetch (Exa MCP)"]
    R --> N["ghi chú trong sandbox: /tmp/work/research/notes"]
    N --> M["Lead gộp: sources.json + report.md"]
    M --> F["execute: finalize_citations.py (có sẵn)"]
    F --> V["execute: check_citations.py"]
    V --> C["citation-checker subagent kiểm tra mẫu"]
    C --> D["download -> reports/slug.md, .sources.json, .meta.json"]
```

Nguồn dữ liệu:

| Nguồn | Dùng để |
|---|---|
| arXiv API `https://export.arxiv.org/api/query` | Tìm bài theo từ khóa, sắp theo ngày |
| Hugging Face Daily Papers `/api/daily_papers` | Bài đang "trending": upvotes, githubRepo, summary |
| Hugging Face papers search `/api/papers/search?q=` | Tìm bài theo chủ đề |
| Web qua Exa MCP (`web_search_exa`, `web_fetch_exa`) | Blog, survey, trang dự án, nội dung đầy đủ của một URL |

## 3. Cấu trúc thư mục

```
Lab/
├── README.md  GUIDE.md  RUBRIC.md  REPORT_TEMPLATE.md   tài liệu
├── topics.md                 5 chủ đề cần chạy
├── requirements.txt  .env.example  .gitignore
├── model.py                  CÓ SẴN - không sửa: tạo mô hình LLM từ biến môi trường
├── sandbox.py                CÓ SẴN - không sửa: sandbox Daytona (hoặc Docker), upload, download
├── self_check.py             CÓ SẴN - không sửa: tự kiểm tra trước khi nộp (python self_check.py)
├── finalize_citations.py     CÓ SẴN - không sửa: script chạy trong sandbox, tự sinh `## References` và đánh số lại trích dẫn
├── tools.py                  SINH VIÊN CÀI ĐẶT: retry + 5 công cụ nguồn dữ liệu
├── agents.py                 SINH VIÊN CÀI ĐẶT: prompt, subagent, lead agent
├── research.py               SINH VIÊN CÀI ĐẶT: script chính
├── check_citations.py        SINH VIÊN CÀI ĐẶT: kiểm tra trích dẫn, chạy TRONG sandbox
└── reports/                  báo cáo sinh ra (bạn commit vào repo nộp)
```

Mỗi tệp "SINH VIÊN CÀI ĐẶT" là **pseudo-code chạy được** (import được): các hàm có docstring mô tả việc cần làm, các `TODO n` đánh số theo `GUIDE.md`, thân hàm đang `raise NotImplementedError`.

## 4. Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                     # rồi điền khóa CỦA BẠN
```

Bạn cần ba loại khóa (điền vào `.env`, **không bao giờ commit** `.env`):

| Khóa | Lấy ở đâu | Ghi chú |
|---|---|---|
| LLM (`LAB_MODEL` + khóa nhà cung cấp) | Nhà cung cấp bạn chọn (OpenAI, Anthropic, Google, OpenRouter, Ollama...) | Mô hình **phải hỗ trợ tool calling**. Chép tên mô hình từ tài liệu của nhà cung cấp. |
| `DAYTONA_API_KEY` | https://app.daytona.io | Kiểm tra gói miễn phí / credit hiện hành. Không có tài khoản hoặc hết credit: đặt `SANDBOX=docker` để chạy sandbox trong container Docker cục bộ (xem `.env.example`). |
| `EXA_API_KEY` (khuyến nghị) | https://dashboard.exa.ai/api-keys | Có thể chạy không khóa, nhưng bản miễn phí của MCP bị giới hạn tốc độ rất nhanh. |

## 5. Làm bài

Làm theo thứ tự (chi tiết trong `GUIDE.md`):

1. `check_citations.py`: khởi động nhẹ, thuần Python.
2. `tools.py`: viết `with_retry` và 5 công cụ. Thử riêng từng công cụ: `python tools.py`.
3. `agents.py`: viết prompt, subagent và lead agent.
4. `research.py`: ghép tất cả; chạy một chủ đề:

```bash
python research.py "survey about world model"
```

Kết quả nằm ở `reports/survey-about-world-model.md` cùng `.sources.json` và `.meta.json`.

## 6. Chủ đề và nộp bài

- Chạy đủ **5 chủ đề** trong [`topics.md`](topics.md), mỗi chủ đề một lần.
- Commit mã nguồn và toàn bộ `reports/`, đẩy lên một **public repo** GitHub và nộp link.
- Kiểm tra trước khi nộp: chạy **`python self_check.py`** (không tốn token): nó kiểm tra đủ 5 báo cáo, `meta.json`, trích dẫn bằng `check_citations.py` của bạn, và không có `.env`/khóa nào trong git.
- Cách chấm: xem [`RUBRIC.md`](RUBRIC.md).

## 7. Thời gian, chi phí và an toàn

- Dùng một mô hình **rẻ nhưng hỗ trợ tool calling**, và **đặt giới hạn** (số lần gọi mô hình/công cụ cho lead và subagent, `recursion_limit`): một prompt hỏng có thể khiến agent lặp rất lâu. Đây là hạng mục 2.5 của `RUBRIC.md`.
- Kết quả có tính ngẫu nhiên: cùng một mã có thể cho báo cáo hợp lệ ở lần này và trích dẫn lỗi ở lần sau. Hãy sửa **prompt và mã**, không sửa tay báo cáo.

- Mỗi lần chạy tốn token LLM và thời gian sandbox. `tokens` trong `meta.json` chỉ đếm tin nhắn của lead, chưa gồm subagent, nên chi phí thật cao hơn. `open_sandbox()` luôn dừng và xóa sandbox khi kết thúc, kể cả khi lỗi. Đừng bỏ qua nó.
- **Không đưa bí mật vào sandbox.** Sandbox không ngăn được prompt injection hay việc đẩy dữ liệu ra mạng; một trang web độc hại có thể khiến agent chạy lệnh bên trong sandbox. Vì vậy mọi công cụ gọi mạng và mọi khóa ở lại phía host.
- Nội dung lấy từ web là **dữ liệu không đáng tin**: agent không được làm theo chỉ dẫn nằm trong đó.
