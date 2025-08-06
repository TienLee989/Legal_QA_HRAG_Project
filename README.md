# Legal_QA_HRAG_Project

## Giới thiệu tổng quan

Legal_QA_HRAG_Project là hệ thống hỏi đáp pháp lý tiếng Việt ứng dụng các kỹ thuật hiện đại trong lĩnh vực Truy xuất thông tin (IR) và Học sâu (Deep Learning). Dự án hướng tới việc xây dựng một pipeline tự động, hiệu quả, có khả năng mở rộng, giúp người dùng truy vấn và nhận được các câu trả lời chính xác từ kho dữ liệu pháp luật Việt Nam.

## Mục tiêu dự án

- Tự động hóa quá trình tìm kiếm, truy xuất và xếp hạng các văn bản pháp luật liên quan đến câu hỏi của người dùng.
- Ứng dụng các mô hình embedding, chỉ mục vector và mô hình xếp hạng hiện đại để tối ưu độ chính xác và tốc độ truy xuất.
- Hỗ trợ các nghiệp vụ pháp lý, nghiên cứu, tra cứu luật cho cá nhân, tổ chức, doanh nghiệp, luật sư, sinh viên,...




## Corpus & cấu trúc dữ liệu

### 1. Dữ liệu cho Fine-tune Re-ranking
- **Cấu trúc:**
  - CSV gồm các trường: question, context, label
  - Ví dụ:
    | question                                      | context                                                        | label |
    |-----------------------------------------------|----------------------------------------------------------------|-------|
    | Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?| Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự...   |   1   |
    | Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?| Doanh nghiệp là tổ chức kinh tế có tư cách pháp nhân...        |   0   |
- **Số lượng:**
  - Tổng số cặp: ~20.000 (sau khi nhân mẫu negative)
  - Chia train/test: 90% train (~18.000), 10% test (~2.000)

### 2. Dữ liệu cho Fine-tune Extractive QA
- **Cấu trúc:**
  - CSV gồm các trường: question, context, answer, start_idx, end_idx
  - Ví dụ:
    | question                                      | context                                                        | answer                        | start_idx | end_idx |
    |-----------------------------------------------|----------------------------------------------------------------|-------------------------------|-----------|---------|
    | Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?| Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự...   | Người từ đủ 16 tuổi trở lên   |     0     |   27    |
- **Số lượng:**
  - Tổng số mẫu: ~10.000
  - Chia train/test: 85% train (~8.500), 15% test (~1.500)

### 3. Dữ liệu cho Fine-tune Generative QA
- **Cấu trúc:**
  - CSV gồm các trường: question, context, answer
  - Ví dụ:
    | question                                      | context                                                        | answer                                                        |
    |-----------------------------------------------|----------------------------------------------------------------|---------------------------------------------------------------|
    | Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?| Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự...   | Theo quy định pháp luật, người từ đủ 16 tuổi trở lên...      |
- **Số lượng:**
  - Tổng số mẫu: ~8.000
  - Chia train/test: 80% train (~6.400), 20% test (~1.600)

### Lưu ý:
- Số lượng mẫu thực tế có thể thay đổi tùy theo từng phiên bản dataset, các con số trên là tham khảo từ các notebook và file dữ liệu đi kèm dự án.
- Dữ liệu negative được sinh tự động bằng cách ghép question với context không liên quan.
- Dữ liệu extractive/generative được kiểm tra thủ công hoặc bán tự động để đảm bảo chất lượng nhãn.

### 1. Thiết lập môi trường & cài đặt thư viện
- Cài đặt các thư viện cần thiết (transformers, sentence-transformers, faiss-cpu, rank_bm25, underthesea, accelerate, datasets, evaluate).
- Kiểm tra GPU, thiết lập các thư mục dự án, mount Google Drive (nếu dùng Colab).

### 2. Chuẩn bị & làm sạch dữ liệu
- Đọc dữ liệu gốc (CSV, Parquet), loại bỏ các dòng lỗi, trống, ký tự không hợp lệ.
- Chuẩn hóa các trường dữ liệu (question, context, answer), giới hạn độ dài ngữ cảnh, loại bỏ các dòng không đạt chuẩn.
- Lưu dữ liệu đã làm sạch phục vụ cho các bước tiếp theo.

### 3. Tiền xử lý & lập chỉ mục
- Sinh embedding cho các context bằng SentenceTransformer.
- Lập chỉ mục FAISS (Dense) cho truy xuất ngữ nghĩa.
- Lập chỉ mục BM25 (Sparse) cho truy xuất từ khóa, sử dụng underthesea để token hóa tiếng Việt.
- Lưu context map, chỉ mục FAISS, mô hình BM25.

### 4. GIAI ĐOẠN RETRIEVAL (HYBRID)
- Nhận câu hỏi người dùng, chuẩn hóa và sinh embedding.
- Truy xuất đồng thời bằng BM25 (sparse) và FAISS (dense).
- Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF).
- Chuẩn bị danh sách tài liệu ứng viên cho re-ranking.


### 5. GIAI ĐOẠN RE-RANKING
- Sử dụng Cross-Encoder để đánh giá lại mức độ liên quan giữa câu hỏi và các context truy xuất được.
- Xếp hạng lại các tài liệu dựa trên điểm số re-rank.
- Lựa chọn top-k tài liệu liên quan nhất cho đầu ra cuối cùng.
- **Fine-tune Re-ranking:**
  - Input: Tập dữ liệu gồm các cặp (question, context, label), ví dụ:
    - question: "Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?"
    - context: "Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm..."
    - label: 1 (liên quan) hoặc 0 (không liên quan)
  - Output: Mô hình phân loại nhị phân, đầu ra là xác suất liên quan cho từng cặp (question, context).
  - Đánh giá: accuracy, F1-score trên tập kiểm thử.



### 6. GIAI ĐOẠN EXTRACTIVE READING
- Áp dụng mô hình extractive QA (ví dụ: BERT, PhoBERT) để trích xuất đoạn văn bản trả lời từ các context đã được re-rank.
- Đầu ra là các span (đoạn) trả lời chính xác nhất từ tài liệu pháp lý.
- **Fine-tune Extractive QA:**
  - Input: Tập dữ liệu gồm các cặp (question, context, answer, start_idx, end_idx), ví dụ:
    - question: "Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?"
    - context: "Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm..."
    - answer: "Người từ đủ 16 tuổi trở lên"
    - start_idx: 0, end_idx: 27 (vị trí trong context)
  - Output: Mô hình dự đoán vị trí bắt đầu/kết thúc của answer trong context.
  - Đánh giá: Exact Match, F1-score, độ chính xác vị trí span.


### 7. GIAI ĐOẠN GENERATION / REFINEMENT
- Sử dụng mô hình sinh (generative model, ví dụ: T5, GPT) để tổng hợp, diễn giải hoặc làm mượt câu trả lời cuối cùng.
- Tinh chỉnh đầu ra, loại bỏ trùng lặp, đảm bảo ngữ nghĩa và tính pháp lý.
- **Fine-tune Generative QA:**
  - Input: Tập dữ liệu gồm các cặp (question, context, answer), ví dụ:
    - question: "Độ tuổi chịu trách nhiệm hình sự là bao nhiêu?"
    - context: "Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm..."
    - answer: "Theo quy định pháp luật, người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm, trừ một số trường hợp đặc biệt."
  - Output: Mô hình sinh câu trả lời tự nhiên, đầy đủ, đúng pháp lý.
  - Đánh giá: BLEU, ROUGE, độ mượt và độ chính xác ngữ nghĩa.

### Tóm tắt các ý chính về fine-tune trong pipeline
- **Chuẩn bị dữ liệu:**
  - Dữ liệu phải được làm sạch, chuẩn hóa, gắn nhãn rõ ràng (liên quan/không liên quan, span, câu trả lời tự do).
- **Chọn mô hình phù hợp:**
  - Retrieval: Bi-Encoder, Cross-Encoder (SentenceTransformer, PhoBERT, XLM-R).
  - Extractive QA: BERT, PhoBERT, XLM-R.
  - Generative QA: T5, GPT, các mô hình transformer sinh.
- **Quy trình fine-tune:**
  - Chia tập train/validation, sử dụng Trainer hoặc HuggingFace Transformers.
  - Theo dõi loss, accuracy, F1, EM, BLEU, ROUGE tùy giai đoạn.
  - Lưu checkpoint tốt nhất, đánh giá trên tập kiểm thử.
- **Ứng dụng:**
  - Sử dụng mô hình đã fine-tune cho từng giai đoạn trong pipeline để tối ưu hóa kết quả QA pháp lý.

## Sơ đồ tổng quan RAG Pipeline

```
┌────────────┐
│ 1. Thiết   │
│ lập môi    │
│ trường &   │
│ thư viện   │
└─────┬──────┘
      │
┌─────▼──────┐
│ 2. Chuẩn   │
│ bị & làm   │
│ sạch dữ    │
│ liệu       │
└─────┬──────┘
      │
┌─────▼──────┐
│ 3. Tiền    │
│ xử lý &    │
│ lập chỉ    │
│ mục        │
└─────┬──────┘
      │
┌─────▼──────┐
│ 4. Hybrid  │
│ Retrieval  │
│ (BM25 +    │
│ FAISS +    │
│ RRF)       │
└─────┬──────┘
      │
┌─────▼──────┐
│ 5. Re-     │
│ Ranking    │
│ (Cross-    │
│ Encoder)   │
└─────┬──────┘
      │
┌─────▼──────┐
│ 6. Extract │
│ Reading    │
│ (Span QA)  │
└─────┬──────┘
      │
┌─────▼──────┐
│ 7. Genera- │
│ tion /     │
│ Refinement │
└────────────┘
```

## Flow tổng quan



```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Legal_QA_HRAG_Project RAG Pipeline                    │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────────┤
│   1. Thiết lập│   2. Làm sạch │   3. Lập chỉ  │   4. Hybrid   │   5. Re-    │
│   môi trường │   & chuẩn bị  │   mục dữ liệu │   Retrieval   │   ranking   │
│   & thư viện │   dữ liệu     │   (FAISS,     │   (BM25+FAISS │   (Cross-   │
│              │               │   BM25, Emb)  │   +RRF)       │   Encoder)  │
├───────────────┴───────────────┴───────────────┴───────────────┴─────────────┤
│   6. Fine-tune Re-ranker      │   7. Đánh giá, trực quan hóa & triển khai   │
│   ┌───────────────────────────┴────────────────────────────────────────────┐ │
│   │ - Tạo tập huấn luyện (Q, C, label)                                   │ │
│   │ - Huấn luyện mô hình Re-ranker                                       │ │
│   │ - Đánh giá, trực quan hóa, đóng gói, triển khai, vận hành hệ thống   │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Thư mục dữ liệu & cấu trúc dự án

- `dataset/`: Chứa dữ liệu gốc, dữ liệu đã làm sạch, các bộ luật, văn bản pháp luật.
- `processed_data/`: Dữ liệu đã xử lý, ánh xạ context, file phục vụ cho training và retrieval.
- `embeddings/`: Chỉ mục FAISS, lưu trữ embedding vector cho truy xuất ngữ nghĩa.
- `retrieval_models/`: Mô hình BM25 cho truy xuất từ khóa.
- `trained_models/`: Các mô hình đã huấn luyện (Bi-Encoder, Cross-Encoder, Re-ranker).
- `qa_logs/`, `reranker_logs/`: Log quá trình huấn luyện, đánh giá mô hình.

## Yêu cầu thư viện

Hệ thống sử dụng các thư viện Python phổ biến trong lĩnh vực NLP và IR:

- `transformers`: Xây dựng và fine-tune các mô hình học sâu.
- `underthesea`: Tiền xử lý, tách từ tiếng Việt.
- `sentence-transformers`: Sinh embedding cho văn bản.
- `faiss-cpu`: Lập chỉ mục và truy xuất vector hiệu quả.
- `rank_bm25`: Truy xuất từ khóa với BM25.
- `accelerate`, `datasets`, `evaluate`: Hỗ trợ huấn luyện, đánh giá mô hình.

## Ứng dụng thực tế

- Tra cứu nhanh các quy định pháp luật liên quan đến câu hỏi thực tế.
- Hỗ trợ nghiệp vụ cho luật sư, chuyên viên pháp lý, sinh viên luật.
- Tích hợp vào chatbot, hệ thống tư vấn pháp luật tự động.
- Làm nền tảng cho các nghiên cứu về QA pháp lý, truy xuất thông tin tiếng Việt.

## Hướng dẫn sử dụng cơ bản

1. Chuẩn bị dữ liệu đầu vào theo định dạng chuẩn (CSV, Parquet).
2. Chạy các notebook theo thứ tự: xử lý dữ liệu → lập chỉ mục → truy xuất → fine-tune → đánh giá.
3. Tùy chỉnh tham số, mô hình theo nhu cầu thực tế.
4. Xem log, biểu đồ để đánh giá chất lượng mô hình.

## Tác giả & liên hệ

Dự án phát triển bởi nhóm nghiên cứu AI pháp lý.
Mọi thắc mắc, đóng góp vui lòng liên hệ qua email hoặc github của nhóm.

## Flow tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                  Legal_QA_HRAG_Project Flow                │
├───────────────┬─────────────────────┬──────────────────────┤
│   Bước 1      │   Bước 2            │   Bước 3             │
│ Thiết lập     │ Chuẩn bị &          │ Tiền xử lý dữ liệu,  │
│ môi trường,   │ làm sạch dữ liệu    │ lập chỉ mục FAISS,   │
│ cài đặt thư   │ (lọc, chuẩn hóa,    │ BM25, tạo embedding  │
│ viện          │ giới hạn ngữ cảnh) │                      │
├───────────────┴─────────────────────┴──────────────────────┤
│   Bước 4: Hybrid Retrieval (BM25 + FAISS + RRF)            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 1. Nhận câu hỏi người dùng                         │   │
│   │ 2. Truy xuất BM25 (Sparse)                         │   │
│   │ 3. Truy xuất FAISS (Dense)                         │   │
│   │ 4. Hợp nhất kết quả bằng RRF                       │   │
│   │ 5. Re-rank bằng Cross-Encoder                      │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│   Bước 5: Fine-tune & Re-ranking                           │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 1. Tạo tập huấn luyện (Q, C, label)                │   │
│   │ 2. Huấn luyện mô hình Re-ranker                    │   │
│   │ 3. Trực quan hóa kết quả huấn luyện                │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Thư mục dữ liệu
- `dataset/`: Chứa các file dữ liệu gốc và đã làm sạch
- `processed_data/`: Dữ liệu đã xử lý, ánh xạ context
- `embeddings/`: Chỉ mục FAISS
- `retrieval_models/`: Mô hình BM25
- `trained_models/`: Các mô hình đã huấn luyện (Bi-Encoder, Cross-Encoder, Re-ranker)

## Yêu cầu thư viện
- transformers
- underthesea
- sentence-transformers
- faiss-cpu
- rank_bm25
- accelerate
- datasets
- evaluate

## Tác giả
- Dự án phát triển bởi nhóm nghiên cứu AI pháp lý.
