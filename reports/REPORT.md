# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Dư Văn Sang

Công cụ gán nhãn đã dùng: CVAT

Sao chép file này thành 
eports/REPORT.md rồi điền vào các mục bên dưới. Mọi con số phải truy được
từ 
eports/rounds_table.md, outputs/selection_round1.csv, outputs/metrics_round*.json hoặc
outputs/round*_diff.md. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

**Giải thích:**
- **Lý do chia theo trục thời gian kèm vùng đệm:** Dữ liệu video giao thông quay từ camera cố định có tính tương quan thời gian rất cao (temporal correlation). Hai khung hình liền kề chỉ cách nhau vài phần mười giây hầu như giống hệt nhau về nền đường, điều kiện chiếu sáng và vị trí các xe. Nếu chia theo các đoạn thời gian riêng biệt và đặt một **vùng đệm (buffer zone)** ở giữa (trong bài có 112 frame buffer ngăn cách giữa 268 frame pool và 20 frame test), ta đảm bảo các frame trong tập test cách biệt đáng kể về thời gian so với tập train/pool. Điều này buộc mô hình phải học khả năng khái quát hóa (generalization) trên các tình huống và thời điểm giao thông mới, thay vì ghi nhớ khung cảnh quen thuộc.
- **Chiều hướng lệch nếu chia ngẫu nhiên:** Số đo trên tập kiểm thử (AP50, Precision, Recall) sẽ bị **lệch lạc quan nghiêm trọng (optimistic bias / rò rỉ dữ liệu - data leakage)**. Mô hình chỉ cần ghi nhớ nền đường và vị trí tương đối của xe ở frame huấn luyện $ là có thể dễ dàng đoán đúng frame kiểm thử  + 0.4s$, tạo ra kết quả kiểm thử cao giả tạo mà không phản ánh đúng năng lực thực tế khi triển khai.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ 
ounds_table.md. Dựa vào outputs/compare_round0.jpg, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

**Dòng vòng 0 từ 
ounds_table.md:**

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

**Phân tích dựa trên số liệu và ảnh outputs/compare_round0.jpg:**
- **Các loại xe không khớp nhãn tham chiếu:**
  1. *Xe kích thước nhỏ ở khoảng cách xa (FN - khung vàng):* Bị bỏ sót nghiêm trọng (chỉ đạt recall 0.182, tức bỏ sót 54/66 box xe nhỏ). Trong điều kiện ban đêm, xe ở xa chỉ hiện lên dưới dạng hai đốm sáng mờ nhạt hoặc khối đen chìm vào nền đường, khiến mô hình COCO pretrained không nhận diện được.
  2. *Xe ở mép ảnh hoặc bị che khuất một phần (occlusion):* Mô hình dễ bỏ qua các xe chỉ nhìn thấy đuôi hoặc nửa thân xe.
  3. *Dự đoán thừa (FP - 16 khung đỏ):* Mô hình bị nhầm lẫn bởi các vệt đèn phản chiếu loang trên mặt đường, ánh đèn đường hoặc biển báo sáng lóa.
- **Độ phủ (recall) theo kích thước xe:** Thể hiện sự chênh lệch rất lớn giữa các nhóm kích thước: xe nhỏ đạt recall rất thấp là 0.182 (18.2%), trong khi xe vừa đạt 0.547 (54.7%) và xe lớn đạt 0.561 (56.1%) — gấp 3 lần xe nhỏ. Điều này chứng minh mô hình pretrained COCO chỉ phát hiện tương đối tốt các xe có đường nét rõ ràng ở cự ly gần và trung bình, nhưng hầu như bất lực trước xe nhỏ ở xa trong môi trường ánh sáng yếu ban đêm.
- **Trường hợp cần rà lại nhãn tham chiếu trước khi kết luận mô hình sai:** Do nhãn tham chiếu của tập test được sinh tự động bởi mô hình và chưa qua rà soát thủ công 100%, có trường hợp mô hình thực sự phát hiện đúng một chiếc xe thật ở xa trong bóng tối nhưng nhãn tham chiếu lại bỏ sót (khiến mô hình bị phạt FP oan); hoặc nhãn tham chiếu vô tình khoanh một vệt đèn phản quang khiến mô hình không đoán vào đó lại bị tính là FN. Người phát triển cần quan sát trực tiếp ảnh gốc để kiểm tra tính đúng đắn của nhãn tham chiếu trước khi kết luận mô hình sai.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức score = W_U·U + W_A·A + W_D·D và vai trò của MIN_GAP_S.
Dẫn ba frame trong 
eports/SELECTION.md và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

**Giải thích công thức và vai trò của MIN_GAP_S:**
- **Công thức score = W_U·U + W_A·A + W_D·D** (với trọng số mặc định  = 0.5$,  = 0.3$,  = 0.2$):
  - $ (Uncertainty - Độ bất định): Đo mức độ lưỡng lự của mô hình đối với 5 box khó nhất trong frame. Với mỗi box có điểm tin cậy $, độ bất định tính bằng (c) = 1 - |2c - 1|$. Khi  = 0.5$, (c) = 1$ (mô hình phân vân nhất giữa xe và nền); khi  \to 0$ hoặc  \to 1$, (c) \to 0$. $ là trung bình 5 giá trị $ cao nhất. Trọng số 0.5 thể hiện ưu tiên cao nhất cho ảnh chứa box khó.
  - $ (Ambiguity - Mật độ box mơ hồ): Tỷ lệ số box có .15 \le c < 0.50$ so với số box mơ hồ lớn nhất của một frame trong pool. Đại diện cho độ phức tạp của khung hình có nhiều mục tiêu chưa dám khẳng định. Trọng số 0.3.
  - $ (Diversity - Khoảng cách thời gian): Khoảng cách thời gian từ frame đang xét tới frame đã được gán nhãn gần nhất, chia cho ngưỡng chặn .0s$ ( \le 1.0$). Càng xa frame đã học thì $ càng cao, giúp phân bổ mẫu rải đều theo thời gian. Trọng số 0.2.
- **Vai trò của MIN_GAP_S (2.0 giây):** Là khoảng cách thời gian tối thiểu giữa hai frame bất kỳ được chọn trong cùng một đợt (batch). Vì video quay từ góc máy cố định, hai frame cách nhau dưới 2 giây có thông tin xe cộ và quang cảnh gần như trùng lặp nhau. MIN_GAP_S buộc giải thuật tham lam bỏ qua các ảnh gần trùng, tránh bắt người gán nhãn phải làm việc thừa thãi trong khi mô hình không học thêm được bao nhiêu tri thức mới.

**Minh chứng qua các frame (từ 
eports/SELECTION.md và file kết quả):**
1. rame_0031.jpg ( = 12.4s$, Hạng 1 trong selection_round2.csv, Score = 0.7793,  = 0.7386$,  = 0.7$,  = 1.0$, 9 box, 7 mơ hồ): Được chọn đầu tiên vì $ cao, nhiều box mơ hồ và  = 1.0$ (cách xa mọi ảnh đã gán ở vòng 1).
2. rame_0074.jpg ( = 29.6s$, Hạng 2, Score = 0.7776,  = 0.7953$,  = 0.6$,  = 1.0$, 8 box, 6 mơ hồ): Được chọn do độ bất định $ xấp xỉ 0.8, nằm ở đoạn thời gian hoàn toàn mới chưa từng học.
3. rame_0388.jpg ( = 155.2s$, Hạng 3, Score = 0.7412 trong CSV,  = 0.8785$,  = 0.9$,  = 0.16$, 11 box, 9 mơ hồ): Minh chứng xuất sắc cho việc xử lý ảnh gần trùng. Mặc dù $ cao kỷ lục (0.8785) và có tới 9 box mơ hồ, nhưng vì  = 155.2s$ chỉ cách các frame đã gán ở vòng 1 (rame_0380 và rame_0392) lần lượt 3.2s và 1.6s, điểm $ bị phạt nặng xuống 0.16, kéo tụt điểm tổng để cảnh báo nguy cơ lãng phí công sức do trùng lặp bối cảnh.
4. rame_0030.jpg ( = 12.0s$, Hạng 13, Score = 0.7037) hoặc rame_0002.jpg ( = 0.8s$, Hạng 6, Score = 0.7288): Ví dụ rame_0030.jpg có điểm số cao nhưng chỉ cách rame_0031.jpg ( = 12.4s$) đúng 0.4 giây. Khi thuật toán chọn rame_0031.jpg, rame_0030.jpg lập tức bị gạt bỏ do vi phạm MIN_GAP_S = 2.0s nhằm tiết kiệm công gán nhãn.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Vì sao?**
**Không.** Điểm bất định cao chỉ phản ánh rằng mô hình *hiện tại* đang lúng túng hoặc thiếu tự tin trước các đặc trưng trong frame đó (có thể do nhiễu hạt ban đêm, chùm đèn xe phản chiếu phức tạp, hoặc vật thể lạ ngoài phân phối). Điểm bất định cao không đảm bảo rằng khi gán nhãn xong thì mô hình sẽ tổng quát hóa tốt hơn trên tập kiểm thử độc lập. Bằng chứng thực tế ở vòng 1 cho thấy: dù chọn các ảnh có bất định cao nhất, AP50 sau khi train vẫn giảm từ 0.771 xuống 0.504 do mô hình bị overfit và co hẹp độ phủ nhận diện.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ 
ounds_table.md. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  outputs/round*_diff.md);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh compare_round*.jpg, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng BLIND_SCAN.md, REVIEW_LOG.csv và 
ound1_diff.md phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

**Bảng so sánh các vòng từ 
ounds_table.md:**

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 317 | 0.504 | -0.268 | 1.000 | 0.166 | 0.285 | 0.000 | 0.155 | 0.512 |

**Trình bày chi tiết Vòng 1:**
- **Mức độ sửa nhãn gợi ý (từ outputs/round1_diff.md):**
  - Tập huấn luyện gồm 12 ảnh. Model đề xuất 169 box pre-label, sau khi can thiệp thủ công tăng lên 317 box (tăng 87.6%).
  - Giữ nguyên (accepted): 153 box (tỷ lệ chấp nhận đạt 91%).
  - Chỉnh sửa biên (edited): 8 box (nắn chỉnh cạnh ôm khít thân xe).
  - Xóa bỏ (deleted - FP của model đề xuất): 8 box (loại bỏ các box khoanh nhầm vệt đèn loang trên mặt đường).
  - Thêm mới (added - FN của model đề xuất): 156 box (bổ sung số lượng xe thực tế bị AI bỏ sót, chủ yếu là xe ở xa và xe khuất mép).
- **Biến động AP50:** AP50 giảm mạnh từ 0.771 xuống 0.504 (giảm -0.268, tức giảm 26.8% so với cold start).
- **Nhóm xe tốt lên / xấu đi trên cùng tập test:**
  - *Tốt lên:* Precision@0.25 đạt mức tuyệt đối 1.000 (tăng từ 0.925), hoàn toàn không còn bất kỳ dự đoán báo động giả (FP = 0) nào trên toàn bộ 20 ảnh kiểm thử.
  - *Xấu đi:* Recall@0.25 tụt nghiêm trọng từ 0.489 xuống 0.166 (tổng số xe bắt trúng TP giảm từ 197 xuống chỉ còn 67, số FN tăng từ 206 lên 336). Trong đó:
    * Xe nhỏ (R small): tụt từ 0.182 về đúng 0.000 (mất hoàn toàn khả năng phát hiện xe nhỏ).
    * Xe vừa (R medium): tụt từ 0.547 xuống 0.155 (giảm hơn 3.5 lần).
    * Xe lớn (R large): giảm nhẹ từ 0.561 xuống 0.512.

**Phân tích ảnh compare_round1.jpg và các ca cụ thể:**
- **Ca kết quả thay đổi sau fine-tune:** Trên ảnh compare_round1.jpg, tại cột mô hình vòng 1, ta thấy mô hình đã loại bỏ triệt để các box đỏ (FP = 0). Tuy nhiên, trên các làn đường phía xa, hàng loạt khung màu vàng (FN) xuất hiện dày đặc thay cho các khung xanh lá trước kia ở cold start. Lý do có thể kiểm chứng: Huấn luyện 50 epoch trên tập dữ liệu quá nhỏ (12 ảnh) khiến mô hình bị co cụm phân phối (overfitting), tự động đẩy ngưỡng tin cậy nội tại lên cao; mô hình trở nên 'quá thận trọng', chỉ dám bắt những xe cực lớn ở cự ly gần và bỏ qua toàn bộ xe nhỏ/xa.
- **Phân biệt ba công cụ:**
  - BLIND_SCAN.md: Là quan sát độc lập bằng mắt thường của người gán nhãn trước khi nhìn thấy gợi ý pre-label (ví dụ quan sát rame_0099.jpg thấy 21 xe, dự đoán AI sẽ sót xe ở góc mép và các xe nhỏ mờ ở xa).
  - REVIEW_LOG.csv: Là nhật ký ghi vết hành động can thiệp thực tế của người gán nhãn lên pre-label (ví dụ: ở rame_0099.jpg thêm box xe bị cắt mép, ở rame_0107.jpg xóa khung vệt đèn phản quang, ở rame_0182.jpg chỉnh lại biên ôm sát đèn xe).
  - 
ound1_diff.md: Là bản đối chiếu tự động định lượng giữa pre-label và nhãn hoàn thiện, tổng kết tỷ lệ chấp nhận và số box accepted/edited/deleted/added.
- **Mô tả ca khó theo guideline:** Ca xe di chuyển ở sát mép khung hình bị cắt một phần thân (như tại rame_0099.jpg). Theo guideline, người gán nhãn chỉ được khoanh phần thân xe còn nhìn thấy được trong ảnh, không được phỏng đoán phần khuất ngoài biên. Điều này tạo ra các bounding box có tỷ lệ kích thước dị biệt so với hình khối ô tô thông thường, khiến bộ tiền gán nhãn của AI rất dễ bỏ sót hoặc ước lượng sai biên.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

**Kết luận và quyết định vòng lặp:**
- **So sánh với cold start:** Vòng 1 đem lại sự cải thiện tuyệt đối về độ chính xác xác thực (Precision đạt 100%, FP = 0), nhưng phải trả giá bằng việc sụt giảm nghiêm trọng độ phủ (Recall giảm sâu từ 0.489 xuống 0.166, AP50 giảm 0.268), đặc biệt là mất khả năng phát hiện xe nhỏ.
- **Lý do tiếp tục vòng sau:** Không được dừng lại ở vòng 1. Mô hình hiện tại mới chỉ học trên 12 ảnh với 317 box — lượng dữ liệu còn quá ít khiến mô hình bị co cụm phân phối và quá thận trọng. Cần tiếp tục nạp lô 12 ảnh của Vòng 2 để đa dạng hóa mẫu và phục hồi độ phủ (recall) cho xe vừa và nhỏ.

**Đề xuất hai ca còn yếu / bất định cho vòng sau (từ selection_round2.csv):**
1. rame_0031.jpg ( = 12.4s$, Hạng 1, Score = 0.7793, 7 box mơ hồ,  = 1.0$): Chi phí rà khoảng 9-15 box. Nguy cơ gần trùng bằng 0 vì cách xa các ảnh vòng 1 (>25s), giúp mô hình học phân cảnh đêm ở giai đoạn đầu video.
2. rame_0001.jpg ( = 0.4s$, Hạng 4, Score = 0.7390, 7 box mơ hồ,  = 1.0$): Chi phí rà khoảng 11-20 box. Nguy cơ gần trùng bằng 0, đại diện cho những giây đầu tiên của video với góc nhìn xe cộ gần dải phân cách.

**Ảnh hưởng của các giới hạn bài toán:**
- **Tập test chỉ có 20 ảnh:** Quy mô mẫu nhỏ khiến phương sai thống kê cao. Sai số trên 1–2 ảnh có thể làm biến động mạnh chỉ số AP50 toàn tập.
- **Luật bỏ qua xe quá nhỏ (< 16 px):** Dù giúp tránh phạt mô hình trước các đốm sáng quá mờ, nhưng cũng che khuất năng lực thực sự của bộ phát hiện trên các phương tiện từ khoảng cách rất xa.
- **Nhãn tham chiếu do mô hình tạo chưa rà thủ công:** Nhãn test chưa phải chân lý tuyệt đối (ground truth có nhiễu). Một số dự đoán đúng của mô hình có thể bị tính nhầm là FP, hoặc mô hình bỏ qua nhãn nhiễu lại bị tính là FN, dẫn đến sai lệch trong việc đánh giá thực chất.

**Các điểm cần kiểm tra khi AP50 giảm trước khi train thêm:**
1. *Kiểm tra chất lượng nhãn huấn luyện:* Rà soát các file trong labels/round1/ xem có box nào bị gán sai nhãn, tọa độ vượt biên hoặc sót xe nhỏ hay không.
2. *Kiểm tra chiến lược huấn luyện và siêu tham số:* Xem xét giảm Learning Rate hoặc đóng băng các tầng trích xuất đặc trưng (freeze backbone) để tránh hiện tượng quên thảm khốc (Catastrophic Forgetting) tri thức pretrained COCO; điều chỉnh số epoch phù hợp thay vì ép quá mức trên 12 ảnh.
3. *Kiểm tra phân phối kích thước bounding box:* So sánh tỷ lệ phân bố giữa tập train và tập test để phát hiện lệch pha dữ liệu.
4. *Kiểm tra ngưỡng Confidence và NMS IoU threshold:* Đảm bảo ngưỡng lọc box không vô tình triệt tiêu các dự đoán có độ tự tin vừa phải của xe nhỏ.
