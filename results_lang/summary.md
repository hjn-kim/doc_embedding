# 임베딩 모델 비교 결과

- 색인 [512]: 후보 264개
- 색인 [full]: 후보 51개
- 질문 수: 150  (정답@k 는 이 중 몇 개를 맞혔는지)
- scope: all
- 청킹: 510token / overlap 100 / 기준 토크나이저 BAAI/bge-m3
- chunks_per_s: 워밍업 후 최소 512개 텍스트를 3회 인코딩한 최고 기록

| model                                | 색인   | hf_id                                   |   dim |   후보수 |   max_seq |   Hit@1 |   Hit@3 |   MRR@3 |   nDCG@3 |   정답@1 |   정답@3 |   VRAM_MB |   index_MB |   chunks_per_s |   query_ms |
|:-------------------------------------|:-------|:----------------------------------------|------:|---------:|----------:|--------:|--------:|--------:|---------:|---------:|---------:|----------:|-----------:|---------------:|-----------:|
| bge-m3 [512] [hybrid(dense+sparse)]  | 512    | BAAI/bge-m3                             |  1024 |      264 |       512 |  0.8667 |  0.9933 |  0.9256 |   0.8992 |      130 |      149 |      1273 |       1.03 |          157.3 |       1.07 |
| kure-v1 [512]                        | 512    | nlpai-lab/KURE-v1                       |  1024 |      264 |       512 |  0.84   |  0.98   |  0.9044 |   0.89   |      126 |      147 |      1287 |       1.03 |          149.7 |       1.48 |
| bge-m3 [512]                         | 512    | BAAI/bge-m3                             |  1024 |      264 |       512 |  0.8133 |  0.9733 |  0.8878 |   0.8653 |      122 |      146 |      1273 |       1.03 |          157.3 |       1.07 |
| e5-small-ko [512]                    | 512    | dragonkue/multilingual-e5-small-ko      |   384 |      264 |       512 |  0.8    |  0.9467 |  0.8689 |   0.8522 |      120 |      142 |       313 |       0.39 |          559.1 |       1.13 |
| harrier-270m [512]                   | 512    | microsoft/harrier-oss-v1-270m           |   640 |      264 |       512 |  0.7067 |  0.9333 |  0.8133 |   0.7949 |      106 |      140 |       675 |       0.64 |          184.7 |       2.29 |
| harrier-0.6b [512]                   | 512    | microsoft/harrier-oss-v1-0.6b           |  1024 |      264 |       512 |  0.7267 |  0.92   |  0.8167 |   0.7861 |      109 |      138 |      1382 |       1.03 |           64.9 |       2.95 |
| e5-large-instruct [512]              | 512    | intfloat/multilingual-e5-large-instruct |  1024 |      264 |       512 |  0.7067 |  0.8867 |  0.7911 |   0.7703 |      106 |      133 |      1273 |       1.03 |          146.6 |       1.47 |
| harrier-0.6b [full]                  | full   | microsoft/harrier-oss-v1-0.6b           |  1024 |       51 |      8192 |  0.94   |  0.9933 |  0.9622 |   0.9702 |      141 |      149 |      2440 |       0.2  |           10.3 |       8.05 |
| bge-m3 [full] [hybrid(dense+sparse)] | full   | BAAI/bge-m3                             |  1024 |       51 |      8192 |  0.94   |  0.98   |  0.9578 |   0.9635 |      141 |      147 |      1536 |       0.2  |           27.6 |       3.44 |
| kure-v1 [full]                       | full   | nlpai-lab/KURE-v1                       |  1024 |       51 |      8192 |  0.92   |  0.9667 |  0.9433 |   0.9494 |      138 |      145 |      1565 |       0.2  |           30   |       3.88 |
| bge-m3 [full]                        | full   | BAAI/bge-m3                             |  1024 |       51 |      8192 |  0.9    |  0.9667 |  0.93   |   0.9394 |      135 |      145 |      1536 |       0.2  |           27.6 |       3.44 |
| harrier-270m [full]                  | full   | microsoft/harrier-oss-v1-270m           |   640 |       51 |      8192 |  0.8933 |  0.9533 |  0.92   |   0.9286 |      134 |      143 |      1360 |       0.12 |           31.6 |       7.31 |

- 모든 지표는 **그 행의 색인 단위에서** 잰 값이다. `512` 행은 청크 랭킹, `full` 행은 문서 랭킹 기준이다.
- 후보 수가 다르므로(512: 청크 274개 / full: 문서 44개) **512 행끼리, full 행끼리만 비교할 것.** 무작위로 찍었을 때의 기준선부터 다르다 (1/274 vs 1/44).
- **Hit@k**  상위 k개 안에 정답이 하나라도 있던 질문 비율. k 가 커지면 절대 낮아지지 않는다 (Hit@1 <= Hit@3)
- **정답@k**  같은 값을 비율이 아니라 실제 맞힌 문제 개수로 센 것
- **MRR@k**  첫 정답의 등수 역수 (1등=1.0, 2등=0.5, 3등=0.33, 밖=0). '찾았나'와 '얼마나 위에 올렸나'를 한 숫자로 묶은 것
- **nDCG@k**  정답을 얼마나 위쪽에 몰아놨는지. 정답이 여러 개일 때도 제대로 계산된다 (최종 정렬 기준)
- **VRAM_MB**  모델 로드 + 인코딩 중 최대 GPU 메모리 사용량
- **index_MB**  청크 벡터를 전부 담은 인덱스 용량 (차원에 비례)
- **chunks_per_s / query_ms**  워밍업 후 측정한 인코딩 처리량 / 질문 1개당 지연

---

# 틀린 문제 목록  (질문 150개 기준)

## 모든 모델이 @1 에서 틀린 문제 — 0개

없음

## 모든 모델이 @3 에서 틀린 문제 — 0개

없음

---

## kure-v1 [512]

### 틀린@1 — 24/150개

- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `en-08`  Find the context of the waiver addressed in Hunter, unlike the Cedric Ray Jones petition, and the type of waiver this petition concerns
- `en-13`  Find what Judge McLeese, concurring only in the judgment, said the majority opinion had given to Carter's race
- `ch-09`  请找出凉城县人民检察院对张某某作出的最终处理决定
- `ch-20`  请找出四川省川谷坊酒业有限公司向不特定多数人集资时承诺的月利率
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `vn-17`  Hãy tìm hai loại hình phạt cùng mức tối thiểu và tối đa được quy định tại khoản 1 Điều 173 BLHS áp dụng đối với Trần Xuân H
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-16`  Jinoyat-Ijroiya kodeksini qo‘llashda voyaga yetmaganlar koloniyalari kattalar muassasalariga qaraganda yaxshiroq natija berganining sababi sifatida ko‘rsatilgan miqyosga oid xususiyatni toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-09`  Найдите документ, который, по разъяснению прокуратуры Центрального района города Оренбурга, выдают при обращении с заявлением в полицию
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-16`  Найдите наказание, назначенное Емельянову Даниилу Игоревичу Бежецким межрайонным судом Тверской области
- `ru-19`  Найдите решение, исключённое апелляцией Тверского областного суда из приговора суда первой инстанции по делу Емельянова Д.И.
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 3/150개

- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay

## kure-v1 [full]

### 틀린@1 — 12/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-01`  Azadov Mirjalol Bahodir o‘g‘li tahsil olayotgan universitet va magistratura yo‘nalishi shifrini toping
- `uz-05`  Abdug‘aniyev Akobir Akmaljon o‘g‘lining muassasasi va ilmiy rahbarini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-11`  Nasimov Umedjon Ulugbekovich tahsil olayotgan universitetni toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `ru-01`  Найдите диапазон номеров, звонки с которых рекомендовано не принимать, указанный вместе со способом «SMS-просьба о помощи»
- `ru-05`  Найдите номер «Телефона доверия» УВМД России и номер дежурной части
- `pil-06`  Hanapin ang tatlong cash reload card na binabalaang hinihingi ng mga scammer kasama ng gift card tulad ng iTunes o Amazon

### 틀린@3 — 5/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping

## e5-large-instruct [512]

### 틀린@1 — 44/150개

- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-09`  판사 정재우·이영제가 함께 심리한 항소심에서, 양형 이유로 건조물침입죄의 침입 태양에 비추어 건조물의 평온이 어느 정도 침해되었다고 본 판단을 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-20`  대출브로커 A·B가 D○○에게 금융회사 8곳으로부터 360억 원 대출을 알선한 개발사업의 명칭을 찾아주세요
- `en-01`  In the Antonio Nathaniel Davenport case, find the murder offense the government had to prove beyond a reasonable doubt
- `en-05`  Find the offense charged in Count 1 against Bryan Lee Burrows and the case number
- `en-06`  In the Cedric Ray Jones case, find the total sentence imposed by the trial court and the consecutive sentence for the §924(c) count
- `en-14`  Find the federal regulation under which John Anthony O'Brien was charged and the date he was arrested
- `en-15`  In the John Anthony O'Brien case, find the words the emergency radio call used to describe his condition
- `en-17`  In the Carlos Meza Guillermo case, find the magistrate judge who wrote the report and recommendation adopted by the court and the date it was written
- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-10`  请找出凉城县人民检察院对张某某作出认定时指出的、引发与杨某某争执的行为以及杨某某所受的骨折部位
- `ch-18`  请找出法院经审理认定的四川省川谷坊酒业有限公司非法吸收公众存款的金额和被吸收人人数
- `ch-19`  请找出对曾庆芬判处的有期徒刑和罚金数额
- `ch-20`  请找出四川省川谷坊酒业有限公司向不特定多数人集资时承诺的月利率
- `vn-03`  Hãy tìm mức án và ngày bắt đầu tính thời hạn tù đối với Võ Minh P, người được thông báo có quyền kháng cáo lên Tòa án nhân dân tỉnh Bà Rịa - Vũng Tàu
- `vn-05`  Hãy tìm chủ tọa phiên tòa và thư ký phiên tòa đã xét xử Lại Văn P cùng với các Hội thẩm nhân dân Vương Tấn Độ, Lê Văn Khanh
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-07`  Hãy tìm loại điện thoại và giá trị định giá của chiếc điện thoại mà Lại Văn P đã trộm của anh Nguyễn Đình V
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `vn-11`  Hãy tìm tài sản mà Lâm Tuấn A đã trộm tại nhà ông Nguyễn Thành P và số tiền trong chiếc bóp
- `vn-13`  Hãy tìm tên cửa hàng điện thoại nơi Lâm Tuấn A bán chiếc Iphone X đã trộm và tên chủ cửa hàng đó
- `vn-20`  Hãy tìm hai vật bị tịch thu tiêu hủy trong vụ án Trần Xuân H
- `uz-02`  BMT Bosh Assambleyasi xulq-atvor kodeksi va Yevropa Kengashi konvensiyasidan keyin korrupsiya ta’rifi sifatida keltirilgan O‘zbekistonning korrupsiyaga qarshi kurashish to‘g‘risidagi qonuni qabul qilingan sanani toping
- `uz-05`  Abdug‘aniyev Akobir Akmaljon o‘g‘lining muassasasi va ilmiy rahbarini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-11`  Nasimov Umedjon Ulugbekovich tahsil olayotgan universitetni toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `ru-01`  Найдите диапазон номеров, звонки с которых рекомендовано не принимать, указанный вместе со способом «SMS-просьба о помощи»
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-11`  Найдите то, что рекомендовано проверить в первую очередь для распознавания подделки после слов «Мошенники создают сайты-клоны торговых площадок»
- `ru-16`  Найдите наказание, назначенное Емельянову Даниилу Игоревичу Бежецким межрайонным судом Тверской области
- `ru-17`  Найдите общую сумму материальных ценностей, присвоенных и растраченных Емельяновым Д.И. в АО «Р»
- `ru-19`  Найдите решение, исключённое апелляцией Тверского областного суда из приговора суда первой инстанции по делу Емельянова Д.И.
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-01`  Hanapin ang ahensyang nagbibigay ng awtoridad na magpayo tungkol sa batas sa imigrasyon, dahil hindi tulad sa Latin Amerika ay hindi maaaring magbigay ng payong legal ang notaryo sa Estados Unidos
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-10`  Hanapin ang numerong dapat pindutin upang piliin ang Tagalog sa hotline ng pagsusumbong ng scam, matapos ang babala tungkol sa hinihinging padala sa MoneyGram o Western Union
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 17/150개

- `en-01`  In the Antonio Nathaniel Davenport case, find the murder offense the government had to prove beyond a reasonable doubt
- `en-05`  Find the offense charged in Count 1 against Bryan Lee Burrows and the case number
- `en-14`  Find the federal regulation under which John Anthony O'Brien was charged and the date he was arrested
- `en-15`  In the John Anthony O'Brien case, find the words the emergency radio call used to describe his condition
- `en-17`  In the Carlos Meza Guillermo case, find the magistrate judge who wrote the report and recommendation adopted by the court and the date it was written
- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-18`  请找出法院经审理认定的四川省川谷坊酒业有限公司非法吸收公众存款的金额和被吸收人人数
- `ch-20`  请找出四川省川谷坊酒业有限公司向不特定多数人集资时承诺的月利率
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-07`  Hãy tìm loại điện thoại và giá trị định giá của chiếc điện thoại mà Lại Văn P đã trộm của anh Nguyễn Đình V
- `vn-20`  Hãy tìm hai vật bị tịch thu tiêu hủy trong vụ án Trần Xuân H
- `uz-05`  Abdug‘aniyev Akobir Akmaljon o‘g‘lining muassasasi va ilmiy rahbarini toping
- `uz-11`  Nasimov Umedjon Ulugbekovich tahsil olayotgan universitetni toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

## e5-small-ko [512]

### 틀린@1 — 30/150개

- `ko-07`  재판장 신광렬이 선고한 건조물침입·업무방해 항소심에서 피고인에게 정해진 벌금액을 찾아주세요
- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ko-26`  K반도체 국가핵심기술 유출사건에서 중국 C社 설립에 출자한 중국지방정부와 중국 반도체설계회사의 금액을 찾아주세요
- `en-12`  In the Donte J. Carter case, find the additional fact the D.C. Court of Appeals said must be considered in deciding whether a seizure occurred
- `en-15`  In the John Anthony O'Brien case, find the words the emergency radio call used to describe his condition
- `en-16`  In the John Anthony O'Brien case, find the date and time of the status conference set by the court that denied the motion to dismiss
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `ch-09`  请找出凉城县人民检察院对张某某作出的最终处理决定
- `ch-18`  请找出法院经审理认定的四川省川谷坊酒业有限公司非法吸收公众存款的金额和被吸收人人数
- `ch-20`  请找出四川省川谷坊酒业有限公司向不特定多数人集资时承诺的月利率
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `vn-11`  Hãy tìm tài sản mà Lâm Tuấn A đã trộm tại nhà ông Nguyễn Thành P và số tiền trong chiếc bóp
- `vn-16`  Hãy tìm mức án và ngày bắt đầu tính thời hạn tù mà Đỗ Văn Ph phải chịu về tội “Tàng trữ trái phép chất ma túy”
- `vn-19`  Hãy tìm tiền án của Trần Xuân H
- `uz-01`  Azadov Mirjalol Bahodir o‘g‘li tahsil olayotgan universitet va magistratura yo‘nalishi shifrini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-15`  Найдите марку автомобиля, в части конфискации которого Верховный Суд России отменил решение по делу Конева И.А.
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-12`  Hanapin ang numero ng administrative matter ng Korte Suprema na nagtatakdang pumalit ang salaysay ni SHEILA MARTIR sa direct examination
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 8/150개

- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ko-26`  K반도체 국가핵심기술 유출사건에서 중국 C社 설립에 출자한 중국지방정부와 중국 반도체설계회사의 금액을 찾아주세요
- `ch-18`  请找出法院经审理认定的四川省川谷坊酒业有限公司非法吸收公众存款的金额和被吸收人人数
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `uz-01`  Azadov Mirjalol Bahodir o‘g‘li tahsil olayotgan universitet va magistratura yo‘nalishi shifrini toping
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari

## bge-m3 [512]

### 틀린@1 — 28/150개

- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-23`  세계 1위 K반도체 국가핵심기술 유출사건에서 국내 A社가 10나노대 D램 공정기술 개발에 투입한 기간과 금액을 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ko-26`  K반도체 국가핵심기술 유출사건에서 중국 C社 설립에 출자한 중국지방정부와 중국 반도체설계회사의 금액을 찾아주세요
- `en-12`  In the Donte J. Carter case, find the additional fact the D.C. Court of Appeals said must be considered in deciding whether a seizure occurred
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `vn-01`  Hãy tìm Thẩm phán – Chủ tọa phiên tòa đã xét xử Võ Minh P cùng với các Hội thẩm nhân dân Bùi Thị Kim Thủy, Dương Thị Được
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `vn-13`  Hãy tìm tên cửa hàng điện thoại nơi Lâm Tuấn A bán chiếc Iphone X đã trộm và tên chủ cửa hàng đó
- `vn-15`  Hãy tìm khối lượng giám định và giá mua của gói Heroin mà Đỗ Văn Ph đã mua tại đường Đặng Huy T và bị phát hiện khi đang tàng trữ
- `vn-17`  Hãy tìm hai loại hình phạt cùng mức tối thiểu và tối đa được quy định tại khoản 1 Điều 173 BLHS áp dụng đối với Trần Xuân H
- `uz-05`  Abdug‘aniyev Akobir Akmaljon o‘g‘lining muassasasi va ilmiy rahbarini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-09`  Найдите документ, который, по разъяснению прокуратуры Центрального района города Оренбурга, выдают при обращении с заявлением в полицию
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-15`  Найдите марку автомобиля, в части конфискации которого Верховный Суд России отменил решение по делу Конева И.А.
- `ru-19`  Найдите решение, исключённое апелляцией Тверского областного суда из приговора суда первой инстанции по делу Емельянова Д.И.
- `pil-01`  Hanapin ang ahensyang nagbibigay ng awtoridad na magpayo tungkol sa batas sa imigrasyon, dahil hindi tulad sa Latin Amerika ay hindi maaaring magbigay ng payong legal ang notaryo sa Estados Unidos
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-12`  Hanapin ang numero ng administrative matter ng Korte Suprema na nagtatakdang pumalit ang salaysay ni SHEILA MARTIR sa direct examination
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 4/150개

- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay

## bge-m3 [512] [hybrid(dense+sparse)]

### 틀린@1 — 20/150개

- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ko-26`  K반도체 국가핵심기술 유출사건에서 중국 C社 설립에 출자한 중국지방정부와 중국 반도체설계회사의 금액을 찾아주세요
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-08`  Hãy tìm lý do Tòa án không áp dụng hình phạt bổ sung là phạt tiền từ 5.000.000 đồng đến 50.000.000 đồng trong vụ án Lại Văn P
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `vn-13`  Hãy tìm tên cửa hàng điện thoại nơi Lâm Tuấn A bán chiếc Iphone X đã trộm và tên chủ cửa hàng đó
- `vn-17`  Hãy tìm hai loại hình phạt cùng mức tối thiểu và tối đa được quy định tại khoản 1 Điều 173 BLHS áp dụng đối với Trần Xuân H
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-19`  Найдите решение, исключённое апелляцией Тверского областного суда из приговора суда первой инстанции по делу Емельянова Д.И.
- `pil-01`  Hanapin ang ahensyang nagbibigay ng awtoridad na magpayo tungkol sa batas sa imigrasyon, dahil hindi tulad sa Latin Amerika ay hindi maaaring magbigay ng payong legal ang notaryo sa Estados Unidos
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 1/150개

- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay

## bge-m3 [full]

### 틀린@1 — 15/150개

- `ko-13`  서울 성동구 공장용지와 그 지상 건물의 재산분할을 정한 이혼 조정을 담당한 법원과 그 사건번호를 찾아주세요
- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `vn-01`  Hãy tìm Thẩm phán – Chủ tọa phiên tòa đã xét xử Võ Minh P cùng với các Hội thẩm nhân dân Bùi Thị Kim Thủy, Dương Thị Được
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `uz-01`  Azadov Mirjalol Bahodir o‘g‘li tahsil olayotgan universitet va magistratura yo‘nalishi shifrini toping
- `uz-05`  Abdug‘aniyev Akobir Akmaljon o‘g‘lining muassasasi va ilmiy rahbarini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `ru-01`  Найдите диапазон номеров, звонки с которых рекомендовано не принимать, указанный вместе со способом «SMS-просьба о помощи»
- `ru-05`  Найдите номер «Телефона доверия» УВМД России и номер дежурной части
- `ru-12`  Найдите в списке, завершающемся словами «Услышав данные фразы, прекратите разговор», требование взять новый кредит и передать полученные деньги сотруднику
- `pil-06`  Hanapin ang tatlong cash reload card na binabalaang hinihingi ng mga scammer kasama ng gift card tulad ng iTunes o Amazon
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 5/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `ru-12`  Найдите в списке, завершающемся словами «Услышав данные фразы, прекратите разговор», требование взять новый кредит и передать полученные деньги сотруднику

## bge-m3 [full] [hybrid(dense+sparse)]

### 틀린@1 — 9/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-01`  Azadov Mirjalol Bahodir o‘g‘li tahsil olayotgan universitet va magistratura yo‘nalishi shifrini toping
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping
- `uz-18`  O‘zbekiston Respublikasi Oliy sudi e’lon qilgan rasmiy statistikada 2024-yilda jinoyat sudlari ko‘rib chiqqan ishlar soni va ular tegishli bo‘lgan shaxslar sonini toping
- `ru-01`  Найдите диапазон номеров, звонки с которых рекомендовано не принимать, указанный вместе со способом «SMS-просьба о помощи»
- `pil-06`  Hanapin ang tatlong cash reload card na binabalaang hinihingi ng mga scammer kasama ng gift card tulad ng iTunes o Amazon
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 3/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-17`  Kamolov Xayrulloxon Juraxon o’g’li egallab turgan lavozimni toping

## harrier-0.6b [512]

### 틀린@1 — 41/150개

- `ko-08`  피고인이 비밀번호를 눌러 침입한 C산후조리원이 있던 고양시 일산동구 건물의 층수와 침입 날짜를 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ko-26`  K반도체 국가핵심기술 유출사건에서 중국 C社 설립에 출자한 중국지방정부와 중국 반도체설계회사의 금액을 찾아주세요
- `en-07`  Find the standard under which the Hunter decision, cited in the Cedric Ray Jones case, held that an appeal waiver could not be enforced
- `en-08`  Find the context of the waiver addressed in Hunter, unlike the Cedric Ray Jones petition, and the type of waiver this petition concerns
- `en-12`  In the Donte J. Carter case, find the additional fact the D.C. Court of Appeals said must be considered in deciding whether a seizure occurred
- `en-13`  Find what Judge McLeese, concurring only in the judgment, said the majority opinion had given to Carter's race
- `en-16`  In the John Anthony O'Brien case, find the date and time of the status conference set by the court that denied the motion to dismiss
- `ch-01`  请找出孙晓龙寻衅滋事案中判处的有期徒刑刑期
- `ch-03`  请找出上海市虹口区人民检察院起诉的俞某2在推特上注册的昵称、账号以及发送的推文数量
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `ch-09`  请找出凉城县人民检察院对张某某作出的最终处理决定
- `ch-18`  请找出法院经审理认定的四川省川谷坊酒业有限公司非法吸收公众存款的金额和被吸收人人数
- `ch-19`  请找出对曾庆芬判处的有期徒刑和罚金数额
- `ch-20`  请找出四川省川谷坊酒业有限公司向不特定多数人集资时承诺的月利率
- `vn-01`  Hãy tìm Thẩm phán – Chủ tọa phiên tòa đã xét xử Võ Minh P cùng với các Hội thẩm nhân dân Bùi Thị Kim Thủy, Dương Thị Được
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-07`  Hãy tìm loại điện thoại và giá trị định giá của chiếc điện thoại mà Lại Văn P đã trộm của anh Nguyễn Đình V
- `vn-11`  Hãy tìm tài sản mà Lâm Tuấn A đã trộm tại nhà ông Nguyễn Thành P và số tiền trong chiếc bóp
- `vn-16`  Hãy tìm mức án và ngày bắt đầu tính thời hạn tù mà Đỗ Văn Ph phải chịu về tội “Tàng trữ trái phép chất ma túy”
- `vn-20`  Hãy tìm hai vật bị tịch thu tiêu hủy trong vụ án Trần Xuân H
- `uz-02`  BMT Bosh Assambleyasi xulq-atvor kodeksi va Yevropa Kengashi konvensiyasidan keyin korrupsiya ta’rifi sifatida keltirilgan O‘zbekistonning korrupsiyaga qarshi kurashish to‘g‘risidagi qonuni qabul qilingan sanani toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-18`  O‘zbekiston Respublikasi Oliy sudi e’lon qilgan rasmiy statistikada 2024-yilda jinoyat sudlari ko‘rib chiqqan ishlar soni va ular tegishli bo‘lgan shaxslar sonini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-05`  Найдите номер «Телефона доверия» УВМД России и номер дежурной части
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-15`  Найдите марку автомобиля, в части конфискации которого Верховный Суд России отменил решение по делу Конева И.А.
- `ru-17`  Найдите общую сумму материальных ценностей, присвоенных и растраченных Емельяновым Д.И. в АО «Р»
- `ru-18`  Найдите период, в течение которого совершено преступление Емельяновым Д.И.
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-01`  Hanapin ang ahensyang nagbibigay ng awtoridad na magpayo tungkol sa batas sa imigrasyon, dahil hindi tulad sa Latin Amerika ay hindi maaaring magbigay ng payong legal ang notaryo sa Estados Unidos
- `pil-04`  Hanapin ang mga paraan ng pagbabayad na hinding-hindi dapat gamitin ng refugee kapalit ng tulong sa imigrasyon, na nakalista kasabay ng payong kumonsulta muna sa case manager ng resettlement agency
- `pil-07`  Hanapin ang inirekomendang paraan ng pakikipag-ugnayan sa kompanya ng utility kapag nag-aalala ka sa hindi nabayarang bill, matapos ang babala tungkol sa hinihinging MoneyPak, Vanilla, at Reloadit
- `pil-10`  Hanapin ang numerong dapat pindutin upang piliin ang Tagalog sa hotline ng pagsusumbong ng scam, matapos ang babala tungkol sa hinihinging padala sa MoneyGram o Western Union
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 12/150개

- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `vn-01`  Hãy tìm Thẩm phán – Chủ tọa phiên tòa đã xét xử Võ Minh P cùng với các Hội thẩm nhân dân Bùi Thị Kim Thủy, Dương Thị Được
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `uz-18`  O‘zbekiston Respublikasi Oliy sudi e’lon qilgan rasmiy statistikada 2024-yilda jinoyat sudlari ko‘rib chiqqan ishlar soni va ular tegishli bo‘lgan shaxslar sonini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-17`  Найдите общую сумму материальных ценностей, присвоенных и растраченных Емельяновым Д.И. в АО «Р»
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-10`  Hanapin ang numerong dapat pindutin upang piliin ang Tagalog sa hotline ng pagsusumbong ng scam, matapos ang babala tungkol sa hinihinging padala sa MoneyGram o Western Union
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

## harrier-0.6b [full]

### 틀린@1 — 9/150개

- `en-18`  Find the rulings the court made on the motion at Docket No. 39 and on the motion at Docket No. 45
- `ch-06`  请找出审判长李凤波宣判时没收并上缴国库的犯罪工具，以及不服判决时可以上诉的上级法院
- `uz-11`  Nasimov Umedjon Ulugbekovich tahsil olayotgan universitetni toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `ru-11`  Найдите то, что рекомендовано проверить в первую очередь для распознавания подделки после слов «Мошенники создают сайты-клоны торговых площадок»
- `ru-12`  Найдите в списке, завершающемся словами «Услышав данные фразы, прекратите разговор», требование взять новый кредит и передать полученные деньги сотруднику
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-10`  Hanapin ang numerong dapat pindutin upang piliin ang Tagalog sa hotline ng pagsusumbong ng scam, matapos ang babala tungkol sa hinihinging padala sa MoneyGram o Western Union
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 1/150개

- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping

## harrier-270m [512]

### 틀린@1 — 44/150개

- `ko-12`  서울북부지방법원이 채권압류 및 추심명령을 발령한 사건에서, 피고인이 2022년 1월부터 9월까지 받은 임대료 수익금 합계와 피해자에게 분배하지 않은 금액을 찾아주세요
- `ko-18`  카카오톡으로 연락해 화물 수령을 부탁한 B가 피고인의 계좌로 입금한 사례금을 찾아주세요
- `ko-20`  대출브로커 A·B가 D○○에게 금융회사 8곳으로부터 360억 원 대출을 알선한 개발사업의 명칭을 찾아주세요
- `ko-23`  세계 1위 K반도체 국가핵심기술 유출사건에서 국내 A社가 10나노대 D램 공정기술 개발에 투입한 기간과 금액을 찾아주세요
- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `en-07`  Find the standard under which the Hunter decision, cited in the Cedric Ray Jones case, held that an appeal waiver could not be enforced
- `en-12`  In the Donte J. Carter case, find the additional fact the D.C. Court of Appeals said must be considered in deciding whether a seizure occurred
- `en-13`  Find what Judge McLeese, concurring only in the judgment, said the majority opinion had given to Carter's race
- `en-15`  In the John Anthony O'Brien case, find the words the emergency radio call used to describe his condition
- `en-16`  In the John Anthony O'Brien case, find the date and time of the status conference set by the court that denied the motion to dismiss
- `ch-14`  请找出重庆恒韵医药有限公司的实际控制人李仕林最终被合并执行的刑罚
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-10`  Hãy tìm chủ tọa phiên tòa đã xét xử Lâm Tuấn A cùng với các Hội thẩm nhân dân Phạm Thị Ngọc, Nguyễn Ngọc Cảnh
- `vn-11`  Hãy tìm tài sản mà Lâm Tuấn A đã trộm tại nhà ông Nguyễn Thành P và số tiền trong chiếc bóp
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `vn-13`  Hãy tìm tên cửa hàng điện thoại nơi Lâm Tuấn A bán chiếc Iphone X đã trộm và tên chủ cửa hàng đó
- `vn-16`  Hãy tìm mức án và ngày bắt đầu tính thời hạn tù mà Đỗ Văn Ph phải chịu về tội “Tàng trữ trái phép chất ma túy”
- `vn-20`  Hãy tìm hai vật bị tịch thu tiêu hủy trong vụ án Trần Xuân H
- `uz-07`  IIV Jinoyat-qidiruv bosh boshqarmasi qayd etgan 2019-yil va 2020-yilning 1-7 oylaridagi firibgarlik jinoyatlari sonini toping
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-10`  Firibgarlik aqlli jinoyat ekanini aytgan Termiz shahar sudi sudyasi yordamchisining ismini toping
- `uz-12`  Kiberjinoyat ta’rifi bilan birga keltirilgan, O‘zbekistonda kiberjinoyatchilikka qarshi kurashish asosi bo‘lgan uchta qonunni toping
- `uz-13`  Birlashgan Millatlar Tashkiloti, Yevropa Ittifoqi va Interpol bilan birga tilga olingan, kiberjinoyatchilikka qarshi kurashdagi asosiy xalqaro konvensiya nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `uz-18`  O‘zbekiston Respublikasi Oliy sudi e’lon qilgan rasmiy statistikada 2024-yilda jinoyat sudlari ko‘rib chiqqan ishlar soni va ular tegishli bo‘lgan shaxslar sonini toping
- `ru-04`  Найдите официальный реестр органа, в котором, по разъяснению прокуратуры Омутинского района Тюменской области, можно проверить, заблокирован ли интернет-ресурс
- `ru-05`  Найдите номер «Телефона доверия» УВМД России и номер дежурной части
- `ru-10`  Найдите сведения, которые, как предупреждает прокуратура Центрального района города Оренбурга, мошенники просят сообщить под предлогом перевода денег за товар, объявление о продаже которого вы разместили в социальных сетях
- `ru-11`  Найдите то, что рекомендовано проверить в первую очередь для распознавания подделки после слов «Мошенники создают сайты-клоны торговых площадок»
- `ru-13`  Найдите наказание, назначенное Коневу Игорю Анатольевичу и.о. мирового судьи в суде первой инстанции
- `ru-18`  Найдите период, в течение которого совершено преступление Емельяновым Д.И.
- `ru-19`  Найдите решение, исключённое апелляцией Тверского областного суда из приговора суда первой инстанции по делу Емельянова Д.И.
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-01`  Hanapin ang ahensyang nagbibigay ng awtoridad na magpayo tungkol sa batas sa imigrasyon, dahil hindi tulad sa Latin Amerika ay hindi maaaring magbigay ng payong legal ang notaryo sa Estados Unidos
- `pil-05`  Hanapin ang numero ng telepono ng USCIS Contact Center at ang TTY na numero para sa mga may kapansanan sa pandinig
- `pil-07`  Hanapin ang inirekomendang paraan ng pakikipag-ugnayan sa kompanya ng utility kapag nag-aalala ka sa hindi nabayarang bill, matapos ang babala tungkol sa hinihinging MoneyPak, Vanilla, at Reloadit
- `pil-10`  Hanapin ang numerong dapat pindutin upang piliin ang Tagalog sa hotline ng pagsusumbong ng scam, matapos ang babala tungkol sa hinihinging padala sa MoneyGram o Western Union
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay
- `pil-12`  Hanapin ang numero ng administrative matter ng Korte Suprema na nagtatakdang pumalit ang salaysay ni SHEILA MARTIR sa direct examination
- `pil-13`  Hanapin ang petsa at tinatayang oras kung kailan nasaksihan ni SHEILA MARTIR ang pangyayari
- `pil-15`  Hanapin ang pangalan at Roll number ng abogado ng Public Attorney’s Office na sumumpang siya mismo ang kumuha ng salaysay ni Sheila Martir sa kanyang opisina at nagtala’t nagsalin nito sa Filipino
- `pil-18`  Hanapin ang numero ng sentro ng USCIS na dapat tawagan kapag may alalahanin sa visa o dokumento sa imigrasyon, sa bahaging nagtuturong magsumbong sa FTC.gov/Complaint

### 틀린@3 — 10/150개

- `ko-25`  중국 C社 관계자들이 위장회사를 통한 입사·중국 이메일 사용 등과 함께, 출국금지·체포 시 전파하기로 정한 암호를 찾아주세요
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `vn-06`  Hãy tìm thời gian và địa điểm Lại Văn P trộm chiếc điện thoại Realme C2
- `vn-10`  Hãy tìm chủ tọa phiên tòa đã xét xử Lâm Tuấn A cùng với các Hội thẩm nhân dân Phạm Thị Ngọc, Nguyễn Ngọc Cảnh
- `vn-16`  Hãy tìm mức án và ngày bắt đầu tính thời hạn tù mà Đỗ Văn Ph phải chịu về tội “Tàng trữ trái phép chất ma túy”
- `uz-08`  Shaffof tizimga o‘tishni ilgari surib, jamiyatda axborot shaffofligini ta’minlash uchun yaratish taklif etilgan tizim nomini toping
- `uz-14`  Zaxira nusxalarni tiklashgacha davom etadigan kiberjinoyatdan zarar ko‘rganda ko‘riladigan choralarning birinchisi va ikkinchisini toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `ru-20`  Найдите вывод Верховного Суда России по кассационной жалобе защитника по делу Емельянова Д.И.
- `pil-11`  Hanapin ang pangalan ng imbestigador na dumating sa bahay ni Sheila Martir kinabukasan ng pamamaril upang kunin ang kanyang salaysay

## harrier-270m [full]

### 틀린@1 — 16/150개

- `ko-02`  서울서부지방검찰청이 적발한, 전국 379개 병·의원 의사·약사에게 제공된 의약품 불법리베이트 총액을 찾아주세요
- `ch-11`  请找出北京华业资本控股股份有限公司的证券代码、股票简称及其公告编号
- `ch-12`  请找出对北京华业资本控股股份有限公司作出刑事裁判的法院和两个案号
- `ch-13`  请找出重庆恒韵医药有限公司因四项罪名并罚而合并执行的罚金总额
- `ch-14`  请找出重庆恒韵医药有限公司的实际控制人李仕林最终被合并执行的刑罚
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `vn-05`  Hãy tìm chủ tọa phiên tòa và thư ký phiên tòa đã xét xử Lại Văn P cùng với các Hội thẩm nhân dân Vương Tấn Độ, Lê Văn Khanh
- `vn-07`  Hãy tìm loại điện thoại và giá trị định giá của chiếc điện thoại mà Lại Văn P đã trộm của anh Nguyễn Đình V
- `vn-12`  Hãy tìm tổng giá trị tài sản do Hội đồng định giá tài sản xác định đối với bị hại bị mất chiếc bóp có thẻ Viettinbank và BIDV
- `uz-06`  Xavfli residivist yoki uyushgan guruh tomonidan sodir etilgan hollarni ham qo‘shib, O‘zbekiston Jinoyat kodeksining 168-moddasidagi firibgarlik uchun eng og‘ir jazo muddatini toping
- `uz-09`  Qashqadaryoda mashhur kompaniyaning soxta aksiyalarini sotgan uyushgan jinoiy guruh o‘zlashtirgan mablag‘ miqdori va jabrlanuvchilar sonini toping
- `uz-11`  Nasimov Umedjon Ulugbekovich tahsil olayotgan universitetni toping
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping
- `ru-11`  Найдите то, что рекомендовано проверить в первую очередь для распознавания подделки после слов «Мошенники создают сайты-клоны торговых площадок»
- `ru-12`  Найдите в списке, завершающемся словами «Услышав данные фразы, прекратите разговор», требование взять новый кредит и передать полученные деньги сотруднику
- `pil-14`  Hanapin kung ilang beses at saang bahagi tinamaan ng putok si Lito Bartolome sa lugar kung saan naroon ang dalawang lalaking sumigaw ng «Raffy!» at «Pare!»

### 틀린@3 — 7/150개

- `ko-02`  서울서부지방검찰청이 적발한, 전국 379개 병·의원 의사·약사에게 제공된 의약품 불법리베이트 총액을 찾아주세요
- `ch-11`  请找出北京华业资本控股股份有限公司的证券代码、股票简称及其公告编号
- `ch-12`  请找出对北京华业资本控股股份有限公司作出刑事裁判的法院和两个案号
- `ch-13`  请找出重庆恒韵医药有限公司因四项罪名并罚而合并执行的罚金总额
- `ch-14`  请找出重庆恒韵医药有限公司的实际控制人李仕林最终被合并执行的刑罚
- `ch-15`  请找出对刘荣华、韦泽禹、白晓敏分别数罪并罚宣告的刑罚中，财务总监刘荣华的有期徒刑和罚金
- `uz-15`  Qodirjonova Madina Ulug’bek qizi keltirgan DOI raqamini toping

