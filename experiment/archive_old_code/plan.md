[APPROVED - 2026-04-27] Redesigned to practical early-warning user-course system.

## Plan: Production-Oriented Early Warning by User-Course

Muc tieu:
- Van hanh he thong du doan canh bao som cho tung user-course truoc khi khoa hoc ket thuc.
- Toi uu cho can thiep thuc te, khong chi danh gia hoc thuat.

## A) Business & Data Contract
1. Chot don vi du lieu 1 mau = 1 user_id + course_id + prediction_time.
2. Chot 3 moc scoring: 25%, 50%, 75% tien do khoa hoc.
3. Chot contract output: risk_score, risk_level, top_risk_factors, recommended_action.
4. Chot SLA van hanh: data freshness, scoring success rate, intervention deadline.

## B) Time-aware Pipeline Refactor
5. Phase 1: bo sung timeline course (start/end/checkpoints) va metadata prediction_time.
6. Phase 2: bo sung check timestamp consistency, duplicate event, out-of-range event.
7. Phase 3: sinh feature theo cua so thoi gian toi prediction_time.
8. Phase 4: tao label risk outcome-driven va stage-warning features user-course.

## C) Split, Modeling, Evaluation
9. Phase 5: split theo thoi gian + group user_course_key de tranh leakage.
10. Phase 6: train model + probability calibration de risk_score dung nghia van hanh.
11. Phase 7: danh gia theo PR-AUC, Recall@HighRisk, Precision@HighRisk, calibration.
12. Phase 8: giai thich top factors de phuc vu can thiep cua giang vien.

## D) Operations & Feedback Loop
13. Xuat danh sach canh bao uu tien theo impact score moi ngay.
14. Luu log can thiep (action taken, response time, outcome).
15. Backtesting hieu qua canh bao theo tung course va tung checkpoint.
16. Lap lich retraining hang thang + drift monitoring theo cohort.

## Verification Checklist
1. Pipeline chay 8 phase khong loi schema/du lieu.
2. Output co day du cot user-course + risk metadata.
3. Khong leak du lieu sau prediction_time vao feature train.
4. Recall nhom High risk dat nguong nghiep vu da chot.
5. Co bao cao interpretability cho tung dot scoring.

## Scope
Trong pham vi:
- Refactor bai toan, data contract, feature-label-eval cho warning user-course.
- Dong bo tai lieu README, kich ban thuc nghiem, design doc.

Ngoai pham vi:
- Trien khai UI production.
- Ket noi he thong nhan su/CRM can thiep ngoai pipeline.
