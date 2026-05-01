#!/usr/bin/env python3
"""
Script kiểm tra và chạy thử nghiệm nhanh cho đồ án DS317
"""

import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'flask', 'joblib', 'xgboost', 'lightgbm'
    ]

    print("[INFO] Kiem tra cac thu vien can thiet...")
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing.append(package)

    if missing:
        print(f"\n[WARN] Thieu cac thu vien: {', '.join(missing)}")
        print("Cài đặt bằng: pip install " + ' '.join(missing))
        return False
    else:
        print("\n[OK] Tat ca thu vien da duoc cai dat!")
        return True

def check_data_files():
    """Kiểm tra file dữ liệu"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "du-lieu-thuc-nghiem"

    print("[INFO] Kiem tra file du lieu...")

    if not dataset_dir.exists():
        print(f"[ERROR] Thu muc du lieu khong ton tai: {dataset_dir}")
        return False

    # Kiểm tra các file cần thiết (có thể thay đổi tùy dataset thực tế)
    data_files = ["user.json"]  # Ví dụ

    for file in data_files:
        file_path = dataset_dir / file
        if file_path.exists():
            print(f"[OK] {file}")
        else:
            print(f"[WARN] {file} khong tim thay (co the chua can thiet)")

        print("[OK] Kiem tra du lieu hoan thanh!")
    return True

def check_scripts():
    """Kiểm tra các script cần thiết"""
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / "experiment" # Corrected path to experiment scripts

    print("[INFO] Kiem tra script...")

    required_scripts = [
        "experiment/phase_1_data_cleaning.py",
        "experiment/phase_2_data_transformation.py",
        "experiment/phase_3_eda.py",
        "experiment/phase_4_data_labeling.py",
        "experiment/phase_5_data_splitting.py",
        "experiment/phase_6_model_training.py",
        "experiment/phase_7_model_evaluation.py",
        "experiment/phase_8_model_interpretability.py",
        "final/source-code/main_experiment.py",
        "final/source-code/demo_app.py"
    ]

    missing = []
    for script in required_scripts:
        script_path = project_root / script
        if script_path.exists():
            print(f"[OK] {script}")
        else:
            print(f"[MISSING] {script}")
            missing.append(script)

    if missing:
        print(f"\n[ERROR] Thieu cac script: {missing}")
        return False
    else:
        print("\n[OK] Tat ca script da co!")
        return True

def run_quick_test():
    """Chạy test nhanh với dữ liệu nhỏ"""
    print("[INFO] Chay test nhanh...")

    project_root = Path(__file__).resolve().parents[2]
    main_script = project_root / "final" / "source-code" / "main_experiment.py"

    # Chạy phase 2 với giới hạn dữ liệu nhỏ
    cmd = [ # Changed to Phase 1 for quick test
        sys.executable,
        str(main_script),
        "--phase", "1",
        "--max-rows", "1000"  # Test với 1000 hàng
    ]

    try:
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("[OK] Test phase 1 thanh cong!")
            return True
        else:
            print("[ERROR] Test phase 1 that bai!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Test timeout!")
        return False
    except Exception as e:
        print(f"[ERROR] Loi khi chay test: {e}")
        return False

def main():
    """Hàm chính"""
    print("[INFO] Kiem tra moi truong do an DS317\n")

    checks = [
        ("Thư viện Python", check_requirements),
        ("File dữ liệu", check_data_files),
        ("Script", check_scripts),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"KIỂM TRA: {name.upper()}")
        print('='*50)
        if not check_func():
            all_passed = False

    print(f"\n{'='*50}")
    print("KẾT QUẢ TỔNG QUAN")
    print('='*50)

    if all_passed:
        print("[OK] Tat ca kiem tra da pass!")
        print("\n[INFO] Chay test nhanh...")

        if run_quick_test():
            print("\n[OK] Moi truong san sang! Co the chay do an.")
            print("Chạy đầy đủ: python final/source-code/main_experiment.py --phase all")
            print("Chạy demo: python final/source-code/demo_app.py")
        else:
            print("\n[WARN] Test that bai. Kiem tra lai cau hinh.")
    else:
        print("[ERROR] Mot so kiem tra that bai. Vui long khac phuc truoc khi chay.")

if __name__ == "__main__":
    main()