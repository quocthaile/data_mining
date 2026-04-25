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

    print("🔍 Kiểm tra các thư viện cần thiết...")
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Thiếu các thư viện: {', '.join(missing)}")
        print("Cài đặt bằng: pip install " + ' '.join(missing))
        return False
    else:
        print("\n✅ Tất cả thư viện đã được cài đặt!")
        return True

def check_data_files():
    """Kiểm tra file dữ liệu"""
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "du-lieu-thuc-nghiem"

    print("🔍 Kiểm tra file dữ liệu...")

    if not dataset_dir.exists():
        print(f"❌ Thư mục dữ liệu không tồn tại: {dataset_dir}")
        return False

    # Kiểm tra các file cần thiết (có thể thay đổi tùy dataset thực tế)
    data_files = ["user.json", "user_school_en.json"]  # Ví dụ

    for file in data_files:
        file_path = dataset_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} không tìm thấy (có thể chưa cần thiết)")

    print("✅ Kiểm tra dữ liệu hoàn thành!")
    return True

def check_scripts():
    """Kiểm tra các script cần thiết"""
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / "experiment" # Corrected path to experiment scripts

    print("🔍 Kiểm tra script...")

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
            print(f"✅ {script}")
        else:
            print(f"❌ {script}")
            missing.append(script)

    if missing:
        print(f"\n❌ Thiếu các script: {missing}")
        return False
    else:
        print("\n✅ Tất cả script đã có!")
        return True

def run_quick_test():
    """Chạy test nhanh với dữ liệu nhỏ"""
    print("🧪 Chạy test nhanh...")

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
            print("✅ Test phase 1 thành công!")
            return True
        else:
            print("❌ Test phase 1 thất bại!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Test timeout!")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi chạy test: {e}")
        return False

def main():
    """Hàm chính"""
    print("🚀 Kiểm tra môi trường đồ án DS317\n")

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
        print("✅ Tất cả kiểm tra đã pass!")
        print("\n🧪 Chạy test nhanh...")

        if run_quick_test():
            print("\n🎉 Môi trường sẵn sàng! Có thể chạy đồ án.")
            print("Chạy đầy đủ: python final/source-code/main_experiment.py --phase all")
            print("Chạy demo: python final/source-code/demo_app.py")
        else:
            print("\n⚠️  Test thất bại. Kiểm tra lại cấu hình.")
    else:
        print("❌ Một số kiểm tra thất bại. Vui lòng khắc phục trước khi chạy.")

if __name__ == "__main__":
    main()