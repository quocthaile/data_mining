import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("1", "stage_1_generate_ground_truth.py", "Generate full-course ground truth labels"),
    ("2", "stage_2_time_window_features.py", "Extract early time-window features"),
    ("3", "stage_3_split_and_smote.py", "Split, encode, scale, and balance training data"),
    ("4", "stage_4_model_training_eval.py", "Train, evaluate, and export deployment model"),
    ("5", "stage_5_explain_model_xai.py", "Generate expected-result summary and XAI outputs"),
]


def _write_overrides(overrides: dict):
    path = BASE_DIR / "runtime_overrides.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)


def run_step(script_name: str, env_overrides: dict | None = None) -> None:
    script_path = BASE_DIR / script_name
    env = None
    if env_overrides:
        _write_overrides(env_overrides)
    result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def interactive_menu():
    print("Simple interactive pipeline manager")
    overrides = {}
    while True:
        print("\nAvailable steps:")
        for sid, fname, desc in STEPS:
            print(f" {sid}. {desc}    ({fname})")
        print(" a. Run all steps")
        print(" r. Run a range (e.g. 1-3)")
        print(" p. Set pipeline parameter override (KEY=JSON_VALUE)")
        print(" s. Show current overrides")
        print(" q. Quit")
        choice = input("Select action: ").strip()
        if choice == "q":
            return
        if choice == "a":
            for sid, fname, _ in STEPS:
                print(f"Running step {sid} -> {fname}")
                run_step(fname, overrides)
            continue
        if choice == "r":
            rng = input("Enter range from-to (e.g. 1-3): ").strip()
            if "-" not in rng:
                print("Invalid range")
                continue
            a, b = rng.split("-", 1)
            try:
                a = int(a); b = int(b)
            except Exception:
                print("Invalid numbers")
                continue
            for sid, fname, _ in STEPS:
                if a <= int(sid) <= b:
                    print(f"Running step {sid} -> {fname}")
                    run_step(fname, overrides)
            continue
        if choice == "p":
            pair = input("Enter KEY=JSON_VALUE (e.g. TRAIN_TARGET_TOTAL_SAMPLES=60000): ").strip()
            if "=" not in pair:
                print("Invalid input")
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            try:
                parsed = json.loads(v)
            except Exception:
                parsed = v
            overrides[k] = parsed
            print(f"Set override {k} -> {parsed}")
            continue
        if choice == "s":
            print(json.dumps(overrides, indent=2, ensure_ascii=False))
            continue
        if choice.isdigit():
            found = False
            for sid, fname, _ in STEPS:
                if sid == choice:
                    found = True
                    run_step(fname, overrides)
            if not found:
                print("Unknown step")
            continue
        print("Unknown option")


def main():
    parser = argparse.ArgumentParser(description="Menu-driven pipeline manager")
    parser.add_argument("--from-step", help="First step to run", choices=[s[0] for s in STEPS])
    parser.add_argument("--to-step", help="Last step to run", choices=[s[0] for s in STEPS])
    parser.add_argument("--param", help="Override param as KEY=JSON (can be repeated)", action="append")
    parser.add_argument("--menu", help="Interactive menu", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.param:
        for p in args.param:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            try:
                overrides[k] = json.loads(v)
            except Exception:
                overrides[k] = v

    if args.menu:
        interactive_menu()
        return

    if args.from_step and args.to_step:
        a = int(args.from_step); b = int(args.to_step)
        for sid, fname, _ in STEPS:
            if a <= int(sid) <= b:
                print(f"Running step {sid} -> {fname}")
                run_step(fname, overrides)
        return

    for sid, fname, _ in STEPS:
        print(f"Running step {sid} -> {fname}")
        run_step(fname, overrides)


if __name__ == "__main__":
    main()
