import subprocess
import random
import itertools

# 🎯 사용자에게 normalization 입력 받기 (검증 포함)
while True:
    normalization = input("normalization (true/false) : ").strip().lower()
    if normalization in ["true", "false"]:
        break
    else:
        print("❌ 'true' 또는 'false'만 입력해주세요.")

# 설정
devmode = "siljunmode"  # 또는 'debugmode'
folders = ["01", "02", "03", "04", "05"]
modal_options = ["t", "v", "a"]

# 🎲 실행 목록 생성: 각 폴더당 10개 (모달 무작위)
execution_list = []
for folder in folders:
    for _ in range(10):
        mod = random.choice(modal_options)
        seed = random.randint(1, 100000)
        execution_list.append({
            "folder": folder,
            "modals": mod,
            "aux_classifier": mod,
            "seed": seed,
            "devmode": devmode,
            "normalization": normalization
        })

# 🔀 실행 순서 무작위로 섞기
random.shuffle(execution_list)

# 🚀 실행
for config in execution_list:
    command = [
        "python", "train_test.py",
        "--folder", config["folder"],
        "--seed", str(config["seed"]),
        "--devmode", config["devmode"],
        "--modals", config["modals"],
        "--aux_classifier", config["aux_classifier"],
        "--normalization", config["normalization"]
    ]
    print("Running:", " ".join(command))
    subprocess.run(command)
