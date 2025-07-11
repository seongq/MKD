import subprocess
import random
import itertools
normalizations = ['true', 'false']



# 설정
devmode = "siljunmode"  # 또는 'debugmode', 'siljunmode'
folders = ["01", "02", "03", "04", "05"]
modal_options = ["tva"]

# 🎲 실행 목록 생성: 각 폴더당 10개 (모달 무작위)
execution_list = []
for folder in folders:
    for _ in range(100):
        mod = random.choice(modal_options)
        normalization = random.choice(normalizations)
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
