import subprocess
import random

devmodes = ["debugmode", "siljunmode"]
normalizations = ["true", "false"]
folders = ["01", "02", "03", "04", "05"]

for devmode in devmodes:
    for normalization in normalizations:
        for folder in folders:
            for _ in range(10):  # 각 세팅당 folder마다 10번씩 실행
                seed = random.randint(1, 100000)
                command = [
                    "python", "train_test.py",
                    "--devmode", devmode,
                    "--normalization", normalization,
                    "--seed", str(seed),
                    "--folder", folder
                ]
                print("Running:", " ".join(command))
                subprocess.run(command)