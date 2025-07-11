import subprocess
import random

devmodes = ["siljunmode"]
#debugmode, siljunmode
normalization = input("normalization : ")
folders = ["01", "02", "03", "04", "05"]

for devmode in devmodes:
    for folder in folders:
        for _ in range(10):  
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