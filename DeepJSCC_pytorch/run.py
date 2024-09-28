import argparse
import os
from subprocess import run


def main():
    c_list = [4, 8, 24, 40]

    for c in c_list:
        run("python main.py --c %d --result_dir %s" % (c, c), shell=True)


def evaluation():
    dir = r".\weight"
    weights = [
        "vae_gan_weight_100.pth",
        "vae_gan_weight_200.pth",
        "vae_gan_weight_300.pth",
        "vae_gan_weight_400.pth",
        "vae_gan_weight_500.pth",
        "vae_gan_weight_600.pth",
    ]
    c = 24
    for w in weights:
        result_dir = "%s_c%d" % (w, c)
        weight_path = os.path.join(dir, w)
        run(
            "python evaluation.py --c %d --result_dir %s --weight %s"
            % (c, result_dir, weight_path)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    eval(args.name + "()")
