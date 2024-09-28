import os


class Logging:
    def __init__(self, dir="", name="log", mode="a", info="", heads=[]):
        file = "%s.txt" % name
        self.path = os.path.join(dir, file)
        self.mode = mode
        if os.path.isfile(self.path):
            print(f"{file} already exists.")
            yn = input("Do you want to override %s? (Y/N):" % file)
            if yn == "y" or yn == "Y":
                os.remove(self.path)
            else:
                raise Exception(f"Please delete the file: {self.path}")

        info = "\n".join(f"{i}" for i in info) + "\n"
        head = ",".join(f"{h}" for h in heads) + "\n"

        with open(self.path, mode=self.mode) as f:
            f.write(info)
            f.write(head)

    def __call__(self, contents):
        content = ",".join(f"{c}" for c in contents) + "\n"
        with open(self.path, mode=self.mode) as f:
            f.write(content)
