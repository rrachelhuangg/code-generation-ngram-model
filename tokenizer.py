import os
import sys
import re
import javalang

args     = sys.argv
num_args = len(args)

for arg in args:
    if arg[-4:] == "*.txt":
        filepath = arg[:-5]
        if os.path.isdir(filepath):
            for file in os.listdir(filepath):
                pass


    if re.match(arg, r"\.*.txt"):
        pass


