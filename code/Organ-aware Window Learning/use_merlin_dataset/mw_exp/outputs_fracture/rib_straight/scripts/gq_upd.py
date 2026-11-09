import sys
p = "GPU_QUEUE.md"; s = open(p).read(); lines = s.split("\n")
i = [k for k, l in enumerate(lines) if "rib_straight 肋骨拉直" in l]; assert len(i) == 1
add = sys.argv[1]
lines[i[0]] = lines[i[0]].rstrip() + " " + add
open(p, "w").write("\n".join(lines)); print("ok")
