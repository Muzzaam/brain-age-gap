import argparse, csv, random, time
ap = argparse.ArgumentParser()
ap.add_argument("-f", required=True)
ap.add_argument("-o", required=True)
a = ap.parse_args()
time.sleep(0.3)  # pretend SPM preprocessing takes a while
# fail on one scan to test failure handling
if "SUB999" in a.f:
    raise SystemExit(1)
with open(a.o, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["File", "brain.predicted.age"])
    w.writerow([a.f, round(random.uniform(68, 85), 2)])
