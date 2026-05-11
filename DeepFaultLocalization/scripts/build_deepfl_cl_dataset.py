#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 DeepFL_CL 数据集：
- 每个 bug 版本使用其原始 Test/TestLabel 作为该版本任务数据；
- 同步写为 Train/TrainLabel 与 Test/TestLabel（用于 continual 的 prequential：先评估再更新）；
- 自动生成 groupfile/<sub>/<ver>/traidata.txt.group（单组，大小为样本数）。
"""
from __future__ import print_function

import argparse
import csv
import os
import shutil
MAX_VER = {
    "Chart": 26,
    "Lang": 65,
    "Math": 106,
    "Time": 27,
    "Closure": 133,
    "Mockito": 38,
}

SUBJECTS = ("Chart", "Lang", "Math", "Time", "Closure", "Mockito")


def _rows(path):
    with open(path) as f:
        return sum(1 for _ in csv.reader(f))


def _safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    pa = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="从 DeepFL 构建无未来信息的 DeepFL_CL（按每版本 Test 作为该版本任务集）。",
    )
    pa.add_argument("--data_root", default=".", help="数据根目录（包含 DeepFL/）")
    pa.add_argument("--src_tech", default="DeepFL", help="源技术目录名")
    pa.add_argument("--dst_tech", default="DeepFL_CL", help="目标技术目录名")
    args = pa.parse_args()

    src_root = os.path.join(args.data_root, args.src_tech)
    dst_root = os.path.join(args.data_root, args.dst_tech)
    os.makedirs(dst_root, exist_ok=True)

    converted = 0
    for sub in SUBJECTS:
        vmax = MAX_VER[sub]
        for ver in range(1, vmax + 1):
            sdir = os.path.join(src_root, sub, str(ver))
            t_src = os.path.join(sdir, "Test.csv")
            yl_src = os.path.join(sdir, "TestLabel.csv")
            if not (os.path.isfile(t_src) and os.path.isfile(yl_src)):
                continue

            ddir = os.path.join(dst_root, sub, str(ver))
            _safe_copy(t_src, os.path.join(ddir, "Train.csv"))
            _safe_copy(yl_src, os.path.join(ddir, "TrainLabel.csv"))
            _safe_copy(t_src, os.path.join(ddir, "Test.csv"))
            _safe_copy(yl_src, os.path.join(ddir, "TestLabel.csv"))

            n = _rows(t_src)
            gpath = os.path.join(dst_root, "groupfile", sub, str(ver), "traidata.txt.group")
            os.makedirs(os.path.dirname(gpath), exist_ok=True)
            with open(gpath, "w") as gf:
                gf.write(str(int(n)) + "\n")

            converted += 1

    print("DeepFL_CL build done. converted_versions=%d" % converted)
    print("dst=%s" % dst_root)


if __name__ == "__main__":
    main()

