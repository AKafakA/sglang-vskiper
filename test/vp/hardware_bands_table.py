#!/usr/bin/env python3
"""Appendix table: the regime-switch K/V band per device, every entry the roofline rule's output (design.SERVED_DECODE_KV_BAND_BY_DEVICE
asserted against roofline.derived_kv_band at boot and by the execution-difference gate). Reads the tree's device_roofline.json and the
rule constants; emits one LaTeX row per device (peak bandwidth, V*, exit/enter band) plus the measured knee where the paper has one.

usage: hardware_bands_table.py --rows out.tex --macros out.tex [--knee NVIDIA_A100=13 ...] [--used NVIDIA_A100,NVIDIA_H100_HBM3,NVIDIA_RTX_A6000]
Runs without torch: imports roofline.py and design.py's band table by path.
"""
import argparse, ast, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); VP = os.path.join(HERE, "..", "..", "python", "sglang", "srt", "vpipe")
sys.path.insert(0, VP)
import roofline  # noqa: E402

NAMES = {"NVIDIA_A100": ("A100-SXM4-80GB (PCIe tuned)", "AHundred"), "NVIDIA_A100_40GB": ("A100-SXM4-40GB", "AHundredForty"),
         "NVIDIA_H100_HBM3": ("H100 80\\,GB HBM3", "Hundred"), "NVIDIA_H100_NVL": ("H100 NVL", "HundredNvl"),
         "NVIDIA_RTX_A6000": ("RTX A6000 48\\,GB GDDR6", "Asix"), "NVIDIA_L40S": ("L40S 48\\,GB GDDR6", "Lforty"),
         "NVIDIA_RTX_5880_Ada_Generation": ("RTX 5880 Ada 48\\,GB", "Fivethousand")}


def band_table() -> dict:
    src = open(os.path.join(VP, "design.py")).read()
    m = re.search(r"SERVED_DECODE_KV_BAND_BY_DEVICE[^=]*=\s*(\{.*?\n\})", src, re.S)
    body = re.sub(r"#.*", "", m.group(1))
    return ast.literal_eval(body)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--macros", required=True)
    ap.add_argument("--knee", action="append", default=[]); ap.add_argument("--used", default="")
    a = ap.parse_args(); knees = dict(x.split("=") for x in a.knee); used = set(a.used.split(",")) if a.used else set()
    table = band_table(); rows, macros = [], []
    for key, (exit_t, enter_t) in table.items():
        tflops, bw = roofline.device_peaks(key); v = roofline.kv_crossover_tokens(key); rule = roofline.derived_kv_band(key)
        if rule != (exit_t, enter_t): sys.exit(f"FATAL {key}: table band {(exit_t, enter_t)} is not the rule's {rule}")
        name, mac = NAMES.get(key, (key.replace("_", " "), re.sub(r"[^A-Za-z]", "", key)))
        knee = knees.get(key, "--"); mark = "yes" if key in used else ""
        rows.append(f"{name} & {bw:,.0f} & {tflops:,.0f} & {v/1e3:.1f}k & {exit_t//1000}k / {enter_t//1000}k & {knee} & {mark} \\\\")
        macros += [f"\\newcommand{{\\vpBand{mac}Exit}}{{{exit_t//1000}}}", f"\\newcommand{{\\vpBand{mac}Enter}}{{{enter_t//1000}}}", f"\\newcommand{{\\vpBand{mac}Vstar}}{{{v/1e3:.1f}}}", f"\\newcommand{{\\vpBand{mac}Bw}}{{{bw:,.0f}}}"]
    macros += [f"\\newcommand{{\\vpBandTauMs}}{{{roofline.DECODE_BODY_TAX_MS}}}", f"\\newcommand{{\\vpBandSkipRatio}}{{{roofline.DESIGN_DECODE_SKIP_RATIO}}}", f"\\newcommand{{\\vpBandEnterFactor}}{{{roofline.KV_BAND_ENTER_FACTOR}}}"]
    open(a.rows, "w").write("\n".join(rows) + "\n"); open(a.macros, "w").write("\n".join(macros) + "\n")
    print(f"{len(rows)} device rows -> {a.rows}; {len(macros)} macros -> {a.macros}")


if __name__ == "__main__": main()
