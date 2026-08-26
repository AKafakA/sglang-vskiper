# The STOCK posture: no vpipe knobs at all. Run with VP_GATE_STOCK=1 so the
# driver also strips its own FD-weights/helper remaps — the served process
# must have zero SGLANG_FD_*/SGLANG_VP_* env. Used by the per-family
# stock-inertness gate (seamed tree vs upstream tree, text sha equality).
