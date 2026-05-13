"""
Network topology constants extracted from 4x4.net.xml
Generated from Eclipse SUMO netedit 1.26.0

Controlled intersections: J1, J2, J3, J6, J7, J8, J11, J12, J13, J16, J17, J18
Grid layout (3 rows x 4 cols):

  col0   col1   col2   col3
   J1     J6    J11    J16    <- row 0  (y ~= +3 to +7)
   J2     J7    J12    J17    <- row 1  (y ~= -37 to -47)
   J3     J8    J13    J18    <- row 2  (y ~= -87 to -97)

Boundary (dead-end) junctions (not controlled):
  Top    : J0(N of J1), J5(N of J6), J10(N of J11), J15(N of J16)
  Bottom : J4(S of J3), J9(S of J8), J14(S of J13), J19(S of J18)
  Left   : J20(W of J1), J22(W of J2), J24(W of J3)
  Right  : J21(E of J16), J23(E of J17), J25(E of J18)
"""

# ── Ordered list of controlled TLS nodes ────────────────────────────────────
CONTROLLED_TLS = ["J1", "J6", "J11", "J16",
                  "J2", "J7", "J12", "J17",
                  "J3", "J8", "J13", "J18"]

NUM_NODES   = len(CONTROLLED_TLS)   # 12
GRID_ROWS   = 3
GRID_COLS   = 4
NUM_ACTIONS = 4                     # phases 0-3 per junction

# ── Grid coordinates (row, col) for each TLS ────────────────────────────────
TLS_GRID_POS = {
    "J1":  (0, 0), "J6":  (0, 1), "J11": (0, 2), "J16": (0, 3),
    "J2":  (1, 0), "J7":  (1, 1), "J12": (1, 2), "J17": (1, 3),
    "J3":  (2, 0), "J8":  (2, 1), "J13": (2, 2), "J18": (2, 3),
}

# Node index in CONTROLLED_TLS list
TLS_INDEX = {tls: i for i, tls in enumerate(CONTROLLED_TLS)}

# ── Edges entering each controlled junction (from net.xml incLanes) ──────────
# Format: junction_id -> [incoming_edge_ids]
TLS_INCOMING_EDGES = {
    "J1":  ["E0",   "-E17", "-E1",  "E16"],
    "J6":  ["E4",   "-E18", "-E5",  "E17"],
    "J11": ["-E19", "-E9",  "E18",  "E8"],
    "J16": ["-E20", "-E13", "E19",  "E12"],
    "J2":  ["E1",   "-E22", "-E2",  "E21"],
    "J7":  ["E5",   "-E23", "-E6",  "E22"],
    "J12": ["-E24", "-E10", "E23",  "E9"],
    "J17": ["-E25", "-E14", "E24",  "E13"],
    "J3":  ["E2",   "-E27", "-E3",  "E26"],
    "J8":  ["-E28", "-E7",  "E27",  "E6"],
    "J13": ["-E29", "-E11", "E28",  "E10"],
    "J18": ["-E30", "-E15", "E29",  "E14"],
}

# ── Phase definitions (from tlLogic in net.xml) ──────────────────────────────
PHASE_DESCRIPTION = {
    0: "N-S straight+right green  (E0 / -E1 at J1-column; equivalent directions at others)",
    1: "N-S yellow transition",
    2: "E-W straight+right green  (E16/-E17 at J1; equivalent directions at others)",
    3: "E-W yellow transition",
}

# ── Movement IDs per junction ────────────────────────────────────────────────
TLS_MOVEMENT_IDS = {
    "J1": {
        "incoming": ["E0", "-E17", "-E1", "E16"],
        "link_groups": {
            "E0":   [0, 1, 2],
            "-E17": [3, 4, 5],
            "-E1":  [6, 7, 8],
            "E16":  [9, 10, 11],
        }
    },
}

# ── Neighbour map (from grid adjacency) ─────────────────────────────────────
def build_tls_neighbor_map():
    """
    Returns dict: tls_id -> list of neighbouring tls_ids
    Based on grid position (N/S/E/W adjacency only, no diagonals).
    """
    neighbor_map = {}
    for tls, (r, c) in TLS_GRID_POS.items():
        neighbors = []
        for tls2, (r2, c2) in TLS_GRID_POS.items():
            if tls2 != tls and abs(r - r2) + abs(c - c2) == 1:
                neighbors.append(tls2)
        neighbor_map[tls] = sorted(neighbors)
    return neighbor_map

TLS_NEIGHBOR_MAP = build_tls_neighbor_map()

# ── Connecting edges between TLS pairs ──────────────────────────────────────
INTER_TLS_EDGES = {
    "E17":  ("J1",  "J6"),    "-E17": ("J6",  "J1"),
    "E18":  ("J6",  "J11"),   "-E18": ("J11", "J6"),
    "E19":  ("J11", "J16"),   "-E19": ("J16", "J11"),
    "E1":   ("J1",  "J2"),    "-E1":  ("J2",  "J1"),
    "E5":   ("J6",  "J7"),    "-E5":  ("J7",  "J6"),
    "E9":   ("J11", "J12"),   "-E9":  ("J12", "J11"),
    "E13":  ("J16", "J17"),   "-E13": ("J17", "J16"),
    "E2":   ("J2",  "J3"),    "-E2":  ("J3",  "J2"),
    "E6":   ("J7",  "J8"),    "-E6":  ("J8",  "J7"),
    "E10":  ("J12", "J13"),   "-E10": ("J13", "J12"),
    "E14":  ("J17", "J18"),   "-E14": ("J18", "J17"),
    "E22":  ("J2",  "J7"),    "-E22": ("J7",  "J2"),
    "E23":  ("J7",  "J12"),   "-E23": ("J12", "J7"),
    "E24":  ("J12", "J17"),   "-E24": ("J17", "J12"),
    "E27":  ("J3",  "J8"),    "-E27": ("J8",  "J3"),
    "E28":  ("J8",  "J13"),   "-E28": ("J13", "J8"),
    "E29":  ("J13", "J18"),   "-E29": ("J18", "J13"),
}

# ── Traffic flow routes (from 4x4.rou.xml) ──────────────────────────────────
TRAFFIC_FLOWS = [
    {"id": "f_0",  "from": "E16", "to": "E20", "via": ["E17","E18","E19"],            "vph": 1800},
    {"id": "f_1",  "from": "E16", "to": "E25", "via": ["E1","E22","E23","E24"],        "vph": 1800},
    {"id": "f_2",  "from": "E16", "to": "E30", "via": ["E1","E2","E27","E28","E29"],   "vph": 1800},
    {"id": "f_3",  "from": "E16", "to": "E3",  "via": ["E1","E2"],                    "vph": 1800},
    {"id": "f_4",  "from": "E21", "to": "E20", "via": ["-E1","E17","E18","E19"],       "vph": 1800},
    {"id": "f_5",  "from": "E21", "to": "E25", "via": ["E22","E23","E24"],             "vph": 1800},
    {"id": "f_6",  "from": "E21", "to": "E30", "via": ["E2","E27","E28","E29"],        "vph": 1800},
    {"id": "f_7",  "from": "E21", "to": "E3",  "via": ["E2"],                         "vph": 1800},
    {"id": "f_8",  "from": "E26", "to": "E30", "via": ["E27","E28","E29"],             "vph": 1800},
    {"id": "f_9",  "from": "E26", "to": "E3",  "via": [],                             "vph": 1800},
    {"id": "f_10", "from": "E26", "to": "E25", "via": ["-E2","E22","E23","E24"],       "vph": 1800},
    {"id": "f_11", "from": "E26", "to": "E20", "via": ["-E2","-E1","E17","E18","E19"],"vph": 1800},
]
