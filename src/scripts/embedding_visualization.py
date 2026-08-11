"""
Embed all tables for every agent in an input file and visualize with UMAP + HDBSCAN and UMAP + EVoC.

Reads agent list from the input file, fetches embeddings (cached), and writes:
  - embedding_clusters_umap_hdbscan.png
  - embedding_clusters_umap_evoc.png
  - embedding_clusters_evoc.png             (EVoC node embedding 2D layout + EVoC clusters)
  - embedding_clusters_fixed_hdbscan.png    (fixed lon/lat coords, no UMAP)
  - embedding_clusters_fixed_evoc.png       (fixed lon/lat coords, no UMAP)
  - embedding_clusters.md
into the output directory.
"""

import argparse
import asyncio
import colorsys
import concurrent.futures
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import evoc
import matplotlib
import matplotlib.artist
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
from evoc.clustering import (
    knn_graph,
    label_propagation_init,
    neighbor_graph_matrix,
)
from evoc.clustering import (
    node_embedding as evoc_node_embedding,
)

matplotlib.use("Agg")
import fast_hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap

import qcp_adaptive_termination.agent_registry as agent_registry
from qcp_adaptive_termination.models import AgentRegistryType, DataAgent
from qcp_adaptive_termination.observability import context
from qcp_adaptive_termination.pipeline import load_config, load_input

MIN_SAMPLES = 1
HDBSCAN_MIN_CLUSTER_SIZE = 5

# Well-separated hues for top-level EVoC clusters
_EVOC_BASE_HUES = [0.00, 0.33, 0.66, 0.08, 0.50, 0.83, 0.16, 0.75, 0.41, 0.58, 0.25, 0.91]

CATEGORICAL_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#b55c1a",
    "#888888",
]
NOISE_COLOR = "#aaaaaa"

# fmt: off
# (longitude, latitude) for every possible DB stem.
# Countries: approximate geographic center.
# Groups/blocs: centroid of member countries or region center.
# Longitude [-180, 180], Latitude [-90, 90].
_DB_GEO: dict[str, tuple[float, float]] = {
    # ── Continents / major regions ────────────────────────────────────────────
    "africa":                        ( 20.0,   0.0),
    "africa_region":                 ( 20.0,   0.0),
    "americas":                      (-80.0,  15.0),
    "americas_region":               (-80.0,  15.0),
    "asia":                          ( 90.0,  35.0),
    "asia_region":                   ( 90.0,  35.0),
    "europe":                        ( 15.0,  50.0),
    "oceania":                       (140.0, -25.0),
    "oceania_region":                (140.0, -25.0),
    # ── UN subregions ─────────────────────────────────────────────────────────
    "northern_europe":               ( 15.0,  62.0),
    "western_europe":                (  5.0,  48.0),
    "central_europe":                ( 15.0,  49.0),
    "eastern_europe":                ( 30.0,  52.0),
    "southern_europe":               ( 15.0,  41.0),
    "north_africa":                  ( 17.0,  25.0),
    "west_africa":                   (  0.0,   8.0),
    "central_africa":                ( 22.0,   5.0),
    "east_africa":                   ( 38.0,   0.0),
    "southern_africa":               ( 25.0, -25.0),
    "subsaharan_africa":             ( 20.0, -10.0),
    "north_america":                 (-100.0,  45.0),
    "central_america":               (-85.0,  12.0),
    "caribbean":                     (-73.0,  17.0),
    "south_america":                 (-55.0, -15.0),
    "latin_america":                 (-60.0, -10.0),
    "western_asia":                  ( 45.0,  32.0),
    "central_asia":                  ( 63.0,  45.0),
    "east_asia":                     (115.0,  35.0),
    "south_asia":                    ( 78.0,  22.0),
    "southeast_asia":                (110.0,   5.0),
    "middle_east":                   ( 45.0,  28.0),
    "oceania_ecozone":               (155.0, -20.0),
    "australia_new_zealand":         (155.0, -32.0),
    "pacific_islands":               (170.0, -10.0),
    # ── WWF Ecozones ──────────────────────────────────────────────────────────
    "nearctic":                      (-100.0,  50.0),
    "palearctic":                    (  60.0,  55.0),
    "afrotropic":                    (  20.0, -10.0),
    "indomalaya":                    (  95.0,  15.0),
    "australasia":                   ( 145.0, -30.0),
    "neotropic":                     ( -60.0, -10.0),
    # ── Climate regions ───────────────────────────────────────────────────────
    "tropical":                      (  20.0,   5.0),
    "arid":                          (  40.0,  25.0),
    "temperate":                     (  20.0,  45.0),
    "continental":                   (  30.0,  55.0),
    "polar":                         (   0.0,  80.0),
    # ── Global North/South ────────────────────────────────────────────────────
    "global_north":                  (  10.0,  55.0),
    "global_south":                  (  20.0, -10.0),
    # ── Political / economic blocs (centroid of members) ─────────────────────
    "european_union":                ( 15.0,  52.0),
    "nato":                          ( -5.0,  50.0),
    "schengen":                      ( 12.0,  50.0),
    "oecd":                          ( 15.0,  45.0),
    "g7":                            (  0.0,  45.0),
    "g20":                           ( 20.0,  25.0),
    "brics":                         ( 50.0,  20.0),
    "asean":                         (110.0,   5.0),
    "arab_league":                   ( 40.0,  25.0),
    "opec":                          ( 30.0,  15.0),
    "mercosur":                      (-55.0, -25.0),
    "commonwealth":                  ( 20.0,  10.0),
    "eea":                           ( 10.0,  57.0),
    "five_eyes":                     (-30.0,  40.0),  # US/UK/CA/AU/NZ centroid
    "eu_candidate":                  ( 22.0,  43.0),
    "african_union":                 ( 20.0,   5.0),
    "visegrad":                      ( 19.0,  50.0),
    "balkans":                       ( 21.0,  42.0),
    "caucasus":                      ( 44.0,  42.0),
    "gulf_states":                   ( 51.0,  24.0),
    "sahel":                         (  5.0,  14.0),
    "maghreb":                       (  5.0,  32.0),
    "nordic_countries":              ( 18.0,  63.0),
    "scandinavia":                   ( 16.0,  62.0),
    "baltic_states":                 ( 24.0,  57.0),
    "dach":                          ( 13.0,  48.0),
    "benelux":                       (  4.5,  51.0),
    "british_isles":                 ( -3.0,  53.0),
    "slavic_countries":              ( 25.0,  52.0),
    "european_microstates":          ( 12.0,  46.0),
    "successor_states_of_yugoslavia":( 20.0,  44.0),
    "kingdom_of_denmark":            ( 12.0,  56.0),
    "kingdom_of_the_netherlands":    (  4.9,  52.0),
    "small_island_states":           (170.0, -10.0),
    # ── Language groups ───────────────────────────────────────────────────────
    "anglophone":                    (-20.0,  30.0),
    "sinophone":                     (115.0,  30.0),
    "lusophone":                     (-15.0,  10.0),
    "hispanophone":                  (-65.0,  10.0),
    "francophonie":                  (  5.0,  20.0),
    # ── Religion ──────────────────────────────────────────────────────────────
    "christian":                     ( 15.0,  35.0),
    "muslim":                        ( 45.0,  25.0),
    "hindu":                         ( 78.0,  22.0),
    "buddhist":                      (100.0,  25.0),
    "jewish":                        ( 35.0,  32.0),
    # ── Geographic characteristics ────────────────────────────────────────────
    "landlocked":                    ( 30.0,  40.0),
    "double_landlocked":             ( 15.0,  47.0),
    "island_nation":                 (140.0,  10.0),
    "nuclear_weapons_state":         ( 30.0,  40.0),
    # ── UTC timezone groups (lon = UTC_offset * 15, lat = mean of members) ────
    "utc_minus_11":                  (-165.0, -15.0),
    "utc_minus_10":                  (-150.0,  20.0),
    "utc_minus_8":                   (-120.0,  45.0),
    "utc_minus_7":                   (-105.0,  40.0),
    "utc_minus_6":                   ( -90.0,  15.0),
    "utc_minus_5":                   ( -75.0,  10.0),
    "utc_minus_4":                   ( -60.0,  15.0),
    "utc_minus_3":                   ( -50.0, -15.0),
    "utc_minus_1":                   ( -15.0,  15.0),
    "utc_0":                         (   0.0,  30.0),
    "utc_plus_1":                    (  15.0,  30.0),
    "utc_plus_2":                    (  30.0,  30.0),
    "utc_plus_3":                    (  45.0,  25.0),
    "utc_plus_3_30":                 (  52.5,  32.0),
    "utc_plus_4":                    (  60.0,  30.0),
    "utc_plus_4_30":                 (  65.0,  33.0),
    "utc_plus_5":                    (  75.0,  30.0),
    "utc_plus_5_30":                 (  78.0,  20.0),
    "utc_plus_5_45":                 (  84.0,  28.0),
    "utc_plus_6":                    (  90.0,  25.0),
    "utc_plus_6_30":                 (  96.0,  20.0),
    "utc_plus_7":                    ( 105.0,  15.0),
    "utc_plus_8":                    ( 115.0,  25.0),
    "utc_plus_9":                    ( 135.0,  35.0),
    "utc_plus_10":                   ( 145.0, -25.0),
    "utc_plus_11":                   ( 160.0, -15.0),
    "utc_plus_12":                   ( 175.0, -20.0),
    "utc_plus_13":                   ( 178.0, -20.0),
    # ── Countries (geographic center) ─────────────────────────────────────────
    "afghanistan":                   (  67.7,  33.9),
    "albania":                       (  20.2,  41.2),
    "algeria":                       (   3.0,  28.0),
    "andorra":                       (   1.6,  42.5),
    "angola":                        (  18.5, -12.5),
    "anguilla":                      ( -63.1,  18.2),
    "antigua_and_barbuda":           ( -61.8,  17.1),
    "argentina":                     ( -64.0, -34.0),
    "armenia":                       (  45.0,  40.0),
    "aruba":                         ( -70.0,  12.5),
    "australia":                     ( 134.0, -27.0),
    "austria":                       (  14.5,  47.5),
    "azerbaijan":                    (  47.6,  40.4),
    "bahrain":                       (  50.6,  26.0),
    "bangladesh":                    (  90.4,  23.7),
    "barbados":                      ( -59.6,  13.2),
    "belarus":                       (  28.0,  53.0),
    "belgium":                       (   4.5,  50.5),
    "belize":                        ( -88.8,  17.2),
    "benin":                         (   2.3,   9.3),
    "bermuda":                       ( -64.8,  32.3),
    "bhutan":                        (  90.4,  27.5),
    "bolivia":                       ( -65.0, -17.0),
    "bosnia_and_herzegovina":        (  17.8,  44.2),
    "botswana":                      (  24.7, -22.3),
    "brazil":                        ( -51.9, -14.2),
    "brunei":                        ( 114.7,   4.5),
    "bulgaria":                      (  25.0,  43.0),
    "burkina_faso":                  (  -2.0,  12.4),
    "burundi":                       (  29.9,  -3.4),
    "cambodia":                      ( 105.0,  12.6),
    "cameroon":                      (  12.4,   3.9),
    "canada":                        ( -96.0,  60.0),
    "cape_verde":                    ( -24.0,  16.0),
    "cayman_islands":                ( -81.2,  19.3),
    "central_african_republic":      (  20.9,   7.0),
    "chad":                          (  18.7,  15.5),
    "chile":                         ( -71.5, -33.5),
    "colombia":                      ( -74.1,   4.6),
    "comoros":                       (  43.4, -11.7),
    "cook_islands":                  (-159.8, -21.2),
    "costa_rica":                    ( -84.1,   9.9),
    "croatia":                       (  15.2,  45.1),
    "cuba":                          ( -79.5,  22.0),
    "cura_ao":                       ( -69.0,  12.2),
    "cyprus":                        (  33.4,  35.1),
    "czech_republic":                (  15.5,  49.8),
    "democratic_republic_of_the_congo": ( 23.7,  -2.9),
    "denmark":                       (  10.0,  56.0),
    "djibouti":                      (  43.1,  11.6),
    "dominica":                      ( -61.4,  15.4),
    "dominican_republic":            ( -70.2,  19.0),
    "ecuador":                       ( -78.5,  -1.8),
    "egypt":                         (  30.1,  26.8),
    "el_salvador":                   ( -88.9,  13.8),
    "equatorial_guinea":             (   8.7,   1.7),
    "eritrea":                       (  39.8,  15.2),
    "estonia":                       (  25.0,  59.0),
    "eswatini":                      (  31.5, -26.5),
    "ethiopia":                      (  40.5,   9.1),
    "falkland_islands":              ( -59.0, -51.8),
    "faroe_islands":                 (  -6.9,  62.0),
    "federated_states_of_micronesia":( 158.2,   6.9),
    "fiji":                          ( 178.1,  -17.7),
    "finland":                       (  26.0,  64.0),
    "france":                        (   2.4,  46.6),
    "gabon":                         (  11.6,  -0.8),
    "georgia":                       (  43.3,  42.3),
    "germany":                       (  10.4,  51.2),
    "ghana":                         (  -1.0,   8.0),
    "greece":                        (  22.0,  39.1),
    "greenland":                     ( -42.0,  72.0),
    "grenada":                       ( -61.7,  12.1),
    "guernsey":                      (  -2.6,  49.5),
    "guatemala":                     ( -90.2,  15.8),
    "guinea":                        ( -11.3,  11.0),
    "guinea_bissau":                 ( -15.2,  12.0),
    "guyana":                        ( -59.0,   5.0),
    "haiti":                         ( -72.3,  18.9),
    "honduras":                      ( -86.5,  15.2),
    "hong_kong":                     ( 114.2,  22.3),
    "hungary":                       (  19.5,  47.2),
    "iceland":                       ( -19.0,  65.0),
    "india":                         (  78.9,  20.6),
    "indonesia":                     ( 113.9,  -0.8),
    "iran":                          (  53.7,  32.4),
    "iraq":                          (  43.7,  33.2),
    "ireland":                       (  -8.2,  53.2),
    "isle_of_man":                   (  -4.5,  54.2),
    "israel":                        (  34.9,  31.5),
    "italy":                         (  12.6,  42.5),
    "ivory_coast":                   (  -5.5,   7.5),
    "jamaica":                       ( -77.3,  18.1),
    "japan":                         ( 138.3,  36.2),
    "jersey":                        (  -2.1,  49.2),
    "jordan":                        (  36.2,  31.2),
    "kazakhstan":                    (  67.0,  48.0),
    "kenya":                         (  37.9,   0.0),
    "kiribati":                      ( 173.0,   1.4),
    "kosovo":                        (  20.9,  42.6),
    "kuwait":                        (  47.5,  29.3),
    "kyrgyzstan":                    (  74.8,  41.2),
    "laos":                          ( 103.9,  18.2),
    "latvia":                        (  25.0,  57.0),
    "lebanon":                       (  35.9,  33.9),
    "lesotho":                       (  28.3, -29.6),
    "liberia":                       (  -9.4,   6.4),
    "libya":                         (  17.2,  27.0),
    "liechtenstein":                 (   9.6,  47.2),
    "lithuania":                     (  24.0,  56.0),
    "luxembourg":                    (   6.1,  49.8),
    "madagascar":                    (  46.9, -20.3),
    "malawi":                        (  34.3, -13.3),
    "malaysia":                      ( 109.7,   3.1),
    "maldives":                      (  73.2,   3.2),
    "mali":                          (  -2.0,  18.0),
    "malta":                         (  14.4,  35.9),
    "marshall_islands":              ( 168.7,   7.1),
    "mauritania":                    ( -11.8,  20.3),
    "mauritius":                     (  57.6, -20.2),
    "mexico":                        ( -102.6,  23.6),
    "moldova":                       (  28.5,  47.4),
    "monaco":                        (   7.4,  43.7),
    "mongolia":                      ( 103.8,  46.8),
    "montenegro":                    (  19.4,  42.7),
    "montserrat":                    ( -62.2,  16.7),
    "morocco":                       (  -7.1,  31.8),
    "mozambique":                    (  35.5, -18.7),
    "myanmar":                       (  96.5,  17.1),
    "namibia":                       (  18.5, -22.0),
    "naoero":                        ( 166.9,  -0.5),
    "nepal":                         (  84.1,  28.4),
    "netherlands":                   (   5.3,  52.3),
    "new_zealand":                   ( 174.9, -40.9),
    "nicaragua":                     ( -85.2,  13.0),
    "niger":                         (   8.1,  16.1),
    "nigeria":                       (   8.7,   9.1),
    "niue":                          (-169.9, -19.1),
    "north_korea":                   ( 127.5,  40.3),
    "north_macedonia":               (  21.7,  41.6),
    "northern_cyprus":               (  33.4,  35.2),
    "norway":                        (   8.5,  60.5),
    "ohio":                          ( -82.9,  40.4),
    "oman":                          (  57.5,  21.5),
    "pakistan":                      (  69.3,  30.4),
    "palau":                         ( 134.5,   7.5),
    "palestine":                     (  35.3,  31.9),
    "panama":                        ( -80.2,   8.9),
    "papua_new_guinea":              ( 145.0,  -6.3),
    "paraguay":                      ( -58.4, -23.4),
    "people_s_republic_of_china":    ( 104.2,  35.9),
    "peru":                          ( -75.0,  -9.2),
    "philippines":                   ( 122.9,  12.9),
    "poland":                        (  20.0,  52.0),
    "portugal":                      (  -8.2,  39.6),
    "qatar":                         (  51.2,  25.4),
    "republic_of_the_congo":         (  15.8,  -0.8),
    "romania":                       (  25.0,  45.9),
    "russia":                        (  95.0,  60.0),
    "rwanda":                        (  29.9,  -1.9),
    "s_o_tom_and_pr_ncipe":          (   6.7,   0.2),
    "saint_helena_ascension_and_tristan_da_cunha": ( -5.7, -15.9),
    "saint_kitts_and_nevis":         ( -62.7,  17.3),
    "saint_lucia":                   ( -60.9,  13.9),
    "saint_vincent_and_the_grenadines": ( -61.2,  13.3),
    "samoa":                         (-172.5, -13.8),
    "san_marino":                    (  12.5,  43.9),
    "saudi_arabia":                  (  45.1,  24.7),
    "senegal":                       ( -14.5,  14.5),
    "serbia":                        (  21.0,  44.0),
    "seychelles":                    (  55.5,  -4.7),
    "sierra_leone":                  ( -11.8,   8.5),
    "singapore":                     ( 103.8,   1.3),
    "sint_maarten":                  ( -63.1,  18.0),
    "slovakia":                      (  19.7,  48.7),
    "slovenia":                      (  14.9,  46.1),
    "solomon_islands":               ( 160.2,  -9.6),
    "somalia":                       (  46.2,   5.2),
    "somaliland":                    (  46.2,   9.5),
    "south_africa":                  (  25.1, -29.0),
    "south_korea":                   ( 127.8,  36.0),
    "south_sudan":                   (  31.3,   7.9),
    "spain":                         (  -3.7,  40.4),
    "sri_lanka":                     (  80.8,   7.9),
    "sudan":                         (  29.9,  15.6),
    "suriname":                      ( -56.1,   3.9),
    "sweden":                        (  18.6,  59.3),
    "switzerland":                   (   8.2,  46.8),
    "syria":                         (  38.5,  35.0),
    "taiwan":                        ( 121.0,  23.7),
    "tajikistan":                    (  71.3,  38.9),
    "tanzania":                      (  35.0,  -6.4),
    "the_bahamas":                   ( -77.4,  24.3),
    "the_gambia":                    ( -15.3,  13.4),
    "thailand":                      ( 101.0,  15.5),
    "thuringia":                     (  11.0,  51.0),
    "timor_leste":                   ( 125.7,  -8.9),
    "togo":                          (   1.2,   8.6),
    "tonga":                         (-175.2, -21.2),
    "trinidad_and_tobago":           ( -61.2,  10.7),
    "tunisia":                       (   9.5,  33.9),
    "turkey":                        (  35.2,  39.0),
    "turkmenistan":                  (  58.4,  40.5),
    "tuvalu":                        ( 179.2,  -8.5),
    "uganda":                        (  32.3,   1.4),
    "ukraine":                       (  31.2,  49.0),
    "united_arab_emirates":          (  53.8,  23.4),
    "united_kingdom":                (  -3.4,  54.7),
    "united_states":                 ( -99.1,  39.4),
    "united_states_virgin_islands":  ( -64.9,  18.3),
    "uruguay":                       ( -56.2, -32.5),
    "uzbekistan":                    (  63.1,  41.4),
    "vanuatu":                       ( 167.0, -16.0),
    "vatican_city":                  (  12.5,  41.9),
    "venezuela":                     ( -66.6,   6.4),
    "vietnam":                       ( 106.3,  16.6),
    "west_bank":                     (  35.3,  32.0),
    "yemen":                         (  47.6,  15.6),
    "zambia":                        (  27.8, -13.1),
    "zimbabwe":                      (  29.9, -20.0),
}
# fmt: on

_SVG_PATH = Path(__file__).parent / "Continents.svg"


def _color_for_label(label: int) -> str:
    if label == -1:
        return NOISE_COLOR
    return CATEGORICAL_COLORS[label % (len(CATEGORICAL_COLORS) - 1)]


def _build_evoc_color_map(cluster_layers: list[np.ndarray], labels: np.ndarray) -> dict[int, str]:
    """
    Build a fine-cluster-id -> hex color map that encodes hierarchy:
    top-level clusters (labels_) get distinct hues; sub-clusters within each
    top-level cluster get lighter/darker shades of that hue.
    """
    fine_labels = cluster_layers[0] if len(cluster_layers) > 1 else labels
    top_labels = labels

    # map fine cluster id -> top cluster id via per-point co-occurrence
    fine_to_top: dict[int, int] = {}
    for fine, top in zip(fine_labels, top_labels, strict=True):
        fine, top = int(fine), int(top)
        if fine >= 0:
            fine_to_top[fine] = top

    top_to_fines: dict[int, list[int]] = defaultdict(list)
    for fc, tc in sorted(fine_to_top.items()):
        top_to_fines[tc].append(fc)

    top_clusters = sorted(set(int(x) for x in top_labels if x >= 0))
    color_map: dict[int, str] = {}
    for tc in top_clusters:
        hue = _EVOC_BASE_HUES[tc % len(_EVOC_BASE_HUES)]
        fines = top_to_fines.get(tc, [])
        n = len(fines)
        for j, fc in enumerate(fines):
            # spread lightness 0.35 (dark) .. 0.62 (light) so shades stay distinct
            lightness = 0.35 + (j / max(n - 1, 1)) * 0.27 if n > 1 else 0.48
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.70)
            color_map[fc] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return color_map


def _db_label(agent: DataAgent) -> str:
    return Path(agent.path).stem


def _geo_coord(stem: str) -> tuple[float, float] | None:
    """Return (lon, lat) for a DB stem, or None if not in _DB_GEO."""
    return _DB_GEO.get(stem)


def _parse_svg_paths() -> list[matplotlib.path.Path]:
    """
    Parse Continents.svg and return matplotlib Path objects in SVG viewBox coordinates.
    Paths in Continents.svg are already in viewBox space (0 0 2752.766 1537.631) with no
    group transforms, so coordinates are used as-is.
    """
    if not _SVG_PATH.exists():
        return []

    tree = ET.parse(_SVG_PATH)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    paths: list[matplotlib.path.Path] = []

    for path_el in list(root.findall(".//svg:path", ns)) + list(root.findall(".//path")):
        if path_el.get("transform"):
            continue
        classes = path_el.get("class", "").split()
        if "ocean" in classes or "lake" in classes:
            continue
        d = path_el.get("d", "")
        if not d:
            continue
        try:
            mpl_path = _svg_d_to_mpl_path(d)
            if mpl_path is not None:
                paths.append(mpl_path)
        except Exception:
            continue

    return paths


def _svg_d_to_mpl_path(d: str) -> matplotlib.path.Path | None:
    """Convert SVG path d-string to a matplotlib Path in viewBox coordinates."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)

    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    MOVETO = int(matplotlib.path.Path.MOVETO)
    LINETO = int(matplotlib.path.Path.LINETO)
    CURVE4 = int(matplotlib.path.Path.CURVE4)
    CLOSEPOLY = int(matplotlib.path.Path.CLOSEPOLY)

    i = 0
    cmd = "M"
    cx, cy = 0.0, 0.0

    def _next_floats(n: int) -> list[float]:
        nonlocal i
        vals = [float(tokens[i + k]) for k in range(n)]
        i += n
        return vals

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            continue

        if cmd in ("M", "m"):
            x, y = _next_floats(2)
            if cmd == "m":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(MOVETO)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x, y = _next_floats(2)
            if cmd == "l":
                x, y = cx + x, cy + y
            cx, cy = x, y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("H", "h"):
            (x,) = _next_floats(1)
            if cmd == "h":
                x = cx + x
            cx = x
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("V", "v"):
            (y,) = _next_floats(1)
            if cmd == "v":
                y = cy + y
            cy = y
            vertices.append((cx, cy))
            codes.append(LINETO)
        elif cmd in ("C", "c"):
            coords = _next_floats(6)
            if cmd == "c":
                coords = [
                    cx + coords[0],
                    cy + coords[1],
                    cx + coords[2],
                    cy + coords[3],
                    cx + coords[4],
                    cy + coords[5],
                ]
            for k in range(3):
                vertices.append((coords[2 * k], coords[2 * k + 1]))
                codes.append(CURVE4)
            cx, cy = coords[4], coords[5]
        elif cmd in ("S", "s"):
            coords = _next_floats(4)
            if cmd == "s":
                coords = [
                    cx + coords[0],
                    cy + coords[1],
                    cx + coords[2],
                    cy + coords[3],
                ]
            cp1 = (
                (2 * cx - vertices[-2][0], 2 * cy - vertices[-2][1])
                if codes and codes[-1] == CURVE4 and len(vertices) >= 2
                else (cx, cy)
            )
            for pt in [cp1, (coords[0], coords[1]), (coords[2], coords[3])]:
                vertices.append(pt)
                codes.append(CURVE4)
            cx, cy = coords[2], coords[3]
        elif cmd in ("Z", "z"):
            if vertices:
                vertices.append(vertices[-1])
                codes.append(CLOSEPOLY)
        else:
            if i < len(tokens) and not tokens[i].isalpha():
                i += 1
            else:
                i += 1
            continue

    if len(vertices) < 2:
        return None

    return matplotlib.path.Path(np.array(vertices), np.array(codes, dtype=np.uint8))


def _render_plot(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    agent_labels: list[str],
    color_fn: "Callable[[int], str]",
    label_fn: "Callable[[int], str]",
    title: str,
    plot_path: Path,
    svg_bg: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    if svg_bg:
        _draw_svg_background(ax)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude", fontsize=9, color="#52514e")
        ax.set_ylabel("Latitude", fontsize=9, color="#52514e")

    scatter_handles: dict[int, matplotlib.artist.Artist] = {}
    for i, (x, y) in enumerate(coords.tolist()):
        lbl = int(cluster_labels[i])
        color = color_fn(lbl)
        sc = ax.scatter(x, y, color=color, s=60, linewidths=0.5, edgecolors="white", zorder=3)
        if lbl not in scatter_handles:
            scatter_handles[lbl] = sc
        ax.annotate(
            agent_labels[i],
            (x, y),
            fontsize=5.5,
            color="#52514e",
            xytext=(3, 3),
            textcoords="offset points",
            clip_on=True,
        )

    legend_elements = []
    for lbl in sorted(scatter_handles.keys()):
        legend_elements.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_fn(lbl),
                markersize=7,
                label=label_fn(lbl),
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(title, fontsize=12, color="#0b0b0b")
    if not svg_bg:
        ax.set_xlabel("UMAP-1", fontsize=9, color="#52514e")
        ax.set_ylabel("UMAP-2", fontsize=9, color="#52514e")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e0e0e0")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


def _draw_svg_background(ax: matplotlib.axes.Axes) -> None:
    """
    Draw Continents.svg as a light-gray background in lon/lat space [-180,180] x [-90,90].
    SVG viewBox 0 0 2752.766 1537.631 maps directly to lon/lat (equirectangular projection):
    x=0..2752.766 -> lon=-180..180, y=0..1537.631 -> lat=90..-90.
    """
    svg_paths = _parse_svg_paths()
    if not svg_paths:
        return

    # Map content bounds derived from linear fit of known country centroids (R²=0.999).
    # Equirectangular but with border padding — actual geographic extent is smaller than viewBox.
    mx0, mx1 = -109.73, 2651.67
    my0, my1 = 40.94, 1484.71

    def _remap(verts: np.ndarray) -> np.ndarray:
        out = verts.copy()
        out[:, 0] = -180.0 + (verts[:, 0] - mx0) / (mx1 - mx0) * 360.0
        out[:, 1] = 90.0 - (verts[:, 1] - my0) / (my1 - my0) * 180.0
        return out

    ax.set_facecolor("#ffffff")  # ocean = white
    for svg_path in svg_paths:
        remapped_verts = _remap(np.asarray(svg_path.vertices))
        remapped_path = matplotlib.path.Path(remapped_verts, svg_path.codes)
        patch = matplotlib.patches.PathPatch(
            remapped_path,
            facecolor="#b9b9b9",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(patch)


def _cluster_summary(cluster_labels: np.ndarray, agent_labels: list[str]) -> dict[int, list[str]]:
    cluster_to_dbs: dict[int, list[str]] = {}
    for i, lbl in enumerate(cluster_labels):
        cluster_to_dbs.setdefault(int(lbl), []).append(agent_labels[i])
    return cluster_to_dbs


def _build_geo_init(agent_labels: list[str], n_components: int = 2) -> tuple[np.ndarray, int] | None:
    """
    Build a geo-biased UMAP init array where every DB stem starts at its geographic position.
    Non-matched stems (unknown DBs) start at zero — UMAP optimizes from there.
    Returns (init_array, n_pinned) or None if no DBs were matched.
    """
    n = len(agent_labels)
    custom_init = np.zeros((n, n_components), dtype=float)

    n_pinned = 0
    for i, label in enumerate(agent_labels):
        geo = _geo_coord(label)
        if geo is not None:
            lon, lat = geo
            # normalize lon [-180,180] → [-1,1], lat [-90,90] → [-1,1]
            custom_init[i, 0] = lon / 180.0
            custom_init[i, 1] = lat / 90.0
            n_pinned += 1

    if n_pinned == 0:
        return None
    return custom_init, n_pinned


async def _run(
    agents: list[DataAgent],
    registry_type: AgentRegistryType,
    dataset_name: str = "",
) -> None:
    output_dir = context.get_output_dir()
    assert output_dir is not None
    print(f"Loading embeddings for {len(agents)} agents …")
    all_table_embeddings: dict[str, dict[str, list[float]]] = await agent_registry.embedding_vectors_for_all_tables(
        agents, registry_type
    )

    agent_names: list[str] = []
    agent_labels: list[str] = []
    vectors: list[list[float]] = []

    for agent in agents:
        tables = all_table_embeddings.get(agent.name, {})
        if not tables:
            continue
        mat = np.array(list(tables.values()), dtype=float)
        vec = mat.mean(axis=0).tolist()
        agent_names.append(agent.name)
        agent_labels.append(_db_label(agent))
        vectors.append(vec)

    X = np.array(vectors, dtype=float)
    print(f"Embedding matrix: {X.shape}")

    # Shared UMAP layout
    reducer = umap.UMAP(random_state=42, n_jobs=1)
    coords = cast(np.ndarray, reducer.fit_transform(X))

    geo_result = _build_geo_init(agent_labels)

    hdb = fast_hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="cosine",
    ).fit(X)
    hdbscan_labels: np.ndarray = cast(np.ndarray, hdb.labels_)
    n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_hdbscan_noise = int((hdbscan_labels == -1).sum())
    print(
        f"HDBSCAN: {n_hdbscan_clusters} clusters, {n_hdbscan_noise} noise points"
        f" (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE})"
    )

    # EVoC clustering
    evoc_min_cluster_size = max(HDBSCAN_MIN_CLUSTER_SIZE, int(len(X) * 0.05))
    evoc_clusterer = evoc.EVoC(min_samples=MIN_SAMPLES, base_min_cluster_size=evoc_min_cluster_size)
    evoc_top_labels: np.ndarray = evoc_clusterer.fit_predict(X)
    # finest layer for coloring; falls back to top labels if only one layer
    evoc_fine_labels: np.ndarray = (
        evoc_clusterer.cluster_layers_[0] if len(evoc_clusterer.cluster_layers_) > 1 else evoc_top_labels
    )
    n_evoc_clusters = len(set(evoc_top_labels)) - (1 if -1 in evoc_top_labels else 0)
    n_evoc_fine_clusters = len(set(evoc_fine_labels)) - (1 if -1 in evoc_fine_labels else 0)
    n_evoc_noise = int((evoc_fine_labels == -1).sum())
    print(
        f"EVoC: {n_evoc_clusters} top-level clusters, {n_evoc_fine_clusters} fine clusters, {n_evoc_noise} noise points"
    )

    evoc_color_map = _build_evoc_color_map(evoc_clusterer.cluster_layers_, evoc_top_labels)

    def _evoc_color(fine_lbl: int) -> str:
        if fine_lbl == -1:
            return NOISE_COLOR
        return evoc_color_map.get(fine_lbl, NOISE_COLOR)

    def _evoc_legend_label(fine_lbl: int) -> str:
        if fine_lbl == -1:
            return "Noise"
        top = int(evoc_top_labels[np.where(evoc_fine_labels == fine_lbl)[0][0]])
        return f"Cluster {top}.{fine_lbl}"

    # EVoC node embedding as 2D layout (EVoC internal DR) + EVoC clusters
    _evoc_rng = np.random.RandomState(42)
    _n_neighbors = evoc_clusterer.n_neighbors
    _nn_inds, _nn_dists = knn_graph(X.astype(np.float32), n_neighbors=_n_neighbors, random_state=_evoc_rng)
    _graph = neighbor_graph_matrix(_n_neighbors, _nn_inds, _nn_dists, symmetrize=True)
    _init_emb = label_propagation_init(
        _graph,
        n_components=2,
        approx_n_parts=int(np.clip(8 * np.sqrt(X.shape[0]), 256, 16384)),
        random_scale=0.1,
        scaling=0.5,
        noise_level=0.5,
        random_state=_evoc_rng,
        data=X.astype(np.float32),
        n_label_prop_iter=20,
    )
    evoc_dr_coords = cast(
        np.ndarray,
        evoc_node_embedding(
            _graph,
            n_components=2,
            n_epochs=50,
            initial_embedding=_init_emb,
            negative_sample_rate=1.0,
            noise_level=0.5,
            random_state=_evoc_rng,
            verbose=False,
            reproducible_flag=True,
            initial_alpha=0.1,
        ),
    )
    print(f"EVoC DR: 2D node embedding shape {evoc_dr_coords.shape}")

    # Build all plot jobs then render in parallel
    title_prefix = f"[{dataset_name} / {registry_type}] " if dataset_name else f"[{registry_type}] "
    plot_jobs: list[tuple] = [
        (
            coords,
            hdbscan_labels,
            agent_labels,
            _color_for_label,
            lambda lbl: "Noise" if lbl == -1 else f"Cluster {lbl}",
            f"{title_prefix}DB Agent Embeddings — UMAP + HDBSCAN clusters",
            output_dir / "embedding_clusters_umap_hdbscan.png",
            False,
        ),
        (
            coords,
            evoc_fine_labels,
            agent_labels,
            _evoc_color,
            _evoc_legend_label,
            f"{title_prefix}DB Agent Embeddings — UMAP + EVoC clusters (hierarchical)",
            output_dir / "embedding_clusters_umap_evoc.png",
            False,
        ),
        (
            evoc_dr_coords,
            evoc_fine_labels,
            agent_labels,
            _evoc_color,
            _evoc_legend_label,
            f"{title_prefix}DB Agent Embeddings — EVoC node embedding + EVoC clusters (hierarchical)",
            output_dir / "embedding_clusters_evoc.png",
            False,
        ),
    ]

    if geo_result is not None:
        fixed_coords = geo_result[0].copy().astype(float)
        fixed_coords[:, 0] *= 180.0
        fixed_coords[:, 1] *= 90.0
        plot_jobs += [
            (
                fixed_coords,
                hdbscan_labels,
                agent_labels,
                _color_for_label,
                lambda lbl: "Noise" if lbl == -1 else f"Cluster {lbl}",
                f"{title_prefix}DB Agent Embeddings — Fixed lon/lat + HDBSCAN clusters",
                output_dir / "embedding_clusters_fixed_hdbscan.png",
                True,
            ),
            (
                fixed_coords,
                evoc_fine_labels,
                agent_labels,
                _evoc_color,
                _evoc_legend_label,
                f"{title_prefix}DB Agent Embeddings — Fixed lon/lat + EVoC clusters (hierarchical)",
                output_dir / "embedding_clusters_fixed_evoc.png",
                True,
            ),
        ]

    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_render_plot, *job) for job in plot_jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    hdbscan_cluster_map = _cluster_summary(hdbscan_labels, agent_labels)
    evoc_cluster_map = _cluster_summary(evoc_top_labels, agent_labels)

    def _md_cluster_section(cluster_map: dict[int, list[str]]) -> list[str]:
        section: list[str] = []
        for lbl in sorted(cluster_map.keys()):
            dbs = sorted(cluster_map[lbl])
            heading = "## Noise" if lbl == -1 else f"## Cluster {lbl}"
            section.append(heading)
            section.append(f"_{len(dbs)} databases_")
            section.append("")
            for db_name in dbs:
                section.append(f"- {db_name}")
            section.append("")
        return section

    md_lines = [
        "# DB Embedding Clusters",
        "",
        f"**Total agents:** {len(agent_names)}  ",
        "",
        "## HDBSCAN",
        "",
        f"**min_cluster_size:** {HDBSCAN_MIN_CLUSTER_SIZE}, min_samples: {MIN_SAMPLES}  ",
        f"**Clusters:** {n_hdbscan_clusters}  ",
        f"**Noise points:** {n_hdbscan_noise}",
        "",
        "![HDBSCAN visualization](embedding_clusters_umap_hdbscan.png)",
        "",
        *_md_cluster_section(hdbscan_cluster_map),
        "## EVoC",
        "",
        f"**min_samples:** {MIN_SAMPLES}  ",
        f"**Top-level clusters:** {n_evoc_clusters}  ",
        f"**Fine clusters:** {n_evoc_fine_clusters}  ",
        f"**Noise points:** {n_evoc_noise}",
        "",
        "![EVoC visualization](embedding_clusters_umap_evoc.png)",
        "",
        "![EVoC DR visualization](embedding_clusters_evoc.png)",
        "",
        *_md_cluster_section(evoc_cluster_map),
    ]

    md_path = output_dir / "embedding_clusters.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report saved: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize agent embeddings via UMAP + HDBSCAN and EVoC")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for agent data files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots and reports")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--config", type=str, default=None, help="Config as YAML file path or inline YAML string")
    args = parser.parse_args()

    input_file = Path(args.input)
    data_dir = Path(args.data_dir) if args.data_dir else input_file.parent
    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_dir = Path(args.output_dir) / datetime_str if args.output_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    config = load_config(args.config)
    registry_type = AgentRegistryType(config.selection_kwargs.get("registry_type", AgentRegistryType.TECHNICAL))

    data, _extra = load_input(input_file)
    agents: list[DataAgent] = data.data_agents

    with context.pipeline_context(
        name="embedding_visualization",
        output_dir=output_dir,
        cache_dir=cache_dir,
        data_dir=data_dir,
        config=config,
    ):
        asyncio.run(_run(agents, registry_type, input_file.stem))


if __name__ == "__main__":
    main()
