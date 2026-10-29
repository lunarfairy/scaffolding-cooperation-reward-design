"""Minimal, self-contained figure style used for all manuscript figures (no external style packages)."""
import matplotlib as mpl

SIZES = (7, 6.5, 6)          # base (titles, axis labels), annotations/legend, ticks
COL = {'e3': '#b35806', 'welfare': '#f1a340', 'desert': '#542788', 'coopmean': '#2166ac', 'coopfinal': '#67a9cf',
       'rawls': '#1b7837', 'sustain': '#35978f', 'encouragement': '#000000', 'conciliation': '#7f7f7f',
       'exclusion': '#bdbdbd', 'static': '#d9d9d9', 'neutral': '#a6a6a6'}
LABEL = {'e3': 'Capital (published reward)', 'welfare': 'Welfare', 'desert': 'Desert-weighted welfare',
         'coopmean': 'Mean cooperation', 'coopfinal': 'Final-round cooperation', 'rawls': 'Maximin capital',
         'sustain': 'Frozen-network cooperation', 'encouragement': 'Distilled published schedule',
         'conciliation': 'Always conciliate', 'exclusion': 'Always exclude', 'static': 'Static network',
         'neutral': 'Neutral schedule'}
SHORT = {'e3': 'Capital', 'welfare': 'Welfare', 'desert': 'Desert', 'coopmean': 'Mean coop.',
         'coopfinal': 'Final coop.', 'rawls': 'Maximin', 'sustain': 'Frozen-net. coop.',
         'encouragement': 'Distilled', 'conciliation': 'Conciliate', 'exclusion': 'Exclude'}
CORE = ['e3', 'welfare', 'desert', 'rawls', 'coopmean', 'coopfinal', 'sustain']


def apply_style():
    b, m, s = SIZES
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': b, 'axes.titlesize': b, 'axes.labelsize': b, 'legend.fontsize': m,
        'xtick.labelsize': s, 'ytick.labelsize': s, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'lines.linewidth': 1.1, 'lines.markersize': 3.2, 'legend.frameon': False,
        'axes.titlelocation': 'left', 'savefig.dpi': 300, 'figure.dpi': 150, 'pdf.fonttype': 42,
        'axes.titlepad': 4})


def panel_letter(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
