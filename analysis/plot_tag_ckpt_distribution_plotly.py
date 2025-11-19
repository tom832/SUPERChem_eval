"""
Single-panel Plotly sunburst of ability sub-categories, which are then
combined into a final 2x2 grid image.
Final version with aesthetic refinements, numerical sorting, and image combination.

Outputs:
- ability_four_panel_combined.png (final 2x2 grid image)
- ability_panel_{id}.html (interactive, one for each main category)
"""

import argparse
import json
import os
import re
import textwrap
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import plotly.graph_objects as go
from plotly.colors import hex_to_rgb

# Add dependency for image combination
try:
    from PIL import Image
except ImportError:
    print("Pillow library not found. Please install it with 'pip install Pillow'")
    Image = None

# -----------------------------
# Defaults (match your repo)
# -----------------------------
DEFAULT_TAG_DATA_FILE = 'data/ckpt_20250915021133_questions_en_true_gemini-2_5-pro_high_1_0_1.jsonl'
DEFAULT_DESC_FILE = 'data/ability_tags_description.json'
DEFAULT_SPLIT_MAP_FILE = 'data/dataset_split_map.json'

TAG_CODE_RE = re.compile(r'^(\d+)\.(\d+)')

# A list of distinct base colors for the top 4 categories
DISTINCT_BASE_COLORS = [
    "#5BA4E6",  # Blue
    "#E36C6C",  # Red
    "#9ED08F",  # Green
    "#F58518",  # Orange
]

# -----------------------------
# I/O
# -----------------------------
def load_valid_ids(split_map_file: str) -> Optional[set]:
    if not os.path.exists(split_map_file):
        print(f"[load_valid_ids] split map missing: {split_map_file} (using all)")
        return None
    with open(split_map_file, "r", encoding="utf-8") as f:
        split_map = json.load(f)
    s = {u for u, d in split_map.items() if d.get("split") in ["release", "holdout"]}
    print(f"[load_valid_ids] non-easy questions: {len(s)}")
    return s

def load_tag_data(filepath: str, split_map_file: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"tag data file not found: {filepath}")
    valid = load_valid_ids(split_map_file)
    knowledge, ability = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid is not None and rec.get("uuid") not in valid: continue
            analyses = rec.get("checkpoints_analysis", [])
            if not isinstance(analyses, list): continue
            for a in analyses:
                knowledge.extend(a.get("knowledge_tags", []) or [])
                ability.extend(a.get("ability_tags", []) or [])
    print(f"[load_tag_data] collected {len(knowledge)} knowledge, {len(ability)} ability tags.")
    return knowledge, ability

def load_desc(desc_file: str) -> Dict:
    if not os.path.exists(desc_file):
        raise FileNotFoundError(f"description file not found: {desc_file}")
    with open(desc_file, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Aggregation
# -----------------------------
def aggregate(ability_tags: List[str]) -> Tuple[Counter, Dict[str, Counter]]:
    main_counts: Counter = Counter()
    sub_counts: Dict[str, Counter] = defaultdict(Counter)
    for tag in ability_tags:
        m = TAG_CODE_RE.match(tag)
        if not m: continue
        mid, code = m.group(1), m.group(0)
        main_counts[mid] += 1
        sub_counts[mid][code] += 1
    return main_counts, sub_counts

# -----------------------------
# Colors and labels Helpers
# -----------------------------
def lighten_hex(color: str, factor: float) -> str:
    r, g, b = hex_to_rgb(color)
    r2, g2, b2 = int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor)
    return f"#{r2:02X}{g2:02X}{b2:02X}"

def main_title_for(mid: str, desc: Dict) -> str:
    return desc.get("main_categories", {}).get(mid) or f"Category {mid}"

def wrap_text(text: str, width: int) -> str:
    """Wraps text using HTML <br> tags."""
    return "<br>".join(textwrap.wrap(text, width=width))

# -----------------------------
# Chart building
# -----------------------------
def build_single_panel(main_counts: Counter, sub_counts: Dict[str, Counter], desc: Dict, mid: str, base_color: str) -> go.Figure:
    """Creates a single sunburst chart figure for one main category."""
    start_x = {'1': 0.215, '2': 0.215, '3': 0.218, '4': 0.202}
    m_title = main_title_for(mid, desc)
    subs = sub_counts.get(mid, {})
    if not subs: return go.Figure()

    sorted_sub = sorted(subs.keys(), key=lambda code: tuple(map(int, code.split('.'))))
    
    labels, ids, parents, values = [m_title], [f"main::{mid}"], [""], [main_counts[mid]]
    colors = [base_color]
    custom = [dict(name=m_title, is_main=True)]

    for j, code in enumerate(sorted_sub):
        sub_label = desc.get("sub_categories", {}).get(code) or code
        labels.append(sub_label)
        ids.append(f"sub::{code}")
        parents.append(f"main::{mid}")
        values.append(subs[code])
        tint = 0.15 + 0.60 * (j / max(len(sorted_sub) - 1, 1))
        colors.append(lighten_hex(base_color, tint))
        custom.append(dict(name=sub_label, is_main=False))

    fig = go.Figure()

    fig.add_trace(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total", maxdepth=2, domain=dict(x=[0, 0.55]),
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{customdata[name]}</b><br>Count: %{value}<br>Share: %{percentParent:.1%}<extra></extra>",
        customdata=custom, textinfo="percent parent",
        # MODIFICATION 1: Set font family to Arial
        textfont=dict(size=24, color="black", family="Arial"),
        sort=False,
    ))

    total_tags_in_cat = main_counts[mid]
    total_tags_overall = sum(main_counts.values()) or 1
    share = 100 * total_tags_in_cat / total_tags_overall
    
    fig.add_annotation(
        text=f"<b>{wrap_text(m_title, 20)}</b><br><sup>{total_tags_in_cat} tags ({share:.1f}% of total)</sup>",
        x=start_x[mid], y=0.5, xref="paper", yref="paper", showarrow=False,
        # MODIFICATION 1: Set font family to Arial
        font=dict(size=24, color="black", family="Arial"),
        align="center", yanchor="middle"
    )
    
    legend_y_start = 0.95
    legend_y_step = 0.07
    for j, code in enumerate(sorted_sub):
        full_name = desc.get("sub_categories", {}).get(code, code)
        slice_color = colors[j + 1]
        fig.add_annotation(
            text=f'<span style="color:{slice_color};font-size:22px;">■</span> {full_name}',
            align='left', showarrow=False, xref="paper", yref="paper", x=0.52,
            y=legend_y_start - (j * legend_y_step), xanchor='left', yanchor='top',
            # MODIFICATION 1: Set font family to Arial
            font=dict(size=30, family="Arial")
        )

    fig.update_layout(
        title=None, paper_bgcolor="white", margin=dict(t=20, l=20, r=20, b=20),
        width=1600, height=700, uniformtext=dict(minsize=10, mode="show")
    )
    return fig

# -----------------------------
# MODIFICATION 2: Image Combination Function
# -----------------------------
def combine_images_to_grid(image_files: List[str], output_file: str):
    """Combines four images into a 2x2 grid."""
    if Image is None:
        print("[combine_images] Pillow is not installed. Skipping image combination.")
        return
    if len(image_files) != 4:
        print(f"[combine_images] Expected 4 images, but found {len(image_files)}. Skipping.")
        return

    images = [Image.open(f) for f in image_files]
    width, height = images[0].size
    
    # Create a new blank image (canvas) with a white background
    grid_img = Image.new('RGB', (width * 2, height * 2), 'white')
    
    # Paste the images into the grid
    grid_img.paste(images[0], (0, 0))
    grid_img.paste(images[1], (width, 0))
    grid_img.paste(images[2], (0, height))
    grid_img.paste(images[3], (width, height))
    
    # Save the final combined image
    grid_img.save(output_file)
    print(f"[output] Combined image saved to: {output_file}")

    # Clean up the individual files
    for f in image_files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing temporary file {f}: {e}")


def combine_svgs_to_grid(svg_files: List[str], output_file: str):
    """Combines four SVG files into a 2x2 grid SVG."""
    if len(svg_files) != 4:
        print(f"[combine_svgs] Expected 4 SVGs, but found {len(svg_files)}. Skipping.")
        return

    svg_contents = []
    for f in svg_files:
        with open(f, 'r', encoding='utf-8') as file:
            svg_contents.append(file.read())

    # Create a new SVG canvas
    combined_svg = f'''<svg width="3200" height="1400" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(0,0)">{svg_contents[0]}</g>
    <g transform="translate(1600,0)">{svg_contents[1]}</g>
    <g transform="translate(0,700)">{svg_contents[2]}</g>
    <g transform="translate(1600,700)">{svg_contents[3]}</g>
    </svg>'''

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(combined_svg)
    
    print(f"[output] Combined SVG saved to: {output_file}")

    # convert to pdf
    try:
        import cairosvg
        pdf_output = output_file.replace('.svg', '.pdf')
        cairosvg.svg2pdf(url=output_file, write_to=pdf_output)
        print(f"[output] Combined PDF saved to: {pdf_output}")
    except ImportError:
        print("[combine_svgs] CairoSVG not installed. Skipping PDF conversion.")

    # Clean up the individual files
    for f in svg_files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing temporary file {f}: {e}")
# -----------------------------
# Main Execution Logic
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Single-panel Plotly sunburst for CCMEBench abilities")
    ap.add_argument("--tags", default=DEFAULT_TAG_DATA_FILE, help="checkpoint-level tags jsonl")
    ap.add_argument("--desc", default=DEFAULT_DESC_FILE, help="ability tag description json")
    ap.add_argument("--split", default=DEFAULT_SPLIT_MAP_FILE, help="dataset_split_map.json (filters to release/holdout)")
    args = ap.parse_args()

    _, ability = load_tag_data(args.tags, args.split)
    if not ability:
        print("[main] no ability tags found.")
        return
    desc = load_desc(args.desc)
    main_counts, sub_counts = aggregate(ability)

    all_ids = [mid for mid, _ in main_counts.most_common(4)]
    all_ids.sort(key=int)
    
    generated_pngs = []

    for idx, mid in enumerate(all_ids):
        print(f"\n--- Generating chart for Main Category '{mid}' ---")
        base_color = DISTINCT_BASE_COLORS[idx % len(DISTINCT_BASE_COLORS)]
        
        fig = build_single_panel(main_counts, sub_counts, desc, mid, base_color)

        html_out = f"ability_panel_{mid}.html"
        png_out = f"ability_panel_{mid}.png"
        svg_out = f"ability_panel_{mid}.svg"

        fig.write_html(html_out, include_plotlyjs="cdn", full_html=True)
        print(f"[output] interactive: {html_out}")

        try:
            fig.write_image(png_out, scale=3)
            print(f"[output] static: {png_out}")
            generated_pngs.append(png_out)
        except Exception as e:
            print(f"[output] PNG export skipped for '{png_out}' (install kaleido): {e}")
        try:
            fig.write_image(svg_out)
            print(f"[output] static: {svg_out}")
        except Exception as e:
            print(f"[output] SVG export skipped for '{svg_out}' (install kaleido): {e}")
    
    # After generating all images, combine them
    if generated_pngs:
        print("\n--- Combining images into a 2x2 grid ---")
        combine_images_to_grid(generated_pngs, "ability_four_panel_combined.png")
        combine_svgs_to_grid(
            [f"ability_panel_{mid}.svg" for mid in all_ids],
            "results/ability_four_panel_combined.svg"
        )

if __name__ == "__main__":
    main()