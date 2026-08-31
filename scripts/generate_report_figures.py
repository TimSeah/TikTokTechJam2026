from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "report_figures"

INK = "#16202A"
MUTED = "#64717D"
GRID = "#DCE2E6"
PAPER = "#FFFFFF"
NAVY = "#164B73"
TEAL = "#00876C"
GREEN = "#62A744"
ORANGE = "#D27728"
RED = "#B94343"
PURPLE = "#6D5A98"
PALE_BLUE = "#E8F1F6"
PALE_GREEN = "#E7F3EF"
PALE_ORANGE = "#F8EEE3"
PALE_RED = "#F7E6E6"


def load_report_metrics() -> dict:
    path = ROOT / "docs" / "report_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_shift_audit() -> dict:
    path = ROOT / "outputs" / "shift_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_native_stress() -> dict:
    path = ROOT / "outputs" / "native_stress.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> float:
    return draw.textbbox((0, 0), text, font=font)[2]


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    draw.text(
        (center_x - text_width(draw, text, font) / 2, y), text, font=font, fill=fill
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str = MUTED,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 16
    left = (
        end[0] - head * math.cos(angle - math.pi / 6),
        end[1] - head * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6),
        end[1] - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=fill)


def box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    detail: Iterable[str],
    fill: str,
    outline: str,
) -> None:
    draw.rounded_rectangle(bounds, radius=10, fill=fill, outline=outline, width=4)
    center_x = (bounds[0] + bounds[2]) / 2
    centered_text(draw, center_x, bounds[1] + 24, title, load_font(29, bold=True))
    for index, line in enumerate(detail):
        centered_text(
            draw,
            center_x,
            bounds[1] + 72 + index * 32,
            line,
            load_font(23),
            MUTED,
        )


def save(image: Image.Image, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / name, optimize=True, dpi=(180, 180))


def horizontal_scale(
    value: float, minimum: float, maximum: float, left: float, right: float
) -> float:
    bounded = min(max(value, minimum), maximum)
    return left + (bounded - minimum) / (maximum - minimum) * (right - left)


def log_scale(value: float, left: float, right: float) -> float:
    bounded = min(max(value, 1.0), 1000.0)
    return left + math.log10(bounded) / 3.0 * (right - left)


def heat_color(value: float) -> str:
    if value < 0.48:
        return "#EAB8B8"
    if value < 0.55:
        return "#F4D4D0"
    if value < 0.70:
        return "#F6E5C8"
    if value < 0.82:
        return "#DCEEDC"
    return "#B9DFC8"


def architecture_figure() -> None:
    metrics = load_report_metrics()["submitted_hybrid"]
    image = Image.new("RGB", (1900, 920), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "One image, two feature paths, one decision",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "The two paths are combined before the final score; later tests show that they do not transfer equally.",
        font=load_font(26),
        fill=MUTED,
    )

    box(
        draw,
        (55, 330, 305, 545),
        "RGB image",
        ("decode once", "send to both branches"),
        "#F4F6F7",
        MUTED,
    )
    box(
        draw,
        (420, 175, 800, 390),
        "Frozen OpenCLIP",
        ("ViT-B/32 image encoder", "512-D L2-normalized feature"),
        PALE_BLUE,
        NAVY,
    )
    box(
        draw,
        (420, 510, 800, 725),
        "Frequency branch",
        ("grayscale + Hann + FFT", "32 radial log-magnitude means"),
        PALE_ORANGE,
        ORANGE,
    )
    box(
        draw,
        (965, 330, 1260, 545),
        "544-D vector",
        ("512 semantic + 32 FFT", "feature concatenation"),
        PALE_GREEN,
        TEAL,
    )
    box(
        draw,
        (1390, 330, 1645, 545),
        "Linear head",
        ("StandardScaler", "logistic regression"),
        "#EEF5E8",
        GREEN,
    )
    box(
        draw,
        (1740, 330, 1870, 545),
        "Score",
        ("P(FAKE)", "0 to 1"),
        "#EEF5E8",
        GREEN,
    )

    arrow(draw, (305, 385), (420, 285))
    arrow(draw, (305, 490), (420, 620))
    arrow(draw, (800, 285), (965, 385))
    arrow(draw, (800, 620), (965, 490))
    arrow(draw, (1260, 438), (1390, 438))
    arrow(draw, (1645, 438), (1740, 438))

    draw.rounded_rectangle(
        (310, 790, 1590, 870), radius=8, fill=PALE_RED, outline=RED, width=3
    )
    centered_text(draw, 950, 808, "Blind-test diagnosis", load_font(25, bold=True), RED)
    centered_text(
        draw,
        950,
        840,
        "The 32-D frequency path was removed from the promoted model; the 512-D semantic path was retained.",
        load_font(23),
        INK,
    )
    draw.text(
        (70, 760),
        f"Submitted representation: {metrics['feature_dimensions']} dimensions | learned head: {metrics['learned_values']} values",
        font=load_font(22),
        fill=MUTED,
    )
    save(image, "architecture.png")


def draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    minimum: float,
    maximum: float,
    ticks: Iterable[float],
) -> None:
    left, top, right, bottom = plot
    for tick in ticks:
        y = bottom - (tick - minimum) / (maximum - minimum) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{tick:.2f}"
        draw.text((left - 75, y - 15), label, font=load_font(23), fill=MUTED)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)


def robustness_figure() -> None:
    rows = load_report_metrics()["submitted_hybrid"]["conditions"]
    family_colors = {
        "clean": INK,
        "jpeg": NAVY,
        "blur": ORANGE,
        "resize": RED,
        "noise": PURPLE,
        "jitter": TEAL,
        "crop": GREEN,
    }

    image = Image.new("RGB", (1900, 1350), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "How the hybrid responds to image changes",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "Held-out CIFAKE images, one transformation applied at a time (20,000 images per condition)",
        font=load_font(26),
        fill=MUTED,
    )

    left, right = 560, 1760
    minimum, maximum = 0.84, 1.0
    for tick in (0.84, 0.88, 0.92, 0.96, 1.0):
        x = horizontal_scale(tick, minimum, maximum, left, right)
        draw.line((x, 185, x, 1245), fill=GRID, width=2)
        centered_text(draw, x, 155, f"{tick:.2f}", load_font(22), MUTED)
    draw.text(
        (left, 1280), "ROC AUC (axis truncated at 0.84)", font=load_font(22), fill=MUTED
    )

    y = 205
    previous_family = ""
    family_labels = {
        "clean": "CLEAN",
        "jpeg": "JPEG",
        "blur": "BLUR",
        "resize": "RESIZE",
        "noise": "NOISE",
        "jitter": "JITTER",
        "crop": "CROP",
    }
    for row in rows:
        family = row["family"]
        if previous_family and family != previous_family:
            draw.line((70, y - 6, 1810, y - 6), fill=GRID, width=2)
            y += 12
        if family != previous_family:
            draw.text(
                (70, y + 8),
                family_labels[family],
                font=load_font(20, bold=True),
                fill=family_colors[family],
            )
        auc = float(row["auc"])
        end = horizontal_scale(auc, minimum, maximum, left, right)
        draw.text((185, y + 5), row["condition"], font=load_font(23), fill=INK)
        draw.rounded_rectangle((left, y + 4, right, y + 39), radius=7, fill="#F1F3F4")
        draw.rounded_rectangle(
            (left, y + 4, end, y + 39), radius=7, fill=family_colors[family]
        )
        draw.text(
            (right + 16, y + 4),
            f"{auc:.3f}",
            font=load_font(23, bold=True),
            fill=family_colors[family],
        )
        y += 58
        previous_family = family

    draw.rounded_rectangle(
        (1020, 1270, 1810, 1325), radius=8, fill=PALE_ORANGE, outline=ORANGE, width=2
    )
    centered_text(
        draw,
        1415,
        1282,
        "Largest drops: resize 0.25x (0.883) and blur sigma 2.0 (0.887)",
        load_font(21, bold=True),
        INK,
    )
    save(image, "robustness_conditions.png")


def grouped_bar_figure() -> None:
    rows = load_report_metrics()["submitted_hybrid"]["ablation"]
    metrics = [
        ("clean_auc", "Clean AUC", NAVY),
        ("robust_auc", "Robust AUC", ORANGE),
    ]

    image = Image.new("RGB", (1900, 1000), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "What changed when features and augmentation were added",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "FFT helps the clean split slightly; augmentation is what improves the transformed split.",
        font=load_font(26),
        fill=MUTED,
    )
    plot = (150, 200, 1830, 760)
    minimum, maximum = 0.88, 1.0
    draw_axes(draw, plot, minimum, maximum, (0.88, 0.91, 0.94, 0.97, 1.0))
    group_width = (plot[2] - plot[0]) / len(rows)
    bar_width = 125
    for group_index, row in enumerate(rows):
        center = plot[0] + (group_index + 0.5) * group_width
        for metric_index, (key, _, color) in enumerate(metrics):
            value = float(row[key])
            x = center + (metric_index - 0.5) * 160
            y = plot[3] - (value - minimum) / (maximum - minimum) * (plot[3] - plot[1])
            draw.rounded_rectangle(
                (x - bar_width / 2, y, x + bar_width / 2, plot[3]), radius=6, fill=color
            )
            centered_text(
                draw, x, y - 36, f"{value:.3f}", load_font(21, bold=True), color
            )
        centered_text(
            draw, center, plot[3] + 18, row["model"], load_font(25, bold=True), INK
        )
        centered_text(draw, center, plot[3] + 51, row["training"], load_font(22), MUTED)
        centered_text(
            draw,
            center,
            plot[3] + 92,
            f"Composite {float(row['composite']):.3f}",
            load_font(22, bold=True),
            TEAL,
        )

    legend_x = 705
    for index, (_, label, color) in enumerate(metrics):
        x = legend_x + index * 360
        draw.rounded_rectangle((x, 920, x + 35, 955), radius=4, fill=color)
        draw.text((x + 50, 922), label, font=load_font(23), fill=INK)
    save(image, "ablation.png")


def blind_transfer_figure() -> None:
    payload = load_report_metrics()["blind_ablation"]
    image = Image.new("RGB", (1900, 940), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "The model order changes on the native-resolution samples",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "ROC AUC on three frozen, balanced samples; red marks indicate chance-level or worse ranking.",
        font=load_font(26),
        fill=MUTED,
    )

    label_right = 500
    cell_width, cell_height, gap = 400, 170, 24
    start_x, start_y = 525, 245
    for column, dataset in enumerate(payload["datasets"]):
        center = start_x + column * (cell_width + gap) + cell_width / 2
        centered_text(draw, center, 188, dataset, load_font(25, bold=True), INK)

    subtitles = [
        "Semantic features only",
        "Semantic + frequency",
        "Semantic + frequency + augmentation",
    ]
    for row_index, model in enumerate(payload["models"]):
        y = start_y + row_index * (cell_height + gap)
        draw.text((70, y + 42), model["model"], font=load_font(27, bold=True), fill=INK)
        draw.text((70, y + 82), subtitles[row_index], font=load_font(22), fill=MUTED)
        draw.line((label_right, y, label_right, y + cell_height), fill=GRID, width=2)
        for column, auc in enumerate(model["auc"]):
            x = start_x + column * (cell_width + gap)
            draw.rounded_rectangle(
                (x, y, x + cell_width, y + cell_height),
                radius=10,
                fill=heat_color(float(auc)),
                outline=PAPER,
                width=5,
            )
            centered_text(
                draw,
                x + cell_width / 2,
                y + 38,
                f"{float(auc):.3f}",
                load_font(39, bold=True),
                INK,
            )
            if auc >= 0.70:
                status, status_color = "ranking retained", GREEN
            elif auc < 0.47:
                status, status_color = "below chance", RED
            else:
                status, status_color = "chance-level", RED
            centered_text(
                draw,
                x + cell_width / 2,
                y + 98,
                status,
                load_font(22, bold=True),
                status_color,
            )

    draw.rounded_rectangle(
        (300, 845, 1600, 910), radius=8, fill=PALE_RED, outline=RED, width=3
    )
    centered_text(
        draw,
        950,
        861,
        "The highest CIFAKE composite is the bottom row; native samples favour the simpler top row.",
        load_font(23, bold=True),
        INK,
    )
    save(image, "blind_transfer.png")


def frequency_extrapolation_figure() -> None:
    data = load_report_metrics()["frequency_diagnosis"]
    values = [
        ("CIFAKE clean", data["cifake_test_max_abs_standardized"], NAVY),
        ("SID-Set blind", data["blind_max_abs_standardized"]["SID-Set"], RED),
        (
            "COCO / DALL-E blind",
            data["blind_max_abs_standardized"]["COCO / DALL-E"],
            RED,
        ),
        (
            "LAION / DALL-E blind",
            data["blind_max_abs_standardized"]["LAION / DALL-E"],
            RED,
        ),
    ]
    image = Image.new("RGB", (1900, 1080), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "The frequency values leave the scale seen during fitting",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "Largest absolute standardized frequency value; logarithmic scale from 1 to 1,000",
        font=load_font(26),
        fill=MUTED,
    )

    left, right = 580, 1780
    for tick in (1, 10, 100, 1000):
        x = log_scale(float(tick), left, right)
        draw.line((x, 190, x, 680), fill=GRID, width=2)
        centered_text(draw, x, 158, f"{tick:,}", load_font(22), MUTED)

    for index, (label, value, color) in enumerate(values):
        y = 230 + index * 110
        end = log_scale(float(value), left, right)
        draw.text((70, y + 8), label, font=load_font(25, bold=True), fill=INK)
        draw.rounded_rectangle((left, y, right, y + 48), radius=8, fill="#F1F3F4")
        draw.rounded_rectangle((left, y, end, y + 48), radius=8, fill=color)
        draw.text(
            (right + 18, y + 6),
            f"{value:.1f}",
            font=load_font(25, bold=True),
            fill=color,
        )

    draw.text(
        (70, 690),
        f"Training distribution: 99.9th percentile = {data['training_p99_9_abs_standardized']:.2f}; maximum = {data['training_max_abs_standardized']:.2f}",
        font=load_font(22),
        fill=MUTED,
    )

    stages = [
        ("Expected scale", "clean max 12.15", PALE_BLUE, NAVY),
        ("Blind input", "feature magnitude about 500", PALE_RED, RED),
        ("Linear extrapolation", "SID logits 7.96 to 121.44", PALE_ORANGE, ORANGE),
        ("Saturated output", "P(FAKE) rounds to 1.0", PALE_RED, RED),
    ]
    stage_width = 365
    for index, (title, detail, fill, outline) in enumerate(stages):
        x = 70 + index * 455
        box(draw, (x, 790, x + stage_width, 1000), title, (detail,), fill, outline)
        if index < len(stages) - 1:
            arrow(draw, (x + stage_width, 895), (x + 445, 895), fill=MUTED, width=4)
    save(image, "frequency_extrapolation.png")


def progression_figure() -> None:
    promoted = load_report_metrics()["promoted_semantic"]
    stages = [
        ("1. Semantic baseline", "Clean .989", "Robust .918", PALE_BLUE, NAVY),
        ("2. Add FFT", "Clean .992", "Robust falls to .911", PALE_ORANGE, ORANGE),
        (
            "3. Add augmentation",
            "Robust rises to .956",
            "Composite .972",
            PALE_GREEN,
            TEAL,
        ),
        (
            "4. CIFAKE selection",
            "Strong CIFAKE score",
            "Submitted hybrid",
            "#EEF5E8",
            GREEN,
        ),
        ("5. Native test", "AUC about .50", "Every image called fake", PALE_RED, RED),
        (
            "6. Branch diagnosis",
            "Frequency |z| about 500",
            "Sigmoid saturation",
            PALE_ORANGE,
            ORANGE,
        ),
        (
            "7. Remove FFT",
            "3-domain semantic probe",
            "48k training rows",
            PALE_BLUE,
            NAVY,
        ),
        ("8. Promote", "SID .992", "WildFake .903 / .913", PALE_GREEN, GREEN),
    ]
    image = Image.new("RGB", (1900, 930), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 45),
        "How the evidence changed the model choice",
        font=load_font(44, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 105),
        "Each evaluation changed the next engineering decision.",
        font=load_font(26),
        fill=MUTED,
    )

    positions = [
        (55, 190),
        (510, 190),
        (965, 190),
        (1420, 190),
        (1420, 575),
        (965, 575),
        (510, 575),
        (55, 575),
    ]
    width, height = 365, 205
    for index, ((title, line1, line2, fill, outline), (x, y)) in enumerate(
        zip(stages, positions)
    ):
        box(draw, (x, y, x + width, y + height), title, (line1, line2), fill, outline)
        if index < 3:
            arrow(
                draw,
                (x + width, y + height / 2),
                (positions[index + 1][0] - 15, y + height / 2),
            )
        elif index == 3:
            arrow(
                draw,
                (x + width / 2, y + height),
                (x + width / 2, positions[4][1] - 15),
                fill=RED,
            )
        elif index < 7:
            arrow(
                draw,
                (x, y + height / 2),
                (positions[index + 1][0] + width + 15, y + height / 2),
            )

    centered_text(
        draw,
        950,
        855,
        f"Promoted semantic-only artifact: {promoted['learned_values']} learned values | all four gates passed | training and promotion in {promoted['training_seconds']:.1f} s",
        load_font(23, bold=True),
        INK,
    )
    save(image, "project_progression.png")


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C1C8",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.facecolor": "white",
        }
    )


def _save_plot(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def _clean_axes(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#C8D0D6")
    axis.spines["bottom"].set_color("#C8D0D6")
    axis.grid(True, axis=grid_axis, color="#E6EBEE", linewidth=0.8)
    axis.set_axisbelow(True)


def _standard_architecture_figure() -> None:
    figure, axis = plt.subplots(figsize=(11.2, 3.9))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 4.4)
    axis.axis("off")
    nodes = [
        (0.35, 1.55, 1.45, 1.2, "RGB image", "decode once", "#F4F6F7", MUTED),
        (
            2.45,
            2.35,
            2.45,
            0.95,
            "Frozen OpenCLIP",
            "512-D normalized feature",
            "#E8F1F6",
            NAVY,
        ),
        (
            2.45,
            0.95,
            2.45,
            0.95,
            "Frequency summary",
            "32 radial log-FFT values",
            "#F8EEE3",
            ORANGE,
        ),
        (
            5.75,
            1.55,
            1.75,
            1.2,
            "544-D vector",
            "concatenate features",
            "#E7F3EF",
            TEAL,
        ),
        (
            8.25,
            1.55,
            1.55,
            1.2,
            "Linear head",
            "scale + logistic fit",
            "#EEF5E8",
            GREEN,
        ),
        (10.25, 1.55, 0.72, 1.2, "Score", "P(fake)", "#EEF5E8", GREEN),
    ]
    for x, y, width, height, title, detail, fill, edge in nodes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.35,
            detail,
            ha="center",
            va="center",
            fontsize=7.5,
            color=MUTED,
            wrap=True,
        )
    links = [
        ((1.8, 2.05), (2.45, 2.82)),
        ((1.8, 2.05), (2.45, 1.42)),
        ((4.9, 2.82), (5.75, 2.15)),
        ((4.9, 1.42), (5.75, 2.15)),
        ((7.5, 2.15), (8.25, 2.15)),
        ((9.8, 2.15), (10.25, 2.15)),
    ]
    for start, end in links:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#84929C", "lw": 1.2},
        )
    _save_plot(figure, "architecture.png")


def _standard_robustness_figure() -> None:
    rows = load_report_metrics()["submitted_hybrid"]["conditions"]
    clean_auc = float(rows[0]["auc"])
    transformed_rows = rows[1:]
    labels = [row["condition"] for row in transformed_rows]
    changes = [float(row["auc"]) - clean_auc for row in transformed_rows]
    colors_by_family = {
        "clean": NAVY,
        "jpeg": NAVY,
        "blur": ORANGE,
        "resize": RED,
        "noise": PURPLE,
        "jitter": TEAL,
        "crop": GREEN,
    }
    colors = [colors_by_family[row["family"]] for row in transformed_rows]
    figure, axis = plt.subplots(figsize=(8.4, 6.1))
    positions = list(range(len(labels)))[::-1]
    axis.barh(positions, changes, color=colors, height=0.62)
    axis.set_yticks(positions, labels)
    axis.set_xlim(-0.12, 0.005)
    axis.axvline(0.0, color=INK, linewidth=0.9)
    axis.set_xlabel(f"Change in ROC AUC from clean ({clean_auc:.3f})")
    for position, change in zip(positions, changes):
        axis.text(
            min(change + 0.002, -0.001),
            position,
            f"{change:+.3f}",
            va="center",
            fontsize=8,
            weight="bold",
            color=INK,
        )
    _clean_axes(axis)
    figure.tight_layout()
    _save_plot(figure, "robustness_conditions.png")


def _standard_ablation_figure() -> None:
    rows = load_report_metrics()["submitted_hybrid"]["ablation"]
    names = [f"{row['model']}\n{row['training']}" for row in rows]
    clean = [float(row["clean_auc"]) for row in rows]
    robust = [float(row["robust_auc"]) for row in rows]
    positions = list(range(len(names)))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.8, 4.3))
    axis.bar(
        [position - width / 2 for position in positions],
        clean,
        width,
        label="Clean AUC",
        color=NAVY,
    )
    axis.bar(
        [position + width / 2 for position in positions],
        robust,
        width,
        label="Robust AUC",
        color=ORANGE,
    )
    axis.set_xticks(positions, names)
    axis.set_ylim(0.88, 1.0)
    axis.set_ylabel("ROC AUC")
    for values, offset, color in (
        (clean, -width / 2, NAVY),
        (robust, width / 2, ORANGE),
    ):
        for position, value in zip(positions, values):
            axis.text(
                position + offset,
                value + 0.002,
                f"{value:.3f}",
                ha="center",
                fontsize=7.5,
                color=color,
            )
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.23))
    _clean_axes(axis, grid_axis="y")
    figure.tight_layout()
    _save_plot(figure, "ablation.png")


def _standard_blind_transfer_figure() -> None:
    payload = load_report_metrics()["blind_ablation"]
    matrix = [[float(value) for value in model["auc"]] for model in payload["models"]]
    figure, axis = plt.subplots(figsize=(8.6, 3.7))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=0.9, aspect="auto")
    axis.set_xticks(range(len(payload["datasets"])), payload["datasets"])
    axis.set_yticks(
        range(len(payload["models"])),
        ["Semantic only", "Hybrid", "Hybrid + augmentation"],
    )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
    colorbar.set_label("Probability ROC AUC", fontsize=8)
    colorbar.outline.set_visible(False)
    _save_plot(figure, "blind_transfer.png")


def _standard_frequency_figure() -> None:
    audit = load_shift_audit()["historical_blind_models"]
    rows = [
        ("SID-Set", audit["sid_validation"]["hybrid_augmented"]),
        ("COCO / DALL-E", audit["wildfake_coco_dalle"]["hybrid_augmented"]),
        ("LAION / DALL-E", audit["wildfake_laion_dalle"]["hybrid_augmented"]),
    ]
    figure, (axis, detail_axis) = plt.subplots(
        1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    positions = list(range(len(rows)))[::-1]
    for position, (_, result) in zip(positions, rows):
        summary = result["branches"]["absolute_standardized_frequency"]
        axis.hlines(position, summary["p05"], summary["p95"], color=RED, linewidth=7)
        axis.plot(summary["median"], position, "o", color=INK, markersize=5)
        axis.plot(summary["maximum"], position, "|", color=RED, markersize=11)
    axis.axvline(5.475487, color=NAVY, linestyle="--", linewidth=1.2)
    axis.text(5.8, 2.25, "training p99.9 = 5.48", color=NAVY, fontsize=7)
    axis.set_yticks(positions, [name for name, _ in rows])
    axis.set_xscale("log")
    axis.set_xlim(0.5, 700)
    axis.set_xticks([1, 10, 100])
    axis.set_xticklabels(["1", "10", "100"])
    axis.minorticks_off()
    axis.set_xlabel(
        "Absolute standardized frequency value (log scale)\nbar p05-p95; dot median; tick maximum"
    )
    _clean_axes(axis)
    width = 0.34
    probability_auc = [result["probability_ranking"]["auc"] for _, result in rows]
    margin_auc = [result["margin_ranking"]["auc"] for _, result in rows]
    positions = list(range(len(rows)))
    detail_axis.bar(
        [position - width / 2 for position in positions],
        probability_auc,
        width,
        color=RED,
        label="Probability",
    )
    detail_axis.bar(
        [position + width / 2 for position in positions],
        margin_auc,
        width,
        color=NAVY,
        label="Raw margin",
    )
    detail_axis.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.9)
    detail_axis.set_ylim(0.45, 0.82)
    detail_axis.set_xticks(positions, ["SID", "COCO", "LAION"])
    detail_axis.set_ylabel("ROC AUC")
    detail_axis.legend(frameon=False, fontsize=7)
    _clean_axes(detail_axis, grid_axis="y")
    figure.tight_layout()
    _save_plot(figure, "frequency_extrapolation.png")


def _standard_native_stress_figure() -> None:
    datasets = load_native_stress()["datasets"]
    names = ["SID-Set", "COCO / DALL-E", "LAION / DALL-E"]
    keys = ["sid_validation", "wildfake_coco_dalle", "wildfake_laion_dalle"]
    clean = [
        datasets[key]["conditions"]["clean"]["probability_ranking"]["auc"]
        for key in keys
    ]
    individual = [
        datasets[key]["summary"]["individual_transform"]["mean_auc"] for key in keys
    ]
    chains = [
        datasets[key]["summary"]["platform_style_chain"]["mean_auc"] for key in keys
    ]
    positions = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.2, 4.1))
    for offset, values, label, color in (
        (-width, clean, "Clean", NAVY),
        (0.0, individual, "14 individual transforms", TEAL),
        (width, chains, "3 ordered chains", ORANGE),
    ):
        axis.bar(positions + offset, values, width, label=label, color=color)
        for position, value in zip(positions + offset, values):
            axis.text(
                position,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                fontsize=7.5,
                color=color,
            )
    axis.set_xticks(positions, names)
    axis.set_ylim(0.70, 1.02)
    axis.set_ylabel("ROC AUC")
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    _clean_axes(axis, grid_axis="y")
    figure.tight_layout()
    _save_plot(figure, "native_stress.png")


def _standard_progression_figure() -> None:
    stages = [
        ("Semantic\nbaseline", "clean .989\nrobust .918", PALE_BLUE, NAVY),
        ("Add FFT", "clean .992\nrobust .911", PALE_ORANGE, ORANGE),
        ("Add\naugmentation", "robust .956\ncomposite .972", PALE_GREEN, TEAL),
        ("Native test", "probability AUC\nabout .50", PALE_RED, RED),
        ("Branch\ndiagnosis", "frequency |z|\nabout 500", PALE_ORANGE, ORANGE),
        ("Remove FFT", "semantic\nmodel", PALE_BLUE, NAVY),
        ("Promote", "SID .992\nWildFake .903 / .913", PALE_GREEN, GREEN),
    ]
    box_width = 1.28
    step = 1.38
    figure, axis = plt.subplots(figsize=(10.8, 2.5))
    axis.set_xlim(0, 9.8)
    axis.set_ylim(0, 1.0)
    axis.axis("off")
    for index, (title, detail, fill, edge) in enumerate(stages):
        x = 0.08 + index * step
        patch = FancyBboxPatch(
            (x, 0.25),
            box_width,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.1,
        )
        axis.add_patch(patch)
        axis.text(
            x + box_width / 2,
            0.61,
            title,
            ha="center",
            va="center",
            fontsize=8.0,
            weight="bold",
            linespacing=1.0,
        )
        axis.text(
            x + box_width / 2,
            0.36,
            detail,
            ha="center",
            va="center",
            fontsize=7.0,
            color=MUTED,
            linespacing=1.1,
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(x + box_width + 0.08, 0.5),
                xytext=(x + box_width + 0.02, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#84929C", "lw": 1.0},
            )
    _save_plot(figure, "project_progression.png")


def main() -> None:
    _standard_architecture_figure()
    _standard_robustness_figure()
    _standard_ablation_figure()
    _standard_blind_transfer_figure()
    _standard_frequency_figure()
    _standard_native_stress_figure()
    _standard_progression_figure()
    print(f"Generated 7 report figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
