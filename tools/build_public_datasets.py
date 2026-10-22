"""把公开数据集转换成 AGV 项目可直接使用的格式（可复现，不依赖 Django）。

在项目根目录执行：

    python tools/build_public_datasets.py \
        --ai4i   "E:/下载文件/ai4i+2020+predictive+maintenance+dataset.zip" \
        --retail "E:/下载文件/online+retail.zip" \
        --m3     "E:/下载文件/数据.zip"

输出：
    datasets/equipment/ai4i_equipment.csv      设备监测（任务 6）：5 特征 + label(normal/worn/fault)
    datasets/forecast/online_retail_daily.csv  订单预测（任务 7）：date, order_count（真实交易按天聚合）
    datasets/m3/*.csv                          M3 制造业新订单/出货/库存（真实，横截面：仅最近 3 个月）

字段映射、派生公式、许可与出处见 datasets/equipment/README.md、datasets/forecast/README.md、
datasets/m3/README.md。转换规则一旦改动，请同时更新这些说明，保证报告与数据一致。
"""

from __future__ import annotations

import argparse
import io
import math
import re
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# AI4I 2020 的工艺参数假设（用于把转矩/转速换算成主轴电流，报告中须注明为派生特征）
SPINDLE_VOLTAGE = 400.0
SPINDLE_EFFICIENCY = 0.85
TOOL_WEAR_MAX_MINUTES = 250.0
NOMINAL_SPEED_RPM = 1500.0
NOMINAL_TORQUE_NM = 40.0

MONTH_TOKENS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)

# 表头脚注：行业名尾部会带上标或普通数字（如 "New Orders4"、"…industries2"）
FOOTNOTE_SUFFIX = re.compile(r"\s*[\d\u00b9\u00b2\u00b3\u2070-\u209f]+$")


def _clean_industry(raw: str) -> str:
    """清理行业名：省略号、句点，以及表头脚注数字。"""
    text = str(raw).replace("…", "").replace(".", "").strip()
    return FOOTNOTE_SUFFIX.sub("", text).strip()


def _read_csv_from_zip(zip_path: Path, suffix: str = ".csv") -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(suffix)]
        if not names:
            raise FileNotFoundError(f"{zip_path} 中没有 {suffix} 文件")
        with archive.open(names[0]) as handle:
            return pd.read_csv(io.BytesIO(handle.read()))


def _read_excel_from_zip(zip_path: Path, suffix: str = ".xlsx") -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(suffix)]
        if not names:
            raise FileNotFoundError(f"{zip_path} 中没有 {suffix} 文件")
        with archive.open(names[0]) as handle:
            return pd.read_excel(io.BytesIO(handle.read()), engine="openpyxl")


# --------------------------------------------------------------------------
# 1) AI4I 2020 -> datasets/equipment/ai4i_equipment.csv
# --------------------------------------------------------------------------
def build_ai4i(zip_or_csv: Path, out_dir: Path) -> Path:
    if zip_or_csv.suffix.lower() == ".zip":
        frame = _read_csv_from_zip(zip_or_csv)
    else:
        frame = pd.read_csv(zip_or_csv)

    temperature = frame["Process temperature [K]"] - 273.15
    speed = frame["Rotational speed [rpm]"].astype(float)
    torque = frame["Torque [Nm]"].astype(float)
    wear = (frame["Tool wear [min]"].astype(float) / TOOL_WEAR_MAX_MINUTES).clip(0, 1)

    # 主轴功率 P = T·ω，再由 P = √3·U·I·η 反推电流（派生特征）
    power_w = torque * speed * 2 * math.pi / 60.0
    current = power_w / (math.sqrt(3) * SPINDLE_VOLTAGE * SPINDLE_EFFICIENCY)

    # AI4I 不含振动通道：用磨损与负载偏差构造「派生振动代理」，README 中已声明
    speed_deviation = (speed - NOMINAL_SPEED_RPM).abs() / NOMINAL_SPEED_RPM
    torque_deviation = (torque - NOMINAL_TORQUE_NM).abs() / NOMINAL_TORQUE_NM
    vibration_proxy = 0.9 + 1.8 * wear + 0.6 * speed_deviation + 0.5 * torque_deviation

    failure_modes = frame[["HDF", "PWF", "OSF", "RNF"]].sum(axis=1)
    label = pd.Series("normal", index=frame.index)
    label[failure_modes > 0] = "fault"
    label[frame["TWF"] == 1] = "worn"

    result = pd.DataFrame(
        {
            "temperature_c": temperature.round(3),
            "vibration_mm_s": vibration_proxy.round(3),
            "current_a": current.round(3),
            "spindle_speed_rpm": speed.round(1),
            "tool_wear": wear.round(4),
            "label": label,
            "machine_failure": frame["Machine failure"].astype(int),
            "udi": frame["UDI"].astype(int),
        }
    )
    out = out_dir / "ai4i_equipment.csv"
    result.to_csv(out, index=False, encoding="utf-8")
    print(
        f"[AI4I] 写出 {out}  行数={len(result)}  "
        f"标签分布={result['label'].value_counts().to_dict()}"
    )
    return out


# --------------------------------------------------------------------------
# 2) Online Retail -> datasets/forecast/online_retail_daily.csv
# --------------------------------------------------------------------------
def build_online_retail(
    zip_or_xlsx: Path, out_dir: Path, fill_missing_days: bool = True
) -> Path:
    if zip_or_xlsx.suffix.lower() == ".zip":
        frame = _read_excel_from_zip(zip_or_xlsx)
    else:
        frame = pd.read_excel(zip_or_xlsx)

    frame = frame.dropna(subset=["InvoiceDate", "InvoiceNo"]).copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame = frame.dropna(subset=["InvoiceDate"])
    # 排除退货单（InvoiceNo 以 C 开头），只统计真实下单
    frame = frame[~frame["InvoiceNo"].astype(str).str.upper().str.startswith("C")]

    daily = (
        frame.groupby(frame["InvoiceDate"].dt.normalize())
        .agg(order_count=("InvoiceNo", "nunique"), total_quantity=("Quantity", "sum"))
        .reset_index()
        .rename(columns={"InvoiceDate": "date"})
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

    if fill_missing_days:
        series = daily.set_index(pd.to_datetime(daily["date"]))["order_count"].asfreq("D")
        filled_days = int(series.isna().sum())
        series = series.interpolate(limit_direction="both").round().astype(int)
        # 该零售商周六闭店（原始数据里没有任何周六交易），插值会凭空造出一个营业日，
        # 因此把周六显式记为 0 单，避免把补出来的数字当成真实订单量。
        closed_saturday = series.index.dayofweek == 5
        closed_days = int(closed_saturday.sum())
        series[closed_saturday] = 0
        filled = series.reset_index()
        filled.columns = ["date", "order_count"]
        filled["date"] = filled["date"].dt.strftime("%Y-%m-%d")
        daily = filled
        print(f"[Retail] 补齐缺失日 {filled_days} 天；其中周六闭店 {closed_days} 天，按 0 单计")

    out = out_dir / "online_retail_daily.csv"
    daily.to_csv(out, index=False, encoding="utf-8")
    print(
        f"[Retail] 写出 {out}  天数={len(daily)}  "
        f"订单量范围={daily['order_count'].min()}~{daily['order_count'].max()}"
    )
    return out


# --------------------------------------------------------------------------
# 3) M3 表格 -> datasets/m3/*.csv（横截面，仅最近 3 个月，不能做时间序列）
# --------------------------------------------------------------------------
def _find_header_rows(raw: pd.DataFrame) -> tuple[int, int]:
    """返回 (月份行, 年份行) 的行号。"""
    for index in range(min(20, len(raw))):
        tokens = [str(value).strip().lower().rstrip(".") for value in raw.iloc[index].tolist()]
        if sum(1 for token in tokens if token[:3] in MONTH_TOKENS) >= 2:
            return index, index + 1
    raise ValueError("未找到月份表头行")


def build_m3(zip_path: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".xlsx"):
                continue
            with archive.open(name) as handle:
                book = pd.read_excel(
                    io.BytesIO(handle.read()), sheet_name=0, header=None, engine="openpyxl"
                )

            month_row, year_row = _find_header_rows(book)
            month_tokens = [str(value).strip() for value in book.iloc[month_row].tolist()]
            # 年份行的合并单元格读进来是空值：先按原值向前填充（NaN 才能被填充），
            # 再统一转字符串，避免 "nan" 字符串把填充截断。
            year_tokens = (
                pd.Series(book.iloc[year_row].tolist())
                .ffill()
                .map(lambda value: "" if pd.isna(value) else str(value).strip())
                .tolist()
            )
            value_columns = [
                column
                for column, token in enumerate(month_tokens)
                if token.lower().rstrip(".")[:3] in MONTH_TOKENS
            ]
            if not value_columns:
                continue

            records = []
            skipped_columns: set[int] = set()
            for row_index in range(year_row + 1, len(book)):
                industry = book.iat[row_index, 0]
                if pd.isna(industry) or not str(industry).strip():
                    continue
                name_text = _clean_industry(industry)
                if len(name_text) < 3:
                    continue
                for column in value_columns:
                    value = book.iat[row_index, column]
                    if pd.isna(value) or not isinstance(value, (int, float)):
                        continue
                    month_text = month_tokens[column].lower().rstrip(".")[:3]
                    month_number = MONTH_TOKENS.index(month_text) + 1
                    year_text = "".join(
                        ch for ch in str(year_tokens[column]) if ch.isdigit()
                    )
                    if len(year_text) < 4:
                        # 年份单元格缺失：宁可跳过该列，也不要写出无法归属月份的 ????-MM
                        skipped_columns.add(column)
                        continue
                    year = int(year_text[:4])
                    records.append(
                        {
                            "industry": name_text,
                            "month": f"{year:04d}-{month_number:02d}",
                            # 原始表格列号：同一(行业, 月份)的季调/未季调/年累计列块靠它区分
                            "source_column": column,
                            "value": float(value),
                        }
                    )
            if not records:
                continue
            if skipped_columns:
                print(
                    f"[M3] {Path(name).stem}: 跳过 {len(skipped_columns)} 个无法解析年份的列"
                )
            out = out_dir / f"{Path(name).stem.replace(' ', '_')}_long.csv"
            pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8")
            written.append(out)
            print(f"[M3] 写出 {out}  记录数={len(records)}（仅最近 3 个月，不能做时间序列）")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="把公开数据集转换成项目可用格式")
    parser.add_argument("--ai4i", help="ai4i+2020+*.zip 或 ai4i2020.csv 路径")
    parser.add_argument("--retail", help="online+retail.zip 或 Online Retail.xlsx 路径")
    parser.add_argument("--m3", help="数据.zip（Census M3 表格）路径")
    parser.add_argument("--out", default=str(DATASETS_DIR), help="输出目录（默认 datasets/）")
    args = parser.parse_args()

    out_root = Path(args.out)
    equipment_dir = out_root / "equipment"
    forecast_dir = out_root / "forecast"
    m3_dir = out_root / "m3"
    for directory in (equipment_dir, forecast_dir, m3_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.ai4i:
        build_ai4i(Path(args.ai4i), equipment_dir)
    if args.retail:
        build_online_retail(Path(args.retail), forecast_dir)
    if args.m3:
        build_m3(Path(args.m3), m3_dir)
    if not any((args.ai4i, args.retail, args.m3)):
        parser.print_help()


if __name__ == "__main__":
    main()
