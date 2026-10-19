# M3 制造业订单横截面数据（真实，但不能做时间序列）

## 文件

| 文件 | 记录数 | 原始表含义 |
| --- | --- | --- |
| `table1a_long.csv` | 360 | Table 1：耐用品**出货与新订单**（季调 / 未季调） |
| `table1p_long.csv` | 710 | Table 1：制造业**出货额**（按行业组） |
| `table2a_long.csv` | 360 | Table 2：耐用品**未完成订单与总库存** |
| `table2p_long.csv` | 380 | Table 2：制造业**新订单额**（按行业组）← 与"订单预测"最相关 |
| `table3p_long.csv` | 360 | Table 3：制造业**未完成订单额** |
| `table4p_long.csv` | 710 | Table 4：制造业**库存额** |

统一字段：`industry`（行业组）、`month`（`YYYY-MM`）、`value`（单位：百万美元）。

## 数据来源

- 来源：**US Census Bureau, Manufacturers' Shipments, Inventories, and Orders (M3) Survey**
- 官方入口：https://www.census.gov/manufacturing/m3/index.html
- 数据期：**2026 年 8 月发布期**（表内为 2026 Aug / Jul / Jun 三个月，部分表含同比与年累计列）

## ⚠️ 关键限制（必须在报告中写明）

**该数据集只包含最近 3 个月的横截面数据，没有历史时间序列，因此不能用于任务 7 的时间序列预测**
（本项目的 `_normalize_history` 要求至少 30 个有效数据点）。

它能承担的作用：

1. 作为**真实工业订单横截面数据**，展示"按行业组的新订单/未完成订单/库存"结构，用于数据分析与可视化；
2. 作为**辅助特征来源**（例如用未完成订单、库存解释出货变化），或与真实时间序列做相关性分析；
3. 演示"数据质量检查与数据不足时的处理策略"——本身就是一种工程能力，可作为报告中的一节。

**若需要长历史月度订单序列**（可做趋势/季节建模），建议补充：

- Census M3 官方历史时间序列（同一下载页面的 historical time series）；
- FRED 序列 [AMTMNO（All Manufacturing New Orders）](https://fred.stlouisfed.org/series/AMTMNO)，页面可直接导出 CSV，
  改列名为 `date, order_count` 即可被本项目直接读取。

## 重新生成

```powershell
python tools/build_public_datasets.py --m3 "E:\下载文件\数据.zip"
```

## 在项目中的使用

当前仅作为数据集文件保留（`datasets/m3/`），尚未接入 REST 接口；如需展示，建议新增一个只读接口返回
"行业组 × 月份"的透视表，并在报告中标注其横截面性质。
