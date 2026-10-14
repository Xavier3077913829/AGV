# 订单需求预测数据集（任务 7）

## 文件

| 文件 | 点数 | 区间 | 说明 |
| --- | --- | --- | --- |
| `online_retail_daily.csv` | 374 | 2010-12-01 ~ 2011-12-09 | 真实电商交易按天聚合的**订单量**，字段 `date, order_count` |

## 数据来源与许可

- 数据集：**Online Retail**（英国注册的非店铺在线零售商 2010-12-01 ~ 2011-12-09 全部交易）
- 出处：UCI Machine Learning Repository，id = 352 —— https://archive.ics.uci.edu/dataset/352/online+retail
- 原始规模：**541,909 条真实交易记录**（22.6 MB xlsx）
- 使用要求：学术使用请引用 UCI 数据集页面（引用格式以该页面为准）

## 处理规则（可复现）

1. 丢弃 `InvoiceDate` 或 `InvoiceNo` 为空的行；
2. **排除退货单**（`InvoiceNo` 以 `C` 开头），只统计真实下单；
3. 按自然日分组，`order_count = 当日不同 InvoiceNo 的个数`（即"订单数"，不是行数）；
4. 日历上缺失的日子（当日无交易）用线性插值补齐并取整，保证日序列连续。

> 因此本文件的 `order_count` 定义是"**每日独立订单数**"，与内置合成序列的 `order_count`（订单件数）语义一致。

## 实测指标（本项目模型在该序列上的结果）

`horizon = 14`，回测窗口 = `min(max(horizon,7), len//5)`，日频数据自动识别季节周期 7：

| 模型 | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| 移动平均（窗口 7） | 23.358 | 33.640 | 35.30% |
| **Holt-Winters** | **17.232** | **24.182** | **24.32%** |
| ARIMA(1,1,1) | 23.340 | 30.240 | 32.71% |

- `auto` 模式按 RMSE 选中 **Holt-Winters**，未来 14 天预测合计约 1131 单；
- 序列统计：均值 69.5 单/天，范围 11 ~ 154 单/天，存在明显周周期（周末显著下降）。

## 与内置合成数据的区别（诚实提示）

1. **真实数据更难预测**：合成序列的 MAPE 通常在 5%~10%，本真实序列最好模型为 **24.32%** —— 这是正常现象，
   报告中应如实呈现，不要把合成数据的漂亮指标当作真实性能。
2. 序列只有 **374 天**，且不含促销、节假日、价格等外生变量；如需提升精度可加入外生变量（SARIMAX 外生项）或换用
   Prophet / LightGBM。
3. 存在明显周内周期，因此 Holt-Winters（季节周期 7）显著优于移动平均与 ARIMA(1,1,1) —— 这正好支撑"多模型对比 +
   自动选型"的设计。

## 重新生成

```powershell
python tools/build_public_datasets.py --retail "E:\下载文件\online+retail.zip"
```

## 在项目中的使用

```powershell
# 查看可用真实序列
GET /api/forecast/series/
# 查看真实序列数据
GET /api/forecast/sample/?series=online_retail_daily
# 直接用真实序列做多模型对比预测（auto 自动选型）
POST /api/forecast/predict/  {"series": "online_retail_daily", "horizon": 14, "model": "auto"}
```

不传 `series` 时仍使用内置合成序列，原有接口行为不变。
