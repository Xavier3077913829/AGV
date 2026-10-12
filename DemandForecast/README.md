# 订单需求预测模块

对应课程任务 7，使用历史订单时间序列预测未来订单量。

## 输入数据

CSV 或 JSON 序列必须包含：

- `date`：订单日期
- `order_count`：订单数量

## 预测模型

1. 移动平均：适合作为稳定需求的基准模型。
2. Holt-Winters：同时建模趋势和 7 天周期。
3. ARIMA：对差分后的自相关结构进行建模。
4. 自动模式：使用最后一段历史数据回测，以 RMSE、MAPE 作为选择依据，再使用完整历史数据预测未来。

## 接口

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| GET | `/api/forecast/overview/` | 预测统计与最近记录 |
| GET | `/api/forecast/sample/` | 获取内置订单序列 |
| POST | `/api/forecast/predict/` | 上传 CSV、提交序列或使用内置样例预测 |
| GET | `/api/forecast/runs/` | 历史预测记录 |

预测请求示例：

```json
{
  "horizon": 14,
  "model": "auto"
}
```

上传 CSV 时使用 multipart 字段 `file`，并同时提交 `horizon` 和 `model`。
