# 设备状态监测模块

对应课程任务 6，使用多变量传感器数据监测设备正常、磨损和故障状态。

## 输入数据

- `temperature_c`：设备温度，单位 ℃
- `vibration_mm_s`：振动速度，单位 mm/s
- `current_a`：主轴电流，单位 A
- `spindle_speed_rpm`：主轴转速，单位 rpm
- `tool_wear`：刀具磨损率，范围 0 到 1

## 算法流程

1. 对发送到接口的 CSV 或 JSON 记录进行字段与数值校验。
2. 使用标准化器消除不同传感器量纲差异。
3. 使用随机森林分类器输出 `normal`、`worn`、`fault` 三类状态及置信度。
4. 使用温度和振动等机理阈值进行规则告警，补充模型判断。
5. 输出状态分布、特征统计、特征重要性和传感器趋势序列，供前端可视化。

## 接口

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| GET | `/api/equipment/overview/` | 诊断统计与模型指标 |
| GET | `/api/equipment/sample/` | 获取内置传感器样例 |
| GET | `/api/equipment/model/` | 模型指标与特征重要性 |
| POST | `/api/equipment/diagnose/` | 上传 CSV、提交 JSON 或使用内置样例诊断 |
| GET | `/api/equipment/records/` | 历史诊断记录 |

诊断请求示例：

```json
{
  "sample": "worn",
  "equipment_code": "CNC-01"
}
```

上传 CSV 时使用 multipart 字段 `file`。

## 真实数据集（可选）

`datasets/equipment/` 下可放入真实数据集 CSV（字段同上，另加 `label` 列），模块会自动识别：

- `ai4i_equipment.csv`：由 UCI AI4I 2020 数据集转换（10,000 行，标签 `normal/worn/fault`）。

```json
// 用真实数据集训练并查看指标
GET  /api/equipment/model/?data_source=ai4i_equipment
// 列出可用数据集
GET  /api/equipment/datasets/
// 取真实数据样本（可直接喂给 /diagnose/）
GET  /api/equipment/sample/?dataset=ai4i_equipment&limit=200
// 用真实数据训练出的模型做诊断
POST /api/equipment/diagnose/  {"dataset": "ai4i_equipment", "limit": 200, "data_source": "ai4i_equipment"}
```

不传 `data_source` 时仍使用内置合成数据训练，原有行为不变。

**实测对比（同一套模型代码）**

| 训练数据 | accuracy | F1 (macro) | 说明 |
| --- | --- | --- | --- |
| 内置合成数据 | 0.99+ | 0.99+ | 三类样本充足、分离度大 |
| AI4I 真实标签数据 | **0.8984** | **0.5042** | `worn` 仅 46 条样本，F1 仅 0.106 |

> 报告建议同时列出两套指标，说明**类别不平衡对真实数据的影响**；`ai4i_equipment.csv` 的
> `current_a` 为派生特征、`vibration_mm_s` 为派生代理，规则阈值需按现场重新标定。
> 详见 [`datasets/equipment/README.md`](../datasets/equipment/README.md)。
