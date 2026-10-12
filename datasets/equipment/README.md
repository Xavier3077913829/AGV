# 设备状态监测数据集（任务 6）

## 文件

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `ai4i_equipment.csv` | 10,000 | 由 UCI AI4I 2020 数据集转换而来，字段与 `EquipmentMonitoring/services.py` 的 `FEATURE_COLUMNS` 完全一致 |

字段：`temperature_c, vibration_mm_s, current_a, spindle_speed_rpm, tool_wear, label, machine_failure, udi`
（`machine_failure` 与 `udi` 为保留的原始信息，训练时只使用 5 个特征 + `label`。）

## 数据来源与许可

- 数据集：**AI4I 2020 Predictive Maintenance Dataset**
- 出处：UCI Machine Learning Repository，DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- 许可：**CC BY 4.0**（可自由使用与改编，需署名）
- 下载：`https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- 引用：AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository.

> ⚠️ **数据性质说明（必须在报告中写明）**：该数据集官方自述为 *"a synthetic dataset that reflects real
> predictive maintenance data encountered in industry"* —— 它是**基于真实工业过程物理模型生成的合成数据**，
> 不是现场实测数据。相比项目早期"用正态分布手工造数"，它的可信度更高（有公开发表、有物理生成机制、可引用），
> 但仍应在报告中如实说明。

## 字段映射与派生规则

| 本项目字段 | AI4I 原始列 | 转换方式 | 性质 |
| --- | --- | --- | --- |
| `temperature_c` | `Process temperature [K]` | 减 273.15 | 真实字段 |
| `spindle_speed_rpm` | `Rotational speed [rpm]` | 直接使用 | 真实字段 |
| `tool_wear` | `Tool wear [min]` | ÷ 250 分钟，截断到 [0,1] | 真实字段（归一化） |
| `current_a` | `Torque [Nm]` + `Rotational speed [rpm]` | `P = T·ω`（ω=2πn/60），`I = P/(√3·U·η)`，取 U=400V、η=0.85 | **派生特征** |
| `vibration_mm_s` | 无对应列 | `0.9 + 1.8·tool_wear + 0.6·|Δn|/1500 + 0.5·|ΔT|/40` | **派生代理（非实测）** |
| `label` | `Machine failure` + 失效模式列 | `TWF=1 → worn`；`HDF/PWF/OSF/RNF=1 → fault`；否则 `normal` | 真实标签 |

**类别分布**：`normal = 9652`、`fault = 302`、`worn = 46`（严重不平衡，比例约 210 : 6.6 : 1）。

## 实测指标（本项目模型在该数据集上的结果）

训练配置与本项目一致：`StandardScaler + RandomForestClassifier(n_estimators=260, max_depth=10, class_weight="balanced")`，
`test_size=0.25`、`random_state=42`、分层抽样：

| 指标 | 数值 |
| --- | --- |
| accuracy | **0.8984** |
| F1 (macro) | **0.5042** |

| 类别 | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| normal | 0.994 | 0.901 | 0.945 | 2413 |
| worn | 0.059 | 0.545 | 0.106 | 11 |
| fault | 0.314 | 0.868 | 0.462 | 76 |

特征重要性（从高到低）：`tool_wear 0.289` > `spindle_speed_rpm 0.250` > `current_a 0.229` >
`vibration_mm_s 0.175` > `temperature_c 0.057`。

## 与内置合成数据的区别（诚实提示）

1. 合成数据（`generate_training_frame`）三类样本充足且分离度大，指标很好看；
   本数据集**极端不平衡**（`worn` 仅 46 条），因此 **macro-F1 只有 0.50，`worn` 类基本不可分**。
2. **振动通道是派生代理**，不是实测值 —— 如需真实振动，应接入 FEMTO-ST / IEEE PHM 2012、CWRU 轴承数据集。
3. `_rule_alerts` 的阈值（电流 6.2 A、磨损 0.78 等）原本按合成数据范围标定；本数据集的电流量级约 10 A，
   因此**规则阈值需按现场重新标定**，报告中不宜把规则告警数量当作真实报警率。
4. 报告建议同时给出"合成数据 + 真实数据集"两套指标，说明模型在不同数据分布下的表现差异。

## 重新生成

```powershell
python tools/build_public_datasets.py --ai4i "E:\下载文件\ai4i+2020+predictive+maintenance+dataset.zip"
```

## 在项目中的使用

```powershell
# 查看可用数据集
GET /api/equipment/datasets/
# 取真实数据样本
GET /api/equipment/sample/?dataset=ai4i_equipment&limit=200
# 用真实数据集训练并查看指标
GET /api/equipment/model/?data_source=ai4i_equipment
# 用真实数据集模型做诊断（records 可来自上面的 sample）
POST /api/equipment/diagnose/  {"dataset": "ai4i_equipment", "limit": 200, "data_source": "ai4i_equipment"}
```

不传 `data_source` 时仍使用内置合成数据，原有接口行为不变。
