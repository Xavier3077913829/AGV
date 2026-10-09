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
