# 视觉部分切换到监督算法（DeepCrack）改动说明

把 `ManufacturingVision` 的主裂纹检测链路从 Frangi 传统方法切换到 DeepCrack
监督分割，并保留 Frangi 作为自动回退。工件识别（任务 5）未改动。

## 改动文件

| 文件 | 改动 |
| --- | --- |
| `ManufacturingVision/services_supervised.py` | 新增 `analyze_crack_supervised`、`annotate_crack_supervised`、`supervised_available`，并让 `get_model` 按设备缓存 |
| `ManufacturingVision/views.py` | `/api/vision/crack/` 默认走 DeepCrack，失败自动回退 Frangi；响应新增 `algorithm / fallback / coverage / max_prob / mean_prob` |
| `ManufacturingVision/views_algo.py` | `algo/status/` 返回 DeepCrack 状态 |
| `ManufacturingVision/tests.py` | 新增默认算法、DeepCrack 回退、算法状态测试；原有断言显式指定 Frangi 以保持确定性 |
| `ManufacturingVision/README.md` | 更新算法说明、参数表、接口示例 |
| `frontend/src/views/VisionView.vue` | 裂纹页增加 DeepCrack / Frangi 算法切换与对应参数、指标展示 |

## 使用方式

直接把这 6 个文件按相对路径覆盖到 `AGV` 项目根目录即可，无需改数据库表。

- 默认算法：`DeepCrack`（需要 PyTorch 和权重 `runs/deepcrack/deepcrack.pth`，
  项目里已有）。
- 未安装 PyTorch 或找不到权重时，接口自动回退 Frangi，并在响应里返回
  `"fallback": true`。
- 显式指定算法：`params` 里传 `"algorithm": "deepcrack"` 或 `"algorithm": "frangi"`。

## 验证

- 后端 4 个 Python 文件已通过语法编译检查。
- 未在本机运行 Django 测试和 Vue 构建，因为这里没有项目的 Python 虚拟环境和
  Node 依赖；覆盖到项目后建议执行：

```powershell
.\venv\Scripts\python.exe manage.py test ManufacturingVision -v 2
cd frontend; npm run build
```
