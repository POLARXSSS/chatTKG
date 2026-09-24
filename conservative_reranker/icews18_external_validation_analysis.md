# ICEWS18 共享服务器实验与结果分析

## 运行约束与资源配置

本次实验未终止、调低优先级或迁移任何其他 `root` 用户进程。候选生成阶段固定使用 CPU 24--47，共 24 个进程；子进程 `nice=10`，且每个进程的 BLAS/OpenMP 线程数限制为 1。GPU 仅供体量较小的训练阶段使用，启动时选择利用率为 0% 的 GPU 2；若启动时利用率高于阈值，代码会自动回退到 CPU。

执行命令：

```bash
python conservative_reranker/run_experiment.py \
  --dataset icews18 \
  --rules output_external_cameo/conservative_reranker_icews18_s13/prefix_rules.json \
  --output-dir output_external_cameo/conservative_reranker_icews18_s13 \
  --queries-per-relation 64 \
  --num-processes 24 \
  --rule-processes 24 \
  --epochs 40 \
  --patience 6 \
  --device auto \
  --cpu-affinity 24-47 \
  --nice-level 10 \
  --gpu-index 2 \
  --max-gpu-utilization 20 \
  --seed 13
```

候选生成耗时约 74.1 分钟，训练与评估约 115.8 秒，总运行约 76 分钟。候选档案包含 16,336 个查询和 3,528,427 个候选。

## 确认集结果

确认集包含 3,067 个严格按时间排序的伪未来查询，未读取 validation 或 test。

| 模型 | MRR | Hits@1 | Hits@3 | Hits@10 | answer coverage |
|---|---:|---:|---:|---:|---:|
| TLogic | 0.324913 | 0.229540 | 0.373329 | 0.507010 | 0.733616 |
| learned_base | 0.347557 | 0.257255 | 0.396479 | 0.513531 | 0.733616 |
| learned_all | **0.371936** | **0.273557** | 0.425823 | 0.551353 | **0.795566** |
| learned_official | 0.369911 | 0.270949 | **0.431040** | **0.552005** | 0.795566 |
| learned_random | 0.369019 | 0.270949 | 0.425497 | 0.546136 | 0.795566 |

主要配对结果：

- `learned_base - TLogic`: MRR +0.022644，95% moving-block CI [0.021367, 0.023800]，17/17 时间块为正。在覆盖率完全相同的情况下仍有稳定增益，明确证明排序质量是独立瓶颈。
- `learned_all - TLogic`: MRR +0.047023，CI [0.042612, 0.052414]，覆盖率增加 6.20 个百分点。候选扩展与学习排序结合后获得最佳总体 MRR。
- `learned_official - learned_all`: MRR -0.002025，CI [-0.005215, 0.000570]；official 没有优于不分组的 all-history。
- `learned_official - learned_random`: MRR +0.000892，CI [-0.000955, 0.002770]，且两个时间方向之一为负；无法证明 official CAMEO 语义优于等规模随机分组。

## 覆盖率诊断

完整候选档案中的 base answer coverage：

- 截断前：12,426 / 16,336 = 0.760651
- 截断后：11,785 / 16,336 = 0.721413
- union：12,795 / 16,336 = 0.783239

仅候选上限截断就丢失了 641 个可覆盖查询，即 3.92 个百分点。这个损失足够大，因此在设计新检索器前，先提高 base/union candidate cap 是成本最低且信息量最高的实验。

## 结论与下一步

本次结果支持以下结论：

1. 排序问题确实存在：`learned_base` 在覆盖率不变时稳定提升 MRR。
2. 覆盖率同样重要：`learned_all` 的额外覆盖带来更大总体收益，且当前为工程上最好的参考模型。
3. 知识语义贡献尚未成立：official 不优于 learned_all，且与 learned_random 的差异置信区间跨零。
4. 预先声明的推进门槛未通过：official 的原正确 top-1 保留率为 97.02%，低于 99%；语义对照也未通过。

下一步优先进行标签盲的 candidate-cap 敏感性实验，例如将 base cap 从 256 提高到 512、union cap 从 384 提高到 640，并保持数据切分、规则、随机种子和训练协议不变。若覆盖率和 MRR 的收益仍未饱和，再考虑增加新的外部知识检索器。若目标是形成 knowledge-driven 研究贡献，则之后必须要求 official 相对 all/random 的配对置信区间严格为正，并恢复至少 99% 的原正确 top-1 保留率。

当前结果仍属于 ICEWS18 的 train-only pseudo-future 内部确认，不能表述为官方 test benchmark 结果。
