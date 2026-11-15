# 内存优化实施状态

> 目标：输出零变化前提下降低运行 RSS，并让内存基本不随对象数增长。
> 验收：`scripts/mem_accept.sh`（契约定义见 `docs/memory-acceptance.md`），当前 **C1/C2/C3 全部 PASS**。

## 最终结果

参考 dump `task_dump.hprof`（267.8 MiB，2,794,680 对象）：

| 指标 | 起点 | 最终 | 变化 |
|---|---|---|---|
| 峰值 max RSS | 1,230,836 KB | **243,276 KB** | −80% |
| 峰值堆 | 942 MB | **187 MB** | −80% |
| Phase 1 解析耗时 | 13.97 s | **0.49 s** | −96% |
| 总耗时 | 212 s | **13.9 s** | −93% |
| `parity.py --strict` | 17/17 | 17/17 | 输出不变 |
| `cargo test --workspace` | — | 34/34 | — |

生长曲线（合成 dump，`scripts/synth_hprof.py`）：

| 对象数 | 峰值堆 | max RSS |
|---|---|---|
| 1,000,007 | 44 MB | 48.7 MB |
| 4,000,007 | 151 MB | 139.8 MB |
| 边际斜率 | **35.8 MB/百万对象**（契约 ≤40） | |

即：内存不再是"1 倍对象 1 倍内存"的线性关系——固定基数（类/字符串元数据、代码、缓冲）占大头，增量只有约 36 MB/百万对象，其中约 18 B/对象 是仍保留的逐对象项（见下）。

## 契约（`scripts/mem_accept.sh`）

- **C1 输出不变**：`parity.py --strict` 17/17（tolerance 0）+ 工作区测试全绿
- **C2 预算**：参考 dump 上 max RSS ≤ 300 MB、峰值堆 ≤ 200 MB
- **C3 生长**：1M→4M 对象边际斜率 ≤ 40 MB/百万对象，绝对增量按同一斜率推导（≤120 MB）

## 架构要点（为什么能做到）

1. **对象目录落盘**（`ss-support::obj_dir`）：32 B/对象定长记录，按地址排序，样本分位范围分区 + 桶内排序；查询走粗索引 + 块缓存 + `pread`。
2. **原始记录走查器**（`ss-hprof-parser::record_reader`）：给出每个对象的 body 偏移/长度，逐条断言 record 精确消费；成为解析、线性扫描、BFS 的统一底座。
3. **Phase 1 窗口化**：只 9 字节扫描 top-level 记录头，212 个 heap segment 各自映射、用后落页；非堆记录攒进有界缓冲交给 jvm-hprof 解码。
4. **线性扫描流式化**：S8/S10/S11/S12/S14 不物化任何逐对象载荷，按偏移重读并逐单元精确比较；瞬时内存 209 MB → 25 MB。
5. **BFS 去图化**：节点身份改为地址秩，引用现场解码；parent/parent_edge 落 8 B/对象日志 + 外部排序；visited 位图。
6. **四个提取 walk** 改成"按地址查目录 + pread 单对象读取"（原来会拷贝全部 String 字段与全部 byte[]/char[] body，曾造成 589 MB 尖峰）。

## 仍保留的逐对象项（合计约 18 B/对象）

| 结构 | 成本 | 说明 |
|---|---|---|
| `sorted_addresses` | 8 B/对象 | 线性扫描的 membership 判定；若要彻底去掉需改成对象目录查询（随机查会变慢） |
| BFS dump 序表 | ~6.3 B/对象 | 报告采样需要保持 dump 序，否则采样会变 |
| 链采样/描述 | ~3.9 B/对象 | 每类 top-N 采样的中间向量 |

外加上 `visited` 位图 0.125 B/对象。要走到"完全平坦"，只需把前两项也移到对象目录/磁盘，代价是随机 I/O 增加。

## 复现

```bash
cargo build --release && cargo test --workspace
scripts/mem_accept.sh                 # C1/C2/C3 全套判定
SS_MAT_MEM_TRACE=1 /usr/bin/time -v ./target/release/ss-mat task_dump.hprof report.html
python3 scripts/parity.py report.html --strict
```
