# 内存优化实施状态

> 目标：输出与 Java 参考实现完全一致的前提下，降低运行内存，并让内存基本不随对象数增长。
> 更新：2026-11-15（Java S15 对齐、debug TSV 归零、madvise 修复之后）。
> 验收：`scripts/mem_accept.sh`（契约见 `docs/memory-acceptance.md`）。当前 C1/C3 PASS；C2 的峰值堆 PASS，max RSS 需按平台解释（见下）。

## 最终结果

同一台机器（macOS arm64，release），两个真实 dump：

| 指标 | task_dump.hprof | appauditd |
|---|---|---|
| dump / 对象数 | 267.8 MiB / 2,765,511 | 567 MiB / 5,916,809 |
| 总耗时 | **18.0 s** | **61.9 s** |
| Phase 1 解析 | 3.15 s | 2.14 s |
| Phase 3 线性扫描 | 2.76 s | 13.14 s |
| Phase 3b BFS | 10.75 s | 43.32 s |
| Phase 4 报告 | 1.25 s | 3.13 s |
| 峰值 live heap（程序自报） | **188,515 KB（184.1 MB）** | **401,161 KB（391.8 MB）** |
| physical footprint | 235,586 KB（230.1 MiB） | 483,607 KB（472.3 MiB） |
| max RSS (`/usr/bin/time -l`) | 480,772 KB | 897,827 KB |
| `parity.py --strict` | **17/17** | —（基线只覆盖 task_dump） |
| Java 数据对账 | 17 项指标 + TSV 全一致 | headline / problem_stats / boxed / s8 / s15 全一致 |

> **内存口径**：macOS 的 `max RSS` 会统计 mmap 的 dump 文件页（task_dump 为 268 MiB），`MADV_DONTNEED` 不会立即归还这些干净页；因此本机 RSS 明显高于 Linux 上用 `mem_accept.sh` 测得的数值（`ec589b7` 时期为 243 MB，S15 改动后未在 Linux 复测）。进程私有内存应看 `physical footprint` / `peak live heap`，两者都远低于 RSS。

### 生长曲线（合成 dump，`scripts/synth_hprof.py`）

| 对象数 | 峰值 live heap | physical footprint | max RSS |
|---|---|---|---|
| 1,000,007 | 45,092 KB（44.0 MB） | 51,545 KB | 61,243 KB |
| 4,000,007 | 154,959 KB（151.3 MB） | 161,678 KB | 196,739 KB |
| 边际斜率 | **35.8 MB / 百万对象**（契约 ≤ 40） | | |

内存不再是"1 倍对象 1 倍内存"：固定基数（类/字符串元数据、代码、缓冲）占大头，增量约 36 MB/百万对象，其中约 18 B/对象 是仍保留的逐对象项（见下）。

## 契约（`scripts/mem_accept.sh`）

- **C1 输出不变**：`parity.py --strict` 17/17（tolerance 0）+ `cargo test --workspace` 34/34 —— **PASS**
- **C2 预算**：参考 dump 上峰值堆 ≤ 200 MB、max RSS ≤ 300 MB
  - 峰值堆 184.1 MB —— **PASS**
  - max RSS 本机 macOS 为 480.8 MB，其中 ~268 MiB 是 mmap 文件页，footprint 为 230.1 MiB；Linux harness 在 S15 改动前测到 243 MB，改动后未复测。按私有内存口径 **PASS**，按 macOS RSS 字面值不可直接比较。
- **C3 生长**：1M→4M 边际斜率 ≤ 40 MB/百万对象 —— **35.8，PASS**

## 2026-11-15 变更（Java 对齐）

1. **Section 15 两阶段复刻**：线性扫描 `ArrayCollectionScanner`（候选类 `isObjectArray && !isFinalizable && refChainOverhead > dupThreshold`、raw 引用字节含 null 的 Java hash/相等、按类 0.1% 阈值、fastutil 第一 bucket 链）产生报告；非 null 时由 BFS 阶段 `ClassDominatedSizeComputer.finalizeReports` 用真实 array body 覆盖。两个真实 dump 的线性阶段报告均为 null，因此 S15 与 Java 一样只渲染空 section。
2. **debug TSV 归零**：`problem_stats.tsv`、`boxed_summary.tsv`、`s8_summary.tsv`、`s15_dup_obj_arrays.tsv` 在 task_dump / appauditd 上与 Java 排序后 **0 diff**。修复点：`category_of` 改为镜像 Java `ContainerRegistry` 注册类别；CSLM 按 `CollectionContainerInstance` 用 ×2；WeakHashMap 一个已被 BFS 访问的实例扣除 88B。
3. **madvise 修复**：`walk_heap` 不再每对象对整个已扫前缀 `madvise`，改为 24 MB 窗口成段释放；随机 body 读取的整图释放 stride 128 → 2048。appauditd Phase 3 从 206.7 s 降到 13.1 s，总耗时 424 s → 61.9 s。
4. **OOM 线程确定化**：按 serial 升序选择，线程列表以 serial 作为同名 tie-breaker，报告不再因 HashMap 顺序漂移。

## 架构要点（为什么能做到）

1. **对象目录落盘**（`ss-support::obj_dir`）：32 B/对象定长记录，按地址排序，样本分位范围分区 + 桶内排序；查询走粗索引 + 块缓存 + `pread`。
2. **原始记录走查器**（`ss-hprof-parser::record_reader`）：给出每个对象的 body 偏移/长度，逐条断言 record 精确消费；成为解析、线性扫描、BFS 的统一底座。
3. **Phase 1 窗口化**：只 9 字节扫描 top-level 记录头，212 个 heap segment 各自映射、用后落页；非堆记录攒进有界缓冲交给 jvm-hprof 解码。
4. **线性扫描流式化**：S8/S10/S11/S12/S14/S15 不物化任何逐对象载荷，按偏移重读并逐单元精确比较。
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

# macOS：/usr/bin/time -l；Linux：/usr/bin/time -v
SS_MAT_MEM_TRACE=1 /usr/bin/time -l ./target/release/ss-mat task_dump.hprof /tmp/task.html
python3 scripts/parity.py /tmp/task.html --strict

# Linux 上的完整契约（C1/C2/C3）
scripts/mem_accept.sh
```
