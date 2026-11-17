# 内存优化实施状态

> 目标：输出零变化前提下把 RSS 降下来，并让内存基本不随对象数增长。
> 验收：`scripts/mem_accept.sh`（定义见 `docs/memory-acceptance.md`）。

## 最终实测（Linux，`/usr/bin/time -v` + `SS_MAT_MEM_TRACE=1`）

| dump | 对象数 | 总耗时 | 峰值 live heap | max RSS | 输出核对 |
|---|---|---|---|---|---|
| `task_dump.hprof`（268 MiB） | 2,794,680 | **7.49 s** | **151,585 KB** | **227,620 KB** | parity 17/17 |
| `appauditd-20261114.hprof`（567 MiB） | 5,916,809 | **15.48 s** | **235,434 KB** | **440,492 KB** | Java 夹具 4/4 |
| `536803-live.hprof`（8.0 GiB，ADG 生产） | **120,190,295** | **887.89 s** | **479,015 KB** | **946,544 KB** | 见下 |

对象数涨 43 倍（279 万 → 1.2 亿），峰值堆只涨 3.2 倍（148 → 468 MB），max RSS 4.2 倍（222 → 924 MB）——**不再是线性**，且最大那份仍在 1 GB 以内。

改造前同一份 task_dump：942 MB 峰值堆 / 1,230,836 KB max RSS / 212 s。

## 各轮改造

| 轮次 | 内容 | 效果（大 dump） |
|---|---|---|
| M0 | `mem_trace` + 计数分配器，阶段/热点采样 | 所有后续优化的定位依据 |
| M1 | 去掉数组 body 的重复拷贝、String 字段借用 | — |
| M2 | 对象目录 `obj_dir`（32 B/对象、地址排序、块缓存）、原始记录走查器 `record_reader`、`page_window`、`external_sort` | 三张地址 hash 表 → 磁盘目录 |
| M4 | `EdgeTable` 扁平化、地址索引排序二分、极大对象反查表 265 MB → 22 MB、`O(classes×objects)` → 单趟 | 峰值 942 → 290 MB（task_dump） |
| Phase 1 | 按 segment 窗口映射 + 逐段落页，非堆记录攒有界缓冲 | Phase 1 14 s → 0.5 s（task_dump） |
| A | 删 `sorted_addresses`（8 B/对象） | 大 dump 峰值 6.69 → 5.65 GB |
| B+C | S8/S11/S14 全部溢写 + 按哈希分组重放，按偏移比较 body | 扫描段 5.37 GB → **209 MB** |
| D | BFS 改"frontier 按偏移排序 → 顺序读 → 目录升序归并"；`root_of` 落盘 | 遍历 48 min → 11 min |
| chain | 引用链改溢写 + 有界 top-K；极大对象反查限定候选集 | `chain-samples` 3.71 GB → 308 MB |
| report | 采样对象 body 改 4 KB 前缀（原整块读，单笔 2.13 GB） | 报告相 2.44 GB → 平 |
| seq-io | 重复重放按偏移顺序读；链采样改外排 + 目录归并 | Phase 3 444 → 226 s，chains 203 → 95 s |
| rss | 大页、滞回量、scratch 映射释放、PageSweeper 线程、读步长 | max RSS 5.2 GB → **946 MB** |

## 等价性怎么证明的（不是只靠 parity）

- task_dump `parity.py --strict` **17/17**、`cargo test --workspace` **35 绿**
- `objects.tsv`（279 万对象逐元组，262 MB）、`objects_summary.txt`、`roots.tsv` **逐字节相同**
- 报告的 Section 1/2/5/6/7/15/19/20/21/22/23/24 与改造前**逐字节相同**
- `appauditd` Java 夹具 `problem_stats`(47,308 行)/`boxed_summary`(23,107)/`s8_summary`(2)/`s15_dup_obj_arrays`(14) **4/4 PASS**
- BFS 改造另行把 279 万对象的 `(parent, field_key, root)` 全量比对，差异 0
- console 汇总行逐字未变：`296654/154204/43441`、`382570/122965/8/151360`、`614688/288060/97222`、`787083/215900/8/176688`

## 仍保留的逐对象项

- `visited` 位图 0.125 B/对象（1.2 亿 = 15 MB）
- 类/字符串名表按"类数"增长（该 dump 16,887 类）
- parent 日志 8 B/对象、dump 序表 4 B/对象：**在磁盘**，边写边还页
- S9 的 `bad_collection_records`（报告按类计数需要逐实例行）约 115 MB（大 dump）

## 已知限制

- 报告本身有运行间不确定性（`ENCODED_STRINGS` id、链节点顺序），整文件 `cmp` 不能当闸门
- `--disk-budget-gb` 默认 4 GB：1.2 亿对象需要 7.4 GB，必须显式调大（尚未改为按对象数自适应）
- 被 `SIGKILL`/OOM 杀掉的进程仍会留下 pid 临时目录（`Drop` 不执行）
- 大 dump 的 max RSS 里仍有约 470 MB 是 dump 文件页；再压需要把顺序读改成 `pread` 进堆缓冲

## 复现

```bash
cargo build --release && cargo test --workspace
scripts/mem_accept.sh
SS_MAT_MEM_TRACE=1 /usr/bin/time -v ./target/release/ss-mat <dump> report.html
python3 scripts/parity.py report.html --strict
python3 scripts/java_parity.py <SS_MAT_DEBUG_DIR> testdata/<dump>/java
```
