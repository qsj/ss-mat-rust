# SS-Mat Rust 内存优化方案：从 2.0 GB 到 ~200 MB 常驻

> 状态：**已实施（2026-11-15）** —— 对象目录、记录走查器、流式线性扫描、BFS 去图化、窗口化 mmap 均已落地；task_dump 峰值 live heap 188.5 MB、appauditd 401.2 MB，边际增长约 35.8 MB/百万对象（仍保留约 18 B/对象）。完整实测、与 Java 的对齐状态和剩余工作见 `docs/memory-reduction-status.md`。
> 目标：分析任意大小的 `.hprof` 时，进程私有内存保持在 200 MB 量级，不随 dump 文件/对象数线性增长；可作为常驻后台服务运行。
> 前提：所有统计结果与当前版本、Java 参考实现保持零差异（`scripts/parity.py` 17/17，tolerance=0）。

---

## 1. 背景与目标

把 Java 版 SS-Mat 重写为 Rust 的一个主要动机是降低内存占用。当前 Rust 版在 appauditd 的 595 MB dump 上峰值 RSS 约 **2.0 GB**，虽然已经低于 Java 版建议的 `-Xmx`（文件大小 GB + 2 ≈ 2.6 GB），但仍随 dump 规模线性增长，无法作为常驻服务长期运行。

本方案的目标：

| 指标 | 目标 |
|---|---|
| 进程私有内存（anon/dirty footprint） | **≤ 200 MB**（appauditd/task_dump 级别 dump） |
| 内存随 dump 增长 | 仅 `visited` 位图（1 bit/对象）等可忽略项；对象元数据、数组 body、图结构全部落盘 |
| 数据正确性 | 与当前实现、Java 参考实现完全一致（零容差） |
| 运行方式 | 后台常驻、单任务、可被 SIGTERM 中断并清理临时文件 |

### 1.1 内存口径

必须区分两种“内存”，否则优化目标会被误读：

1. **进程私有内存**（本方案的目标）：堆分配 + 脏页，`footprint`/`ps` 中的 anon 部分。
2. **文件页缓存**：mmap 的 dump 文件、`pread` 读入的内核页缓存。它们不属于进程私有内存，系统内存紧张时可回收，但**会计入 RSS**（mmap 方式下）。

因此方案要求：

- 不再整份 `mmap` 并常驻 dump（当前直接贡献 567 MB RSS），改为**窗口化访问**：顺序扫描用 mmap + 定期 `madvise(MADV_DONTNEED)`，随机读取用 `pread` 到固定大小缓冲；
- 所有临时文件同理：顺序扫描用窗口 mmap，随机读写用 `pread`/`pwrite`，不靠整文件映射。

> 验收时用 `/usr/bin/time -l` 的 max RSS、`footprint` 和 `vmmap -summary` 三个指标交叉验证，并说明其中文件页缓存占多少。

---

## 2. 现状实测（appauditd，595 MB dump）

复现命令：

```bash
cd /Volumes/KODAK/qsj/testproject/ss-mat-rust
/usr/bin/time -l ./target/release/ss-mat /tmp/appauditd-20261114.hprof /tmp/appauditd-mem.html
```

| 指标 | 数值 |
|---|---|
| dump 文件 | 594,605,733 B（567 MiB） |
| 峰值 max RSS | 2,077,196,288 B ≈ 1.94 GiB（2.08 GB） |
| peak memory footprint (`time -l`) | 2,106,149,576 B ≈ 1.96 GiB |
| 总耗时 | 424 s（7 分 04 秒） |
| 报告体积 | 4.7 MB |
| 报告内堆总量 | 390,966 KB（live 263,783 KB / garbage 127,183 KB） |

分阶段 RSS（每 1.5 s 采样，日志带时间戳）：

| 阶段 | 耗时 | RSS 峰值 | 说明 |
|---|---|---|---|
| Phase 1 解析建索引 | 29.5 s | 1.13 GB | 对象索引 + 字符串表 + mmap 页 |
| Phase 3 线性扫描 | 5.3–6.1 s | **1.80 GB 采样 / 2.08 GB 真实峰值** | 峰值所在 |
| Phase 3b BFS | 382.7 s | 1.80 GB → 1.30 GB | 时间长，但不是内存峰值 |
| Phase 4 生成报告 | 6.3 s | 1.37 GB | 只增加约 50 MB |

Phase 3 期间（RSS 1.7 GB 时）`vmmap -summary`：

```text
mapped file        567.1M resident      ← dump 文件页
MALLOC_LARGE       833 MB dirty
MALLOC_SMALL       397 MB dirty
MALLOC_REALLOC       9 MB
Physical footprint 1.24 GB
```

BFS 阶段 `vmmap`：malloc 已分配 1.1 GB、共 283 万次分配，其中 756 MB 已被压缩/换出。

---

## 3. 内存构成分析

按代码结构 + 本次 dump 的精确计数（5,916,809 对象、787,083 原始数组、606,058 对象数组、606,687 字符串、47,315 类）：

| 占用项 | 大小 | 代码位置 | 性质 |
|---|---|---|---|
| mmap dump 文件页 | 567 MB | `hprof-parser`、`linear_scan`、`heap_traverser` | 整份驻留 |
| `address_to_index`（地址→对象索引） | ≈143 MB | `hprof-parser/src/hprof_wrapper.rs:348` | 5.9M 条目 × 16 B，全程不释放 |
| `object_to_class_idx`（地址→类索引） | ≈143 MB | `hprof-parser/src/hprof_wrapper.rs:349` | 第二份 5.9M 条目，仅线性扫描用，却全程驻留 |
| `PrimArrayData.raw_body` | 182 MB + 19 MB Vec 头 | `traversal/src/linear_scan.rs:485` | 原始数组 body 第 1 份 |
| `prim_body_map`（`.clone()`） | 182 MB + 19 MB Vec 头 | `traversal/src/linear_scan.rs:532-535` | 为 String 扫描复制第 2 份，生命周期到 `run()` 结束 |
| `ObjArrayData.element_refs` | 56 MB（7,345,913 个 u64）+ 15 MB Vec 头 | `traversal/src/linear_scan.rs` | 对象数组引用序列 |
| `StringSpan`（`content` + `chars`） | ≈50–80 MB | `traversal/src/linear_scan.rs:1460` | 614,688 个 String 的两份内容 |
| `string_table` + 类元数据 | ≈100 MB | `hprof-parser/src/hprof_wrapper.rs` | 606,687 条名称 + 47,315 类 |
| BFS `objects: Vec<ObjectEntry>` | 226 MB | `heap_traverser.rs:134` | 5.9M × 40 B |
| BFS `addr_to_idx` | ≈143 MB | `heap_traverser.rs:135` | 第三份地址表 |
| BFS `outgoing_refs: Vec<Vec<OutEdge>>` | 142 MB 头 + 边 + malloc 开销 | `heap_traverser.rs:322` | 590 万个小 Vec |
| BFS `parent/parent_edge/root_of` | 71 MB | `heap_traverser.rs` | 3 × 4 B/对象 |
| malloc 头/对齐/哈希表容量 | 数百 MB | — | 283 万次分配，hashbrown 按 2 的幂扩容 |

### 3.1 为什么峰值出现在 Phase 3

- Phase 3 开始前，`HeapDump` 已持有两张 5.9M 条目地址表（286 MB）+ 字符串表 + 类元数据；
- Phase 3 又收集全部数组/String 数据：原始数组 body 被复制两份（364 MB），对象数组引用 56 MB，`spans` 再复制字符串内容；
- 此时 dump 文件页（567 MB）也在驻留；
- 三者叠加 = 1.7–2.0 GB。

BFS 阶段虽然同样庞大（对象表 226 MB + 地址表 143 MB + 边表 142 MB + parent 数组 71 MB ≈ 700 MB），但它发生在 Phase 3 之后、Phase 3 的临时向量已释放，所以没有超过 Phase 3 的峰值。

**结论：没有一项是“必须全量在内存”的。** 对象 body 本来就在 dump 文件里，可按偏移随用随读；数组内容、边、parent 都可以落盘。

---

## 4. 目标架构

四个原则：

1. **对象目录落盘**：单遍扫描建按地址排序的定长记录文件，替代三张 HashMap。
2. **全程流式 + 外部排序**：不再收集 `Vec<PrimArrayData>`、`Vec<ObjArrayData>`；重复检测用“分桶 + 桶内排序 + 顺序扫组”。
3. **BFS 不建图**：队列只存对象索引，出队时按 `file_offset` 读 body、现场解码字段、现场解析子对象地址。
4. **有界缓冲**：目录页缓存、排序、队列、I/O 都有固定上限，超限落盘。

```text
                    ┌──────────────────────────────────────────────┐
  dump.hprof ──────▶│ 单遍扫描（顺序读，窗口化 mmap）               │
                    │  · 类元数据 / roots / 字符串名表（小，驻留）  │
                    │  · 每对象 → objdir 记录（顺序写）             │
                    └───────────────┬──────────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────────┐
                    │ objdir（按地址排序的定长文件，32 B/对象）      │
                    │  + 内存粗索引（每 4096 条一个锚点）           │
                    └───────┬───────────────────────┬──────────────┘
                            ▼                       ▼
            ┌───────────────────────┐   ┌──────────────────────────┐
            │ 流式统计（多 pass）    │   │ BFS（不建图）             │
            │  · String → 数组读取   │   │  · 队列分块，超限落盘     │
            │  · 重复检测 → 外部排序 │   │  · 现场解码 body          │
            │  · 坏集合/坏数组聚合   │   │  · visited 位图（驻留）   │
            └───────────┬───────────┘   │  · parent 日志 append     │
                        │               └───────────┬──────────────┘
                        ▼                           ▼
            ┌──────────────────────────────────────────────────────┐
            │ 聚合结果 + top-N（小，驻留） → 流式生成 HTML 报告      │
            └──────────────────────────────────────────────────────┘
```

---

## 5. 关键设计

### 5.1 对象目录 `objdir`

定长记录，32 字节：

```text
ObjRec {
    addr: u64,          // 对象地址（排序键）
    file_offset: u64,   // 该对象记录在 hprof 中的字节偏移
    class_idx: u32,     // 类索引
    kind: u8,           // 0=instance 1=prim array 2=obj array 3=class object
    elem: u8,           // 元素宽度（prim）或 oop 宽度（obj array）
    len: u32,           // 数组长度
    flags: u16,         // 保留（generation 等）
}
```

- **排序**：按地址高 8 位分 256 个桶，每桶在内存中排序（本 dump 每桶约 2.3 万条 = 0.7 MB），再顺序拼接成全局有序文件；不引入外部归并排序的复杂度。
- **粗索引**：每 4096 条记录保留一个 `(addr, file_pos)` 锚点；5.9M 对象仅 1445 条 × 16 B ≈ 23 KB。
- **查询**：粗索引二分 → 一次 `pread` 取 128 KB 块（4096 × 32 B）→ 块内二分。配 32 MB LRU 块缓存，利用对象地址的局部性。
- **替换**：`address_to_index`、`object_to_class_idx`、BFS 的 `addr_to_idx` 三张表全部删除。
- **file_offset 来源**：HPROF 顶层记录格式为 `tag u8 + time u32 + length u32 + body`，自写一个记录遍历器即可得到每个 sub-record 的文件偏移，不需要修改 `jvm-hprof`。

API 草案：

```rust
pub struct ObjDir { /* file: File, coarse: Vec<(u64,u64)>, cache: BlockCache */ }

impl ObjDir {
    pub fn lookup(&self, addr: u64) -> Option<ObjRec>;   // ~1 次 pread
    pub fn get(&self, idx: u32) -> ObjRec;              // 顺序读，走缓存
    pub fn len(&self) -> u64;
}
```

### 5.2 流式线性扫描

现状：Phase 3 把 78.7 万原始数组、60.6 万对象数组、61.5 万 String 全量收集后再处理。

改为按类别分 pass，每个 pass 顺序扫 dump，内存只保留当前对象：

| pass | 内容 | 内存 |
|---|---|---|
| String pass | 解析 String 实例，经 objdir 查 `value` 数组 offset，直接读 body 解码 | O(1) |
| Prim array pass | 逐个数组计算零值统计、哈希、内容 → 外部排序 | O(排序缓冲) |
| Obj array pass | 逐个数组统计空/单元素/全 null/稀疏；引用序列哈希 | O(1) |
| Collection pass | 容器字段遍历、backing array 定位、坏集合判定 | O(1) |

要点：

- **删除 `prim_body_map` 副本**：String 扫描按需查 objdir + 读 body，省掉 200 MB 冗余；
- **重复检测外部排序**：把 `(content_hash, addr, len)` 追加写盘，按 hash 高位分 256 桶，桶内排序，再顺序扫组累计“组数/浪费字节/top-N”；
- **坏集合/坏数组**：只保留聚合计数 + top 50 明细，不保留 40 万条记录副本（当前 `BadCollectionRecord` 全量驻留）；
- **top-N**：固定大小堆，内存 O(N)。

### 5.3 BFS 去图化

- `visited`：位图，1–3 bit/对象，驻留内存（本 dump 0.74 MB，1 亿对象 12.5 MB）；
- 队列：`ChunkedQueue<u32>`，上限 32 MB，超出部分写盘、按批回读（`crates/support` 已有 `ChunkedQueue` 骨架，需增加溢出文件）；
- 出队：拿对象索引 → objdir 查 `file_offset` → 读 body → 解码字段/数组元素 → 得到子对象地址；**不存 `outgoing_refs`**；
- `parent/parent_edge`：不维护随机写数组。发现子节点时顺序 append `(child_idx, parent_idx, field_key)` 到日志，BFS 结束后按 `child_idx` 外部排序一次，得到链式查询文件（本 dump 约 4.2M × 12 B ≈ 50 MB 磁盘，内存只占排序缓冲）；
- `root_of`：按 GC root 分别记录，落盘或只保留每类的聚合计数 + 树结构所需的最小信息；
- 引用链：BFS 结束时把报告需要的链（top 50 类、每类 ≤10 条）**固化成小结构**，报告不再持有 `objects/parent` 数组（当前 `TraversalData` 被 `stats.traversal_data` 持有到报告结束）。

可选加速（二选一）：

1. objdir 配 32 MB 页缓存（简单，不改语义）；
2. frontier 按地址排序后与 objdir 做 merge join，把随机读变顺序读（更快，代码更复杂）。

### 5.4 窗口化 mmap / pread

- 保留 `jvm-hprof` 对整文件的解析方式，但增加 `PageWindow`：每消费 16 MB 就 `madvise(MADV_DONTNEED)` 丢弃身后页；顺序扫描时 RSS 恒定在窗口大小；
- BFS 的随机 body 读取用 `pread` 到 1 MB 临时缓冲，避免页缓存计入进程 RSS；
- 临时文件同理：顺序读写用 mmap + 窗口，随机写用 `pwrite` 批量缓冲。

### 5.5 内存预算

| 项 | 预算 |
|---|---|
| 类元数据 + 名称（47k 类；随类数而非对象数增长） | 25–40 MB |
| 聚合统计 + top-N（报告用） | 10–15 MB |
| objdir 页缓存（可调） | 32 MB |
| 外部排序缓冲 | 32 MB |
| BFS 队列 + 位图 + 批缓冲 | 24–32 MB |
| I/O 缓冲、代码、栈、杂项 | 15–20 MB |
| **合计** | **~140–170 MB** |

运行时参数：

```text
--mem-budget 200m      私有内存上限，超限时强制 flush / 丢弃页缓存
--temp-dir /path       临时目录（本地盘）
--disk-budget 4g       临时文件配额，启动前预检
--cache-mb 32          objdir 页缓存大小
```

---

## 6. 分阶段落地

每个阶段都必须在 appauditd + task_dump 上回跑：

```bash
cargo test --workspace
python3 scripts/parity.py /tmp/<report>.html --strict
```

| 阶段 | 内容 | 预期峰值 | 工期 |
|---|---|---|---|
| **M0 度量** | `SS_MAT_MEM_TRACE`：每阶段输出 anon RSS/footprint、分配次数；CI 断言内存上限 | 2.0 GB（现状） | 0.5 天 |
| **M1 零算法瘦身** | 去掉 `prim_body_map` 克隆；合并两张地址表；BFS 复用同一张表；Phase 3 向量用完即释放 | ~1.2 GB | 1–2 天 |
| **M2 objdir 落盘** | 记录遍历器（带 file_offset）、分桶排序、粗索引 + 块缓存；替换三张 HashMap | ~600–800 MB | 2–3 天 |
| **M3 流式扫描 + 外排** | S8/S9/S10/S11/S14 全部流式；hash 分桶排序；坏记录只留聚合 + top-N | ~300–400 MB | 3–4 天 |
| **M4 BFS 去图化** | 队列分块落盘、现场解码、parent 日志外排、链固化 | **~200–250 MB** | 3–5 天 |
| **M5 收尾** | 报告流式输出、`--mem-budget`/`--temp-dir`/磁盘配额、崩溃清理、服务化 | ≤200 MB | 2 天 |

单人约 **2–2.5 周**。M1 就能砍掉 40%，且不改变任何统计逻辑，风险最低。

### 6.1 代码落点

| 新增/修改 | 位置 |
|---|---|
| 记录遍历器（tag/length/offset） | `crates/hprof-parser/src/record_reader.rs`（新） |
| objdir + 粗索引 + 块缓存 | `crates/support/src/obj_dir.rs`（新） |
| 外部排序（分桶/归并） | `crates/support/src/external_sort.rs`（新） |
| 窗口化 mmap | `crates/support/src/page_window.rs`（新） |
| 流式线性扫描 | `crates/traversal/src/linear_scan.rs` 重构 |
| BFS 去图化 | `crates/traversal/src/heap_traverser.rs` 重构 |
| 链固化 / 报告解耦 | `crates/stats/src/traversal_data.rs`、`crates/report` |
| 内存预算与临时目录 | `crates/cli/src/main.rs`、`crates/support` |

---

## 7. 代价

| 项 | 影响 |
|---|---|
| **时间** | BFS 从“内存图遍历”变成“目录查询 + 读 body”，本 dump 预计 7 分钟 → 15–25 分钟；SSD + 页缓存好会接近下限。作为后台服务可接受 |
| **磁盘** | 临时文件约为 dump 的 0.4–0.6 倍：objdir 189 MB + parent 日志 50 MB + 外排 55 MB ≈ 300 MB；10 GB dump 约 3–5 GB。需要 `--temp-dir`、启动前空间预检、退出清理、崩溃残留清理 |
| **复杂度** | objdir 查询、外部排序、崩溃恢复是新增风险；用零容差 parity 回归锁住语义 |

不建议用 SQLite 做中间存储：定长记录文件 + 排序足够快，且没有额外的进程内存和依赖。

---

## 8. 验收标准

1. **内存**
   - `/usr/bin/time -l` max RSS + `footprint` 双指标；
   - appauditd ≤ 250 MB、task_dump ≤ 200 MB；
   - 10 GB 合成 dump：对象数 ×10，内存曲线基本平坦，只有时间/磁盘线性增长。
2. **数据**
   - `scripts/parity.py --strict` 17/17（全部 tolerance=0）；
   - Java 侧 `headline.tsv` / `problem_stats.tsv` / `s8_summary.tsv` 等全字段一致。
3. **服务化**
   - 单实例锁；SIGTERM 清理临时目录；内存看门狗（超预算强制 flush/丢弃页缓存）；临时目录启动时清理残留。

---

## 9. 风险与备选

| 风险 | 缓解 |
|---|---|
| BFS 随机读导致大幅变慢 | objdir 页缓存 32 MB；frontier 排序后 merge join；`madvise(MADV_RANDOM)` |
| 临时磁盘耗尽 | `--disk-budget` 预检；按哈希桶大小估算峰值；失败时清理并报错 |
| `MADV_DONTNEED` 后重复缺页 | 窗口 16 MB + 顺序预读；只在顺序 pass 使用 |
| 语义漂移 | 每个里程碑跑 17/17 + Java TSV 对账；统计逻辑改动单独提交 |
| 200 MB 太紧 | 先做 M1–M3，落在 300–400 MB（仍只有 Java `-Xmx` 的 1/7），再按 M4 推进 |

### 备选：分层预算

如果“内存不随 dump 增长”可以放宽为“有上限地增长”，可先实现 512 MB 档：objdir 留在内存（32 B/对象，5.9M = 189 MB），只把数组 body、边、parent 落盘。代码量减少约 40%，但 1 亿对象的 dump 会回到 3 GB+，不满足常驻目标。

---

## 附录 A：对象目录空间估算

| dump 对象数 | objdir（32 B/对象） | parent 日志（12 B/节点） | 外排临时 |
|---|---|---|---|
| 5.9 M（appauditd） | 189 MB | 71 MB | ~55 MB |
| 20 M | 640 MB | 240 MB | ~180 MB |
| 100 M | 3.2 GB | 1.2 GB | ~900 MB |

磁盘需求按对象数线性增长，内存不增长；临时目录建议独立挂载并设置配额。

## 附录 B：参考数据

- 当前回归：`task_dump.hprof` 17/17 PASS，tolerance 全部 0；
- appauditd 对账：总对象 5,916,809、堆 390,966 KB、live/garbage 4,206,283/1,710,526，与 Java 完全一致；
- 内存实测产物：`/tmp/prof-run.log`、`/tmp/prof-rss.log`、`/tmp/peak-vmmap.txt`、`/tmp/peak-footprint.txt`。
