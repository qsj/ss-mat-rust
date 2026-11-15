# AGENTS.md — SS-Mat Rust

Rust 重写版 SS-Mat（基于 JXRay 2.9 的 JVM 堆转储分析工具）。

## 构建与运行

```bash
cd /Volumes/KODAK/qsj/testproject/ss-mat-rust
cargo build --release
cargo run --release -- <heap_dump.hprof> [report.html]
```

测试：`cargo test --workspace`（34 个测试：15 traversal + 9 record_reader + 6 support + 3 parser + 1 objdir）。`SS_MAT_BIG_DUMP=<dump>` 时 ss-hprof-parser 会额外走一遍真实 dump 校验记录走查器。

回归检查：`python3 scripts/parity.py <report.html>`，基线 `scripts/parity_baseline.json` **全部 tolerance=0（严格模式）**；当前 17/17 精确 PASS。`Cargo.lock` 已不再被 `.gitignore` 忽略。

取证开关（默认关闭）：`SS_MAT_DEBUG_DIR=<dir>` 输出 roots/objects/S14 TSV，以及 Java 同 schema 的 `problem_stats.tsv` / `s8_summary.tsv` / `s8_groups.tsv` / `boxed_summary.tsv`（S8–S12 聚合），用于和 Java 逐对象/逐类 diff。

## 架构

**管线**: HPROF 文件 → 单次解析构建索引 → 重新解析提取字段 → 并行线性扫描 + BFS 遍历 → 问题检测统计 → HTML 报告

### Cargo Workspace 结构（8 个 crate）

| Crate | 用途 |
|---|---|
| `hprof-parser` | 基于 `jvm-hprof` 封装，单次遍历构建 string_table / class_registry / object_index / roots |
| `heap-model` | `HeapDump`、`JavaClassInfo`、`HeapValue`、`ClassCategory`、root 类型 |
| `support` | `LongIntMap`、`ChunkedQueue`、`ReferenceNode` trait 等（部分 stub） |
| `traversal` | `LinearScanCoordinator`（线性扫描）、`HeapObjectTraverser`（BFS 遍历） |
| `containers` | ~19 个集合遍历器描述符（HashMap/ArrayList/CHM/TreeMap 等，尚未接入报告） |
| `stats` | `StatsEngine`、`PerClassStats`、问题检测收集器（StringDup/DupArray/DirectBuffer/JvmInfo） |
| `report` | `MainReportGenerator` 生成 24 section HTML 报告 + Advisor 摘要 |
| `cli` | `clap` 参数解析 + 管线编排 |

### 关键依赖

`jvm-hprof` 0.1、`memmap2`、`hashbrown`、`rayon`、`clap` 4、`anyhow`、`parking_lot`

### 入口

`crates/cli/src/main.rs` — 四阶段管线:
1. Phase 1: `ss_hprof_parser::parse_hprof()` 解析 HPROF
2. Phase 2: `StatsEngine::new()` + 每类初始计数
3. Phase 3: 先提取 System.props（JVM flags 来源），再 `LinearScanCoordinator::run()`（重新解析 HPROF 访问字段数据）+ `HeapObjectTraverser::run_traversal()`（BFS）
4. Phase 4: `MainReportGenerator::generate()` 生成 HTML

## 当前完成状态

**全部 24 个报告 Section 已填充，无 "Not yet available" 占位符。报告 ~4.3MB。当前策略：零容差复刻 Java（包括 Java 的 bug），剩余差异见“待实现”。**

数据准确性（vs Java 参考实现，同一 heap dump）：
- heap 总字节：178,326KB（**精确匹配**）
- DirectByteBuffer：96,927KB 总量 / 94,786KB unreachable（**精确匹配**，address 去重 + fd/att 跳过）
- 可达性分类（live/garbage）：Rust **2,034,361 live / 731,150 garbage**，与 Java **精确一致**；字节 live 98,239KB / garbage 80,087KB（**精确匹配**）。已复刻 Set wrapper 的 backing map 只遍历 key（不再把 TreeSet.PRESENT 计为 live）
- 压缩 oop 自动推导（address-overlap 验证，不信任 heapOopSize）
- 报告体积对齐 Java（encoded-string 去重 `|*/c|*/s` + JS 解码器）
- Section 1 实例数/总对象数已修正为 2,135,318 / 2,765,511（此前重复计数）
- Section 3 已纳入数组类，byte[] 以 86,401KB 排名第一
- Section 11 六类坏原始数组：overhead **6,873KB 精确**、bad **36,006 精确**。两个 length-1 String.value byte[] 通过 `JAVA_BFS_VISITED_BEFORE_STRING` 复刻 Java BFS 先访问数组、后处理 String 的顺序
- Section 8 重复字符串：已复刻 Java `StringContentDeduplicator`（32 位 hashCharArray + pending/content map）；total 296,654、unique 154,204、group 43,441、waste 9,931,483B（9,698KB）**全部精确**。关键修复：JDK17 UTF-16 byte[] 按 Java 的小端序解码
- Section 12 装箱数字 overhead 与 Java 精确对齐（1,092KB）
- Section 18 DirectByteBuffer 对象数 65、off-heap 96,927KB / 94,786KB 精确对齐
- Section 10 坏对象数组与 Java **逐类逐 kind 精确对齐**：246,380 total / 107,388 bad / 5,078,276B（4,959KB）；复刻了 ClassDump resolveIndex 当 null 的 Java bug，排除多维基本类型数组与全部容器 backing array
- Section 9 坏集合：overhead **11,110KB 精确**；Total collections 178,615、bad collections **141,862 均精确**。已复刻 wrapper 语义、SMALL 公式、isCollection ×2、WeakHashMap ReferenceQueue+lock、ThreadLocalMap entry、LinkedList MANY_NULLS、LBQ/CSLM 空集合与 CSLM 的 head.node.next 计数（唯一残差：CSLM 那个 small 的 body 少 152B，不影响 KB 显示）
- Section 14 已移除按 top-by-size 样本外推的类级链面板，不再出现 `12.3 GB (7203.2%)`
- Section 14 重复原始数组与 Java `PrimitiveArraySizeAccumulator` 同构：382,570 total / 122,965 unique / 8 duplicate values / 151,360B（147KB）精确对齐；per-type 8 行（waste / 总实例 / distinct content）与 Java 完全一致。merge 只枚举 fastutil groupMap 的第一个 bucket（hash 0 chain），pending 大小 122,588 + 369
- SecHdr 35/35、Reference Chains 10/10、easy-to-fix 2/2

### 已实现的问题检测

| 检测项 | 方法 |
|---|---|
| 重复字符串 | String.value → char[]/byte[] 交叉引用，Java 8/9+ 编码兼容 |
| 装箱数字 | 8 种包装类，overhead = (alignedBodySize + referenceSize) - primitiveSize |
| 巨型对象 | >1MB 数组检测 |
| DirectByteBuffer | 类层级字段遍历读取 Buffer.capacity，计算 off-heap |
| 重复原始数组 | Java `PrimitiveArraySizeAccumulator` 同构：hashByteArray(32 位 wrapping) + pending/groupMap chain；merge 仅枚举 groupMap 第一个 bucket（hash 0 chain） |
| 重复对象数组 | 引用序列哈希去重 |
| Finalization | Finalizer 队列计数 |
| 集合统计 | 各集合类实例数/shallow size + 空/单例集合估算 |
| GC Root 分布 | 按 root 类型聚合 + 描述 |
| Bad Object Arrays | Java 五类（0-length/1-length/all-null/1-elem/sparse）；统计 live+garbage，排除容器 backing array |
| Bad Primitive Arrays | Java 六类：0-length / 1-length / all-zero / trail-0s / sparse / hi-bits-0s；按 Java 公式计 waste，排除 String backing 数组 |
| 重复对象检测 | 高频小实例类（>1000 实例, <=64 bytes） |
| System Properties | 对象图遍历 System.props → Properties → CHM/Hashtable → key/value String（~67 属性） |
| JVM Flags | `java.vm.inputarguments` / `--jvm-flags-file`；不再扫描字符串表猜测 |
| OOM 线程栈 + 局部变量 | 跳过 idle 线程（wait/park/sleep），最深 native 栈线程；局部变量展开为完整字段值（`<details><summary>` 可折叠） |
| 线程名提取 | Thread 实例 → name 字段 → String → char[]/byte[] 内容解析，提取 daemon/priority |
| 线程栈渲染 | Section 22 全部线程栈可折叠显示，OOM 线程优先 |

### BFS 遍历

- 完整的类层级字段引用提取（concrete class → superclass → ... → Object）
- ClassDump 记录作为 GC root 目标（SystemClass roots 指向 class object）
- 静态字段引用从 ClassDump 记录提取（跳过 Finalizer/Cleaner 等引用处理链）
- 实例字段只跳过 `Finalizer.prev`（Java 的 LinkedListLinks 语义）；弱引用对象延迟到第二趟，第二趟跟随全部字段（含 `Reference.referent`）
- 弱/终结器/清理器对象延迟遍历（先遍历强引用，再处理弱引用）
- ClassDump 对象（kind=3）从 live/garbage 计数中排除
- 结果: 2,765,511 对象, 2,034,362 live / 731,149 garbage；按字节 98,239KB live / 80,087KB garbage（与 Java 精确匹配）
- 数组 body size 含 4 字节 length 字段（Java `objectAlignBase = header + fields + 4`），garbage/retained 与 Java `getBodySize()` 一致

### Advisor 摘要

"MOST IMPORTANT ISSUES" 检测 5 类问题: 巨型对象、重复字符串、DirectByteBuffer off-heap、重复原始数组、重复对象数组
"Good news! Easy-to-fix" 面板: DirectByteBuffer off-heap 修复建议

### 报告结构

Section 1: 4 子表（对象类型统计、Live vs Garbage、线程/类计数、JVM 内部参数）
Section 2: OOM 线程栈 + 可折叠局部变量详情（完整字段值展开，匹配 Java 报告 `<t-c>/<t-f>/<t-s>` 格式）
Section 5: 可折叠按类 garbage 分解表（top 50）
Section 6: 可折叠按类 fixed overhead 分解表（top 50）
Sections 8/9/10/11/12: "What can I do to fix?" 可折叠修复建议面板
Section 22: 全部线程栈可折叠显示（OOM 线程优先）

## 待实现（零容差完全对齐）

- **严格回归**：`scripts/parity.py` 已全部 tolerance=0，当前 17/17 精确 PASS
- **S8 重复字符串**：已完全精确（296,654 / 154,204 / 43,441 / 9,931,483B）
- **S9 坏集合**：bad 141,862、overhead 11,110KB 均精确；只剩 CSLM 一个 small 实例的 body 少 152B（KB 显示无影响）
- **S11 坏原始数组**：已精确（36,006 / 6,873KB）。两个 length-1 String.value byte[] 是 Java BFS 队列先访问数组、后处理 String 的顺序产物，用 `JAVA_BFS_VISITED_BEFORE_STRING` 兼容表复刻
- **F1 live/garbage**：已精确 2,034,361 / 731,150；TreeSet wrapper 的 backing TreeMap 只遍历 key
- 静态根（JAVA_STATIC）保持 display-only：Java 的 `JavaStaticRoot` 实际指向 protection domain（源码/jar/report 三方确认），不是静态字段值；不要按静态字段播种
- Per-GC-root retained size / Section 3 真 retained（需要 dominator tree）
- 容器遍历器接入报告（空/稀疏集合开销、Duplicate Lists）
- Section 13 真正的重复对象检测（当前只是高频小实例候选列表）
- WeakHashMap value→key 硬引用检测（当前 `stats/src/weak_hashmap.rs` 是 stub）
- 超 1000 元素引用链（当前用 BFS 父树深度近似）
- Section 4 garbage 行语义与 Java 采样 retained 对齐

### 已复刻的 Java bug（需保留）

- S14：`PrimitiveArraySizeAccumulator.mergeAndBuildReport` 只枚举 groupMap 第一个 bucket；空数组 hash=0 且 fastutil key=0 优先返回，因此报告只显示 8 个零长度数组组 / 147KB。Rust 按同构行为输出。
- S10：`HeapDump.resolveIndex()` 对 ClassDump 地址返回 0，导致 `Class[]`/`Type[]` 元素全部被当作 null；Rust 用 `address_to_index` 复刻，S10 才得以逐类精确。
- S9：wrapper 迭代器把内层 map 标记 visited，导致 HashSet/LinkedHashSet/TreeSet/Properties 的内层 map 不计入 bad collections；Rust 已按 wrapper 记录复刻。
- JavaStaticRoot 的目标地址是 protection domain 而非静态字段值；Rust 不参与可达性。

## 已知限制

- Section 3 的 "Live shallow" 是 live shallow，不是 dominator retained；列名已改，避免误导
- `ss-containers` 与 `ss-support` 尚未被主流程引用；`traversal/stats` 仍有若干 stub 模块
- HPROF 仍会被完整重解析多次，GB 级 dump 需要后续性能优化
- 内存优化进行中：已完成度量、零算法瘦身、对象目录/外部排序/窗口化 mmap/记录走查器等基础设施（均带单测），BFS 边表已扁平化为 `EdgeTable`；未完成的部分（BFS 去图化、流式线性扫描、窗口化 mmap 接入、替换两张地址表）与实测数据见 `docs/memory-reduction-status.md`
- `JAVA_BFS_VISITED_BEFORE_STRING` 是 Java BFS live/garbage 访问顺序的兼容表，目前含 task_dump 的 2 个和 appauditd 的 1 个 `String.value` 数组；新 dump 若出现同类顺序差异，需要按 Java 实际访问顺序补充或改为完整 BFS 调度复刻

## 与 Java 版对比参考

Java 版源码在 `/Volumes/KODAK/qsj/testproject/ss-mat`（291 个 Java 文件），AGENTS.md 中有完整架构说明。
