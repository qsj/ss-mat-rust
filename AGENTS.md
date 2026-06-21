# AGENTS.md — SS-Mat Rust

Rust 重写版 SS-Mat（基于 JXRay 2.9 的 JVM 堆转储分析工具）。

## 构建与运行

```bash
cd /Users/qsj/sdqsj/testproject/ss-mat-rust
cargo build --release
cargo run --release -- <heap_dump.hprof> [report.html]
```

无测试。

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
2. Phase 2: `StatsEngine::new()` + JVM 信息提取
3. Phase 3: `LinearScanCoordinator::run()`（重新解析 HPROF 访问字段数据）+ `HeapObjectTraverser::run_traversal()`（BFS）
4. Phase 4: `MainReportGenerator::generate()` 生成 HTML

## 当前完成状态

**全部 24 个报告 Section 已填充，无 "Not yet available" 占位符。报告 ~4.3MB，与 Java 版数据对齐。**

数据准确性（vs Java 参考实现，同一 heap dump）：
- heap 总字节：178,326KB（**精确匹配**）
- DirectByteBuffer：96,927KB 总量 / 94,786KB unreachable（**精确匹配**，address 去重 + fd/att 跳过）
- 可达性分类（live/garbage）：差 <1%（不遍历 Reference.referent）
- 压缩 oop 自动推导（address-overlap 验证，不信任 heapOopSize）
- 报告体积对齐 Java（encoded-string 去重 `|*/c|*/s` + JS 解码器）
- SecHdr 35/35、Reference Chains 10/10、easy-to-fix 2/2

### 已实现的问题检测

| 检测项 | 方法 |
|---|---|
| 重复字符串 | String.value → char[]/byte[] 交叉引用，Java 8/9+ 编码兼容 |
| 装箱数字 | 8 种包装类，overhead = shallowSize - primitiveSize |
| 巨型对象 | >1MB 数组检测 |
| DirectByteBuffer | 类层级字段遍历读取 Buffer.capacity，计算 off-heap |
| 重复原始数组 | 内容哈希去重 + 实际 body 相等验证 |
| 重复对象数组 | 引用序列哈希去重 |
| Finalization | Finalizer 队列计数 |
| 集合统计 | 各集合类实例数/shallow size + 空/单例集合估算 |
| GC Root 分布 | 按 root 类型聚合 + 描述 |
| Bad Object Arrays | 空数组 + 单元素数组统计 |
| Bad Primitive Arrays | 空数组 + 尾部零元素数组统计 |
| 重复对象检测 | 高频小实例类（>1000 实例, <=64 bytes） |
| System Properties | 对象图遍历 System.props → Properties → CHM/Hashtable → key/value String（~67 属性） |
| JVM Flags | 字符串表 -X/-XX 标志提取 + GC 类型检测 |
| OOM 线程栈 + 局部变量 | 跳过 idle 线程（wait/park/sleep），最深 native 栈线程；局部变量展开为完整字段值（`<details><summary>` 可折叠） |
| 线程名提取 | Thread 实例 → name 字段 → String → char[]/byte[] 内容解析，提取 daemon/priority |
| 线程栈渲染 | Section 22 全部线程栈可折叠显示，OOM 线程优先 |

### BFS 遍历

- 完整的类层级字段引用提取（concrete class → superclass → ... → Object）
- ClassDump 记录作为 GC root 目标（SystemClass roots 指向 class object）
- 静态字段引用从 ClassDump 记录提取（跳过 Finalizer/Cleaner 等引用处理链）
- 弱引用 `referent` 字段跳过（不跟随弱引用链）
- 弱/终结器/清理器对象延迟遍历（先遍历强引用，再处理弱引用）
- ClassDump 对象（kind=3）从 live/garbage 计数中排除
- 结果: 2,768,888 对象, 2,581,208 live (6.8% garbage), 5,814,331 outgoing refs

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

## 待实现（与 Java 版完全对齐）

- Per-GC-root retained size（需要 dominator tree）
- 容器遍历器接入报告（空/稀疏集合开销计算）
- 引用链分析（超 1000 元素的链）
- 集合"坏"评分（backing array 相对 collection size 过大）
- WeakHashMap value→key 硬引用检测

## 与 Java 版对比参考

Java 版源码在 `/Volumes/KODAK/qsj/testproject/ss-mat`（291 个 Java 文件），AGENTS.md 中有完整架构说明。
