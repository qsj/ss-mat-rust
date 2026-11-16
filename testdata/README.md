# testdata — Java 对账夹具

这里放**与 Java 参考实现逐项对账**所需的输入和期望输出，便于在其他机器上复现
SS-Mat Rust 的对齐测试与内存测试。

## 目录

```text
testdata/
├── appauditd/
│   ├── appauditd-20261114.hprof        595 MB，未纳入 git（见下）
│   ├── appauditd-20261114.hprof.sha256
│   └── java/
│       ├── headline.tsv                Java 顶层指标（instances/bytes/live/garbage…）
│       ├── problem_stats.tsv           S9/S10/S11/S12 逐类问题聚合
│       ├── boxed_summary.tsv           S9/S12 汇总与逐 kind 行
│       ├── s8_summary.tsv              S8 重复字符串总计
│       ├── s8_groups.tsv               S8 重复字符串分组明细
│       ├── s15_dup_obj_arrays.tsv      S15（Java 两阶段）报告/阈值/候选类/组
│       ├── primitive_array_*.tsv       S14 原始数组累计器
│       └── objects_summary.tsv / roots_summary.tsv
└── task/
    └── java/                           同上（dump 使用仓库根目录的 task_dump.hprof）
```

`testdata/task/` 没有单独的 dump：直接使用仓库根目录的 `task_dump.hprof`
（267.8 MiB，同样被 `.gitignore` 忽略）。

## 关于 dump 文件

- `appauditd-20261114.hprof` 为 **595 MB**，超过 GitHub 单文件限制，因此**不提交**，
  只保留 sha256。拿到本仓库后，从原机器拷贝：

  ```bash
  rsync -av --progress <source>:/path/appauditd-20261114.hprof testdata/appauditd/
  # 或 scp
  shasum -a 256 testdata/appauditd/appauditd-20261114.hprof
  # 期望：ae281aab415d94c98c58d3ab35893354e5df48c13934e00df2cae780b17fe404
  ```

- `task_dump.hprof` 同理（267.8 MiB），放在仓库根目录。

## Java 夹具的来源

夹具由 Java 版 SS-Mat（JXRay 2.9 基线 + `DebugDumper`）在 **2026-11-15** 生成：

```text
SS_MAT_SKIP_REPORT=1 SS_MAT_DEBUG_DIR=<dir> java -jar target/mat-2.9.jar <dump>
```

- Java 报告生成在 appauditd 上会卡住，因此只 dump `StatsEngine` 对象，不生成 HTML；
  夹具是分析完成后的统计快照，与报告数据同源。
- `problem_stats.tsv` / `boxed_summary.tsv` / `s8_summary.tsv` / `s15_dup_obj_arrays.tsv`
  是 Rust 的验收 gate；行序可能不同，比较时按**多重集合**（multiset）处理。
- 没有拷贝 `objects.tsv`、`counted_calls.tsv`、`bad_collections.tsv` 等大文件：
  它们体积大，且 `bad_collections.tsv` 的逐实例行并非字节兼容（Java 会写重复地址行）。
  S9 的类级合计已包含在 `problem_stats.tsv` / `boxed_summary.tsv` 中。

## 复现步骤

```bash
cargo build --release
cargo test --workspace

# --- task_dump（已有 parity 基线，不需要 Java 也能判 C1） ---
./target/release/ss-mat task_dump.hprof /tmp/task-rust.html
python3 scripts/parity.py /tmp/task-rust.html --strict      # 期望 17/17

SS_MAT_DEBUG_DIR=/tmp/rust-task ./target/release/ss-mat task_dump.hprof /tmp/task-rust.html
python3 scripts/java_parity.py /tmp/rust-task testdata/task/java

# --- appauditd（与 Java 夹具逐项对账） ---
SS_MAT_MEM_TRACE=1 /usr/bin/time -l \
  ./target/release/ss-mat testdata/appauditd/appauditd-20261114.hprof /tmp/appauditd-rust.html

SS_MAT_DEBUG_DIR=/tmp/rust-appauditd \
  ./target/release/ss-mat testdata/appauditd/appauditd-20261114.hprof /tmp/appauditd-rust.html
python3 scripts/java_parity.py /tmp/rust-appauditd testdata/appauditd/java
```

`java_parity.py` 的 gate 文件：`problem_stats.tsv`、`boxed_summary.tsv`、
`s8_summary.tsv`、`s15_dup_obj_arrays.tsv`；全部以 multiset 比较，0 diff 才 PASS。
`--all` 会比较两个目录中所有同名文件（包含 `s8_groups.tsv` 等）。

> `SS_MAT_DEBUG_DIR` 会额外写 `objects.tsv` 等取证明细（appauditd 约 1.3 GB），
> 只做 gate 对账时会写盘但不需要保留；确保临时目录所在磁盘有足够空间。


## 期望结果（2026-11-15，macOS arm64，release）

| 项目 | task_dump | appauditd |
|---|---|---|
| `parity.py --strict` | 17/17 | —（基线只覆盖 task_dump） |
| `java_parity.py` gate 文件 | 0 diff | 0 diff |
| S15 | `report_present=0`，0/0/0/0（Java 为空报告） | 同左 |
| 总耗时 | ~18 s | ~62 s |
| 峰值 live heap | ~188 MB | ~401 MB |
| physical footprint | ~230 MiB | ~472 MiB |

## 注意

- macOS 的 `max RSS` 会包含 mmap 的 dump 文件页，明显高于 `physical footprint`；
  进程私有内存看 `[mem] peak: ... live_heap=` 和 `footprint`。
- `bad_collections.tsv` / `objects.tsv` 属于取证明细，不参与 gate；如需对比，先按地址
  聚合再比较。
- 修改 Java 夹具需要 Java 工程（不在本仓库）；日常回归只需要这里的 TSV。
