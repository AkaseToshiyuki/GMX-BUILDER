# GMXBUILDER 用户手册

<p><a href="USER_MANUAL.md">English</a> · <strong>简体中文</strong></p>

| 项目 | 内容 |
|---|---|
| 文档版本 | 1.0.0 |
| 适用软件 | GMXBUILDER 1.0.0 |
| 编写人 | Haochen Yang |
| 发布日期 | 2026-10-08 |
| 文档状态 | 正式发布 |

## 变更日志

| 版本 | 日期 | 内容 |
|---|---|---|
| 1.0.0 | 2026-10-08 | 正式版；更新资产 v7、五类工作流、异步 API、最终结构确认、安装要求与输出目录 |

本手册对应软件 1.0.0。脂质资产 v7 包含 37 个已验收组合、134,000 个初始化构象，
另附 67 份 GAFF2/AM1-BCC 参数缓存。参数缓存不等于构象验收，初始化可用也不等于
膜体系已达平衡。完整组合见[发布支持清单](release-support.zh-CN.md)。

## 1. 认识 GMXBUILDER

GMXBUILDER 为 GROMACS 准备膜蛋白、纯脂质双分子层和溶液相体系。网页、CLI
和 HTTP API 共用同一套模块与验证规则。

### 1.1 可用工作流

| 工作流 | 用途 |
|---|---|
| Bilayer Builder | 处理并定向膜蛋白，构建膜，加水和离子 |
| Pure Bilayer System | 构建无蛋白双分子层；可选择干膜或湿体系 |
| Solvator | 构建无膜的蛋白、标准线性 DNA/RNA、蛋白–核酸或含配体水溶液体系 |
| Martini 3 Bilayer Builder | 按明确叶片数量和组分构建平面膜或蛋白–膜粗粒化体系 |
| Martini 3 Solvent Builder | 构建标准蛋白水相粗粒化体系 |

页面显示但灰色禁用的工作流不属于当前可用能力。以主页和
`GET /api/task-types` 的运行时结果为准。

### 1.2 Check、Viewer 和 Build

每次 **Check** 都会保存一个 Task 独有的坐标检查点。回退并重新 Check 上游步骤
后，下游旧检查点会失效，必须重新确认。

原子级湿体系最终 Build 读取 Ion Check 的坐标；干纯膜读取 Membrane Check 的坐标。
Martini 3 读取用户已经在 Final Structure Review Viewer 中确认的 `cg_system` 检查点。
Build 不会重新运行结构处理、定向、铺膜、加水或离子放置，只完成拓扑、索引、
MDP、运行脚本和 ZIP 打包。因此 Check 后 Viewer 和下载包共享同一坐标来源。

### 1.3 科学边界

- 力场必须属于明确兼容的同一参数家族，不能因为文件能 include 就任意混用。
- 自动质子化、净电荷建议、蛋白朝向和默认 MDP 都需要结合研究体系复核。
- “构建成功”表示体系可进入最小化和平衡，不表示已经完成生产级平衡或采样。
- 不支持或不完整的脂质、修饰、配体参数会明确报错或禁用，不会静默忽略。

以下两条边界会在构建过程中主动报告，而不是留待事后发现。

**游离蛋白端基。** 未封端的端基按标准带电模板 NH3+ / COO− 构建，未实现中性
端基微观状态。边界由模型 pKa 推导而来，而非人为选定：pH 4.45–7.05 之间两种
标准态占比均不低于 90%，构建静默通过；pH 3.50–8.00 之间标准态仍是多数物种，
构建继续并报告实际占比——因为分子动力学必须为每个可滴定基团指定一个离散状态，
惯例是指定优势态；超出 pKa 之后标准带电态成为少数物种，构建中止。

模型 pKa 是构建策略的近似输入，不是特定蛋白端基的实验测量。ACE/FOR/NME 会
改变分子化学身份，只应在实验构建体或建模目标需要相应封端时选择。封端不是绕过
未支持 pH 或中性游离端基状态的通用办法。

**水模型。** 力场与水模型的组合在使用前会被分级。仅仅"文件存在"而非力场默认、
也未被回归测试覆盖的组合，会被判定为 expert-unvalidated，需显式设置
`allow_unvalidated_water_model: true` 才能继续。分级结果及其策略版本会写入导出
的体系元数据。

### 1.4 支持的输入边界

以下是当前构建实现的边界，不是对科学有效性的判断。超出时会报错并指出具体数值。

| 输入项 | 边界 |
|---|---|
| AA 每叶起始脂质数；Martini 每叶精确数 | 64 至 5000 |
| 双层或溶剂化盒子单轴 | 100 nm |
| 溶剂化盒子体积 | 50000 nm³ |
| 侧链 pKa 估计 | pH 1.0 至 13.0 |
| GAFF2 或 CGenFF 配体原子数 | 2048 |
| 上传结构 | 32 MB，250000 原子（见下） |

上传限制可通过环境变量调整，超出硬上限时会**钳到上限**而不是静默回落到默认值：
请求高于上限则取上限，低于下限则取下限，两种情况都会写入服务日志。

## 2. 安装与启动

### 2.1 引导环境要求

- Linux x86-64，Python 3.10 或更高版本；
- Git、CMake、C++17 编译器及 Python `venv` 模块；
- 首次安装时能够访问网络；
- 仅当自动构建的 GROMACS 需要 CUDA 加速时，才需要包含 `nvcc` 的 NVIDIA CUDA
  Toolkit。

安装脚本会管理必需的 GROMACS 运行时、Python 环境、GAFF2/AM1-BCC 工具、力场
数据和已验证的预构建脂质资产。因此用户不需要另行安装 GROMACS、AmberTools、
ACPYPE、Open Babel、Martini 3 数据或 Python 包。完整自动安装的 GAFF 工件锁
目前仅验证 Linux x86-64；其他平台需要单独验证的科学运行时。受管理 Web 服务还
需要 systemd 用户管理器、cgroup v2、Landlock ABI 3+、FUSE3、`fusermount3`、
`/dev/fuse`，以及构建所需的 `libfuse3-dev` 和 `pkg-config`。系统前置条件由管理员
安装后，应用及科学运行时可安装在用户目录。详见[资源配置](anonymous-resources.md)。

### 2.2 一键安装本地服务

在仓库根目录运行：

```bash
./install-local.sh
```

不带参数时脚本全程无人值守：缺省监听回环地址的 7788 端口，分配检测到的 CPU
核心数的一半，推导可整除的并发队列数并启动用户级服务。脚本优先复用用户明确
指定或 PATH 中可见的 GROMACS 2026.0 及以上版本；否则下载官方 GROMACS 2026.3
源码归档，校验固定的 SHA-256 后，在用户的 GMXBUILDER 数据目录内构建私有运行时。
检测到 `nvcc` 时启用 CUDA，否则构建功能完整的 CPU 版本。

脚本同时建立私有 GAFF2/AM1-BCC 环境，从 conda-forge 安装固定版本的 AmberTools、
ACPYPE 和 Open Babel；随后下载 `scripts/external_assets.json` 列出的单独分发力场及
清单固定的 v7 脂质归档（schema 4）并校验摘要，再建立锁定的 Python 环境并写入用户缓存。
安装不要求 Git LFS、GitHub Token 或 root 权限。运行 `./install-local.sh --help`
可查看命令行覆盖项，运行 `./install-local.sh --interactive` 才会逐项询问。任何无
登录保护的非回环地址（含 `0.0.0.0`/`::` 通配地址）都必须通过
`--allow-unsafe-deployment` 显式选择。

如需复用现有兼容版本，可显式指定：

```bash
./install-local.sh --gmx-bin /opt/gromacs/bin/gmx
```

即使本机存在 CUDA，也可强制构建 CPU 版本：

```bash
GMXBUILDER_GROMACS_FORCE_CPU=1 ./install-local.sh
```

### 2.3 手动安装

```bash
git clone https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER
python3 scripts/install_gromacs.py
python3 scripts/install_gaff_runtime.py
python3 scripts/install_external_assets.py
python3 scripts/fetch_prebuilt_assets.py
uv sync --frozen --no-dev
source .venv/bin/activate

gmxbuilder --version
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
```

手动安装后，在当前 shell 中导出受管理运行时路径：

```bash
export GMX_BIN="$HOME/.local/share/gmxbuilder/runtime/gromacs-2026.3/bin/gmx"
export GMXBUILDER_GAFF_ENV="$HOME/.local/share/gmxbuilder/gaff-env"
```

下载引导与 `prebuilt-assets install` 会依次校验归档 SHA-256、严格库 schema，并确认归档内
每个构象库条目可由当前运行时加载，然后才写入缓存。发布包只包含生成时已经
通过生产质量门槛的力场/脂质组合；其他兼容组合会明确保持不可用。安装不会覆盖
已存在的较新条目。公开仓库和公开 Release 资产不要求 GitHub Token。

### 2.4 启动服务与资源限制

```bash
gmxbuilder serve --host 127.0.0.1 --port 7788
```

浏览器访问 `http://127.0.0.1:7788/`。按部署配额限制资源：

```bash
gmxbuilder serve \
  --cpu-cores 16 \
  --task-threads 4 \
  --max-builds 4 \
  --gpu-count 1
```

`--task-threads` 必须整除 `--cpu-cores`。它只限制当前步骤内部安全的数值内核
或外部工具，同一 Task 的步骤不会并行。使用 `CUDA_VISIBLE_DEVICES` 规定物理
GPU 的可见范围和顺序；`--gpu-count 0` 强制 CPU-only。

启动后检查：

```bash
curl -fsS http://127.0.0.1:7788/health
```

缺省监听地址和端口是 `127.0.0.1:7788`。在 `local` / `trusted-lan` 模式下，
没有全局认证的非回环监听必须显式启用 `--allow-unsafe-deployment`。公网部署
另有 `public`（全局认证）和 `public-anonymous`（受管理的匿名访问）两种模式；
两者均要求配置可信 TLS 代理和明确的 HTTPS 来源，匿名模式还要求受管理资源隔离。

IPv4/IPv6 通配地址同样适用这条规则。选择通配地址本身不构成许可——它是暴露面
最广、也最容易被误选的一种——未显式允许时服务将拒绝启动：

```bash
gmxbuilder serve --host 0.0.0.0 --port 7788 --allow-unsafe-deployment
gmxbuilder serve --host 192.0.2.10 --allow-unsafe-deployment
GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=1 \
  gmxbuilder serve --host 192.0.2.10
```

安装器执行同样的要求，并把选择写入生成的本地服务启动脚本：

```bash
./install-local.sh --bind-host 0.0.0.0 --allow-unsafe-deployment
./install-local.sh --bind-host 192.0.2.10 \
  --deployment-mode local --allow-unsafe-deployment
```

`local` 和 `trusted-lan` 模式均没有终端用户登录保护，只能放在受防火墙保护的
私有网络，绝不能直接暴露到互联网。该允许开关不会放宽 `public` 模式；公网模式
仍必须按附录 A 配置全局认证、TLS 反向代理、HTTPS 来源和可信代理地址。
`public-anonymous` 是单独配置的匿名模式，不能用 unsafe 开关替代其资源隔离要求。

## 3. Web 网页端

### 3.1 Task ID 和网址

选择 Bilayer Builder 后，地址形如 `/BilayerBuilder/Step1`，后续步骤使用
`StepN`。地址只包含工作流和步骤，不包含 Task ID。同一标签页刷新时，客户端会
恢复其保存的任务及服务端检查点；没有已保存任务的步骤网址会回到该工作流首步。
未提交表单不会成为检查点。Pure Bilayer System、Solvator 和 Martini 3 使用各自路由。

请立即保存 Task ID。它同时是访问该任务的 bearer capability，不应公开分享。
从主页输入 Task ID 后，程序会依据真实检查点恢复到下一个未完成步骤；已经完成
Build 的任务直接进入 Build/下载页面。页面标题栏的 Task ID 右侧有 Copy 按钮。

### 3.2 Bilayer Builder

#### Step 1 — Input Structure

上传页面支持当前版本声明的结构格式。Check 后查看：链、原子、残基、保留的
小分子、缺失原子、alternate location、非标准残基和 Viewer。

已知修饰残基会先记录修饰身份，再转换为标准母体；Step 3 会提出目标力场支持
的对应修饰。无法识别、化学身份有歧义或目标力场不可用的项以 warning 显示，
需要用户决定是否修复源结构或删除该组分。

不完整的标准蛋白重原子会阻断拓扑生成。不要把缺失侧链当作可由力场自动分配
部分电荷的有效残基。

#### Step 2 — Force Field

选择蛋白力场、膜后端、保留小分子后端和水模型。可用选项由当前结构和安装的
参数共同过滤：

- Amber 蛋白优先使用覆盖该物种的 Lipid21；混合 Lipid21/GAFF2 仅在客体及采样宿主契约均通过时可用。整张膜 GAFF2 是显式替代方案；GAFF2 配体需要整数净电荷；
- CHARMM 蛋白与 CHARMM 脂质；配体默认使用本地 CHARMM-compatible 识别与完整模板，受限实验分配需显式启用，也可导入匹配的 CGenFF 文件；
- 不能把 Amber 蛋白/小分子与 CHARMM 膜混合。

水模型在本步骤锁定，后续 Solvent & Box 只显示和使用该模型。选择混合膜时，
Step 5 会对全部组成重新确认同一膜后端。

#### Step 3 — Structure Processing

页面包含 Protonation、Termini 和 Modifications。

Protonation 的 pH 改变后需重新点击 Calculate。PROPKA 给出结构相关 pKa 建议，
用户可逐位点复核 HIS tautomer 和其他可滴定残基。它不等同于 constant-pH MD，
金属、催化位点和埋藏氢键网络尤其需要人工判断。

Termini 只显示目标力场原子完整的选项。标准 NH3+/COO− 以及受支持的 ACE/NME
可以使用；灰色项不能通过同名近似代替。

Modifications 会自动显示 Step 1 识别的候选位点。逐一核对类型和编号后再 Check。
只有目标力场具有完整原子、键合参数、电荷和几何验证的修饰可选；复杂糖基化、
长链脂化和缺少原生模板的修饰保持不可用。

#### Step 4 — Protein Orientation

自动方法给出本地 PPM-like、疏水矩或跨膜区域近似结果。观察灰色膜边界平面与
蛋白的跨膜段、胞内外结构。如果自动角度不合理，切换 Manual Adjustment 修改
Z offset、tilt 和 rotation。

手动调整后的 Viewer 就是 Check 将保存的坐标。灰色球平面表示近似水相/疏水区
边界，用于判断蛋白暴露，不等同于弛豫膜中特定原子层的精确 DHH。

#### Step 5 — Membrane Builder

设置上、下叶组成、比例和起始数量（64–5000）。Check 根据两叶各自的目标 APL、
膜片层内的蛋白占据面积，以及蛋白 XY padding，计算共同的 XY 盒面积，再分别
填充两叶可用面积。因此即使组成相同，实际上下叶数量也可以不同，且可高于起始数。
若任一叶需要超过 5000 个脂质，则明确拒绝构建。

Check 前只显示估计；Check 后展示保存结构的实际上下叶总数及各物种数量，恢复
任务时也保留这些结果。编辑输入会使旧结果失效。目标面积只用于构建，不能证明
平衡态 APL 或膜稳定性。每种脂质仍须通过所选力场的支持和构象检查。

Check 后检查 Viewer 和质量报告：所有头部应朝溶剂、上下叶尾部相对、疏水核心
贴合、蛋白周围无严重冲突、XY 周期边界密封且没有大面积空隙。

申请新脂质时，点击脂质选择器中的 **Contact administrator**，从首页或公告板
获取管理员邮箱，在邮件中提供脂质名称、SMILES（含适用的立体化学及形式电荷）
和所需力场。网页不接收新分子或启动其参数化、预平衡，管理员审核并离线验证后
再安排提供。首页链接在新标签页打开，保留当前构建页面。

已完成的历史任务私有脂质仍可使用。未完成的历史计算不会自动恢复，也不能在
网页重试；请联系管理员处理，或使用已安装脂质新建任务。

#### Step 6 — Solvent & Box

`Z Padding` 表示从上下脂质分子外表面分别向水相延伸的距离，与蛋白外形无关。
蛋白如果伸出太远，程序会要求增大 padding，而不是让上下水层不对称。膜体系
X/Y 由 Membrane Check 锁定。

设置 padding 和 overlap scale 后点击 Check。Viewer 盒线应包围并居中显示完整
体系，统计表中的水数和盒尺寸来自已保存检查点。

#### Step 7 — Ions

设置盐浓度、中和选项和离子种类。固定种子的均匀随机替换是已验证且推荐的缺省
方法，离子使用被替换完整水分子的氧坐标。形式电荷排序和无量纲 Metropolis
位点优化也替换完整水分子，但属于明确标记的实验性启发式方法；它们不是平衡态
离子采样，不能解释为预测的离子氛围。离子不应在盒角或溶质局部形成非物理堆积。

点击 Check Ion Counts 后，进入独立的 Final Structure Review 步骤。

#### Step 8 — Final Structure Review

最终 Viewer 显示检查点中的蛋白/其他溶质、膜、水、离子和周期盒，组分数量
直接来自检查点。加载过程会显示准备、下载、解析和绘制阶段，完整画面绘制后
才开放 Confirm Simulation System。仅显示时省略氢原子，导出坐标保持不变。
可用组分开关检查空间分布，加载失败可点击 Retry viewer 重试。确认绑定到当前
检查点；上游重新 Check 后必须重新复核。Solvator、无水 Pure Bilayer 和两种
Martini 3 工作流也使用独立的最终复核步骤。

#### Step 9 — Simulation Parameters 与 Build

能量最小化、每个平衡阶段和每个生产阶段各自管理 MDP 设置。不存在会覆盖它们
的 Global 物理参数。可以取消某个平衡或生产卡片，但能量最小化始终保留。

重点复核：温度、压力耦合类型、`tau_t`/`tau_p`、约束、COM removal 和组、时间步、
总步数、轨迹/能量输出间隔及 cutoff。力场切换会恢复对应家族的非键缺省值；
CHARMM 的 force-switch/`DispCorr=no` 不可直接用于 Amber/GAFF。

Hardware 只决定生成的 `run_md.sh`。可选择 GPU、逻辑 GPU ID、CPU threads、MPI
ranks 和运行模式；这些字段不会改变 MDP 物理参数。

点击 Build 后可能立即开始或进入 FIFO 队列。保存 Task ID；队列预计开始时间是
调度提示，不是完成保证。Build 完成后下载 ZIP。

### 3.3 Pure Bilayer System

该工作流不上传蛋白，也没有 Structure Processing 和 Orientation。选择力场后，
直接设置每叶起始脂质数与上下叶组成，两叶按共同面积自动调整数量。Membrane Check 使用与 Bilayer Builder 相同的
朝向、贴合、冲突和周期密封质量门槛。

勾选水和离子后继续 Solvent & Box、Ions、Final Structure Review 和 Simulation Parameters。取消溶剂化
则从 Membrane Check 导出干膜；为防止误把干膜当作水相生产体系，干膜包不生成
MDP 和 `run_md.sh`。

### 3.4 Solvator

Solvator 执行 Input Structure、Force Field、Structure Processing、Solvent & Box、
Ions、Final Structure Review、Simulation Parameters 和 Build，不包含膜与 Orientation。box padding 从
溶质在六个方向的外形计算；压力耦合缺省为 isotropic。

标准线性 DNA/RNA 会显示为链而不是小分子。选择 CHARMM36m 或已安装的 Amber14SB + OL24；Structure
Processing 使用 GROMACS 原生数据库补氢、设置 5′/3′ 羟基末端、连接 O3′–P
聚合物键并核对链净电荷。纯 DNA/RNA、蛋白–核酸复合物和带有已提供 CGenFF
参数的非共价小分子均可进入水化和离子步骤。修饰核苷、共价配体、金属配位、
其他 Amber 核酸组合及膜内核酸当前会明确阻断，不能以普通小分子方式绕过。
该操作会将上传的核酸坐标替换为补全氢原子的 `pdb2gmx` 输出；继续流程前必须
在 Step 3 Viewer 中逐链复核。

### 3.5 Martini 3 Builders

这是两个独立粗粒化入口，不与 Amber/CHARMM 原子级参数混合：Martini 3
Bilayer Builder 与 Martini 3 Solvent Builder，流程内部不再选择体系类型。
Solvent 必须上传标准蛋白；Bilayer 可取消蛋白构建纯膜，并指定每叶精确脂质总数、
上下叶对称/非对称组成。蛋白映射可选 folded、tm_helix 或 disordered，并明确
决定 Elastic Network。膜蛋白映射后进入独立 Orientation：缺省 PPM-like
扫描会同时评估全蛋白主轴和连续疏水片段，检测到可信跨膜片段时要求其进入膜
疏水核心；手动模式用于已知实验构象。

Final CG System Check 会一次性组装实际导出的完整体系，并检查净电荷、盐浓度、
脂质朝向、头基间距、双侧水层和适用的跨膜范围。先在 Viewer 中检查蛋白的生物学
朝向，再进入 Final Structure Review 点击确认；确认不会重建或移动坐标。Simulation Parameters 使用 Martini
专用 Reaction-Field、20 fs 生产步长和按环境选择的压力耦合。

首版不支持配体、PTM、糖链、核酸、自定义 CG 分子、曲面膜和 backmapping；这些
对象会明确阻断而不是被删除。精确边界和脂质列表可查询
`GET /api/coarse-grained/capabilities`。

周期盒由定位后的 CG 包络、溶剂 padding，以及膜体系的保守初始装箱面积与明确
脂质数量自动推导，不要求用户输入盒长。装箱面积不是平衡 APL，周期面积由后续
NPT 松弛。当前随附且构型后端可建立的官方 PC/PE/PG/PS/SM/甾醇条目会显示在
选择器中。Simulation Parameters 分别
管理最小化、可选 NVT、可选 NPT、生产、输出/COM 去除和执行硬件；各阶段保持严格
串行，并固定 Martini 3 兼容的非键相互作用缺省值。

## 4. CLI

### 4.1 命令发现

始终以当前安装版本的帮助为准：

```bash
gmxbuilder --help
gmxbuilder build --help
gmxbuilder serve --help
gmxbuilder lipid-library --help
gmxbuilder prebuilt-assets --help
gmxbuilder martini3-bilayer --help
gmxbuilder martini3-solvent --help
```

常用命令：

| 命令 | 用途 |
|---|---|
| `info` | 概述 PDB/CIF 输入结构 |
| `list-ff`、`list-water`、`list-lipids` | 查询当前安装能力 |
| `build` | 从 YAML 串行执行本地 Pipeline |
| `serve` | 启动 Web 与 API |
| `prebuilt-assets status/install` | 检查或安装随版本资产 |
| `lipid-library status/build/queue` | 查询或维护力场特异性脂质库 |
| `martini3-bilayer` | 串行构建 Martini 3 膜或蛋白–膜体系 |
| `martini3-solvent` | 串行构建 Martini 3 蛋白水相体系 |

Martini 3 混合膜示例：

```bash
gmxbuilder martini3-bilayer \
  --upper POPC:3 --upper CHOL:1 \
  --lower POPE:1 --lower POPG:1 \
  --lipids-per-leaflet 150 --padding 2 --salt 0.15 \
  --output ./martini-system
```

### 4.2 Solvator YAML 示例

```yaml
system_name: solution_system
output_dir: ./output
seed: 42

modules:
  input:
    pdb: ./protein.pdb

  forcefield:
    name: amber14sb
    lipid_ff: none
    ligand_ff: none
    water_model: tip3p

  structure:
    pH: 7.0
    prepare_standard_termini: true

  solvation:
    box_padding: 2.0

  ions:
    cation: NA
    anion: CL
    concentration: 0.15
    neutralize: true
    ion_method: random

  topology: {}

  simparams:
    schema_version: 2
  export:
    write_mdp: true
    execution_hardware:
      mode: thread-mpi
      cpu_threads: 8
      mpi_ranks: 2
      use_gpu: true
      gpu_count: 1
      gpu_ids: [0]
      gmx_command: gmx
```

执行：

```bash
gmxbuilder build --config build.yaml
gmxbuilder build --config build.yaml --output ./other-output
```

`--output` 覆盖 YAML 的 `output_dir`，不修改源文件。

YAML/CLI 构建中的执行硬件位于 `export.execution_hardware`，而不是
`simparams`。它只改变生成的运行脚本，不会改变已确认的分子坐标或 MDP 物理设置。
HTTP Build 请求使用已安装 OpenAPI schema 中单独的 `modules.execution` 对象。

### 4.3 Bilayer 差异

在 `structure` 后加入：

```yaml
  orient:
    method: ppm

  membrane:
    lipid_composition:
      upper:
        - {name: POPC, ratio: 70}
        - {name: POPE, ratio: 30}
      lower:
        - {name: POPC, ratio: 70}
        - {name: POPS, ratio: 30}
    box_padding: 2.0
```

不要使用旧的 `composition_upper`/`composition_lower` 字段。全部组成必须能由
同一膜后端覆盖。

### 4.4 Simulation Parameters 契约

`simparams` 顶层只接受当前 schema 声明的 `minimization`、`eq_stages`、
`prod_iters` 和 `schema_version`。MDP 物理参数写在对应阶段中，
不能使用旧的 global override。

显式 `dt` 必须同时写 `dt_unit: fs` 或 `dt_unit: ps`。执行硬件只配置运行脚本，
不能再放入 `simparams`。省略阶段配置时，程序按体系和力场生成缺省协议；用户仍需
在正式模拟前复核。

## 5. HTTP API

### 5.1 接口发现与错误处理

服务启动后访问：

- Swagger UI：`http://127.0.0.1:7788/docs`
- OpenAPI：`http://127.0.0.1:7788/openapi.json`
- 健康检查：`GET /health`
- 匿名最小存活检查：`GET /health/live`

先查询 `/api/task-types`、`/api/options` 和 `/api/hardware`，不要把选项固定在
客户端。非 2xx 响应都必须视为失败：`400` 为输入无效，`404` 为任务/结果不存在，
`409` 为步骤/检查点/兼容性冲突，`503` 为队列暂时不能接收。

### 5.1.1 受管理的异步操作

受管理服务的上传、Check、Viewer 准备及 Build 等耗时请求可能先返回 **202**，
响应头含 `X-GMXBUILDER-Operation`，JSON 中含 `operation_id`。202 表示已接收，
不表示该步骤成功。轮询操作状态，直到 `response_ready` 为 true，再读取原请求
的结果并检查其 HTTP 状态；成功后才能进行下一步。浏览器会自动完成这一过程。

```bash
API=http://127.0.0.1:7788
curl -fsS "$API/api/operations/<operation-id>"
curl -sS "$API/api/operations/<operation-id>/result"
```

下列示例中的每个请求都必须按上述方式等到最终结果。不要在尚未完成的上传响应中
读取 Task ID，不要并行提交同一任务的后续步骤。410 表示已过期或已清理，507 表示
存储不足。受管理安装的任务缺省从创建起保留 24 小时，等待、刷新或下载不会延长
期限；应及时保存下载结果。`GET /api/resource-policy` 报告实际配置及隔离是否生效。

力场兼容性查询返回 `503` 和 `code: missing_compatibility_data` 时，所需力场或检查点数据不完整。
管理员应完成安装，用户随后重新执行输入 Check；该响应不会放行构建。

### 5.2 Solvator 最小调用顺序

```bash
API=http://127.0.0.1:7788

curl -sS -X POST "$API/api/upload-pdb" \
  -F "file=@protein.pdb" \
  -F "task_type=solvator"

TASK_ID=<返回的任务ID>

curl -sS -X POST "$API/api/step/$TASK_ID/input" \
  -H "Content-Type: application/json" -d '{"config":{}}'

curl -sS -X POST "$API/api/step/$TASK_ID/forcefield" \
  -H "Content-Type: application/json" \
  -d '{"config":{"name":"amber14sb","lipid_ff":"none","ligand_ff":"none","water_model":"tip3p"}}'

curl -sS -X POST "$API/api/step/$TASK_ID/structure" \
  -H "Content-Type: application/json" \
  -d '{"config":{"pH":7.0,"prepare_standard_termini":true}}'

curl -sS -X POST "$API/api/step/$TASK_ID/solvation" \
  -H "Content-Type: application/json" -d '{"config":{"box_padding":2.0}}'

curl -sS -X POST "$API/api/step/$TASK_ID/ions" \
  -H "Content-Type: application/json" \
  -d '{"config":{"cation":"NA","anion":"CL","concentration":0.15,"neutralize":true,"ion_method":"random"}}'
```

Bilayer 任务使用 `task_type=membrane-bilayer`，并在 Structure 后依次调用
`orient` 和 `membrane`。Pure Bilayer 通过 `POST /api/tasks` 创建无上传任务。
精确请求 schema 以 OpenAPI 为准。

#### 最终结构确认

离子步骤成功后，获取并检查完整最终体系。JSON 包含用于显示的坐标、组分、盒、
`source_step` 和 `revision`；也可在 Web 的 Final Structure Review 中复核。
确认的是实际已查看的当前 revision，不能用旧 revision 或未经查看的结果代替。

```bash
curl -fsS --compressed "$API/api/step/$TASK_ID/ions/viewer.json" -o final-system.json
# 检查 final-system.json 对应的最终结构后再提交确认。
curl -sS -X POST "$API/api/task/$TASK_ID/final-review" \
  -H "Content-Type: application/json" \
  -d '{"source_step":"ions","revision":"<reviewed-revision>"}'
```

干纯膜使用 `membrane`，Martini 3 使用 `cg_system`，湿原子级体系使用 `ions`。
上游检查点改变后必须重新查看和确认；未经当前检查点确认，Build 会拒绝执行。

### 5.3 Build、队列和下载

```bash
curl -sS -X POST "$API/api/build" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"'"$TASK_ID"'",
    "task_type":"solvator",
    "system_name":"solution_system",
    "modules":{
      "simparams":{"schema_version":2},
      "export":{"write_mdp":true}
    }
  }'

curl -sS "$API/api/build/$TASK_ID/queue-status"
curl -sS "$API/api/build/$TASK_ID/log?since=0"
curl -fL "$API/api/task/$TASK_ID/download" -o result.zip
```

通过 `GET /api/steps/{task_id}` 查询检查点，通过
`GET /api/task/{task_id}/resume` 获取恢复 URL。`GET /api/tasks` 是管理接口，
需要 `X-Admin-Token`；Task ID 本身不是管理员令牌。

### 5.4 Martini 3 最小调用顺序

先查询 `GET /api/coarse-grained/capabilities`，然后创建任务：

```bash
curl -sS -X POST "$API/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{"task_type":"martini3-bilayer"}'
```

膜体系按 `input → cg_model → cg_mapping → cg_orientation → cg_environment →
cg_solvation → cg_system` 依次调用 `/api/step/{task_id}/{step}`；
`cg_orientation` 缺省使用本地 PPM-like 转移自由能扫描，并要求用户根据灰色疏水
核心边界复核构象。盒尺寸由定位后的蛋白外包络、精确脂质数和保守初始装箱面积
自动确定；该面积不是平衡 APL，后续 NPT 会松弛周期面积。`input` 使用
`{"include_protein":false,"environment":"bilayer"}`。蛋白体系先通过
`/api/upload-pdb` 将文件附加到同一 Task。Final Viewer 检查后调用
`GET /api/step/{task_id}/cg_system/viewer.json` 获取并查看当前结构，将返回的
`source_step` 和 `revision` 提交给 `POST /api/task/{task_id}/final-review`，再提交
`/api/build`。所有布尔值
必须使用 JSON `true` 或 `false`。

## 6. 输出包与运行

### 6.1 输出目录

```text
README.txt
manifest.json
CITATIONS.json
run_md.sh                 # 湿体系且启用了相应运行阶段
structure/
  input.gro
  input.pdb               # 超过 PDB 格式容量时省略
  index.ndx
topology/
  topol.top
  forcefield/             # 原子级力场数据库
  <分子参数>.itp
mdp/
  mini.mdp
  equili_<n>.mdp
  production.mdp 或 production_<n>.mdp
```

`topology/topol.top` 是参数 include 清单。Martini 3 的固定参数与蛋白 ITP 放在
`topology/toppar/`。所有工作流均提供 `manifest.json` 和 `CITATIONS.json`；具体
组分、索引组与文件名由体系决定。干 Pure Bilayer 不含水、离子、MDP 或运行脚本。

清单记录版本、输入哈希、构建设置、输出文件及校验值。原始上传文件不随包分发；
需要重建时应保留原始输入、配置与匹配的参数来源。先核对清单，再运行模拟。
安装包目录与下载项见[1.0.0 发布文件](RELEASE.zh-CN.md)。

### 6.2 一键运行

```bash
unzip result.zip -d simulation
cd simulation
chmod +x run_md.sh
./run_md.sh
```

脚本依次执行最小化、用户启用的平衡阶段和生产分段，不并行执行阶段。按自己的
集群环境修改脚本的 GROMACS 命令、MPI launcher、CPU 和 GPU 映射，但不要因此
改变 MDP 科学参数。

运行前先阅读包内 `README.txt`，检查 `topology/topol.top`、`structure/index.ndx`、总电荷、盒尺寸、
膜/溶剂/离子分布及每阶段 MDP。

`README.txt` 同时记录构建种子与速度种子。速度种子由构建种子推导，并以显式正整数
`gen-seed` 写入 MDP，而不是运行时随机的 `gen-seed = -1`，因此一个包的初始速度是
可复现的：只改种子的两次构建会得到不同的初始速度，用同一种子重跑则得到相同结果。

每个生成的 MDP 文件都会在**应用完全部高级覆盖之后**再做校验，也就是说被检查的是
最终交付的文件，而不是浏览器表单。校验涵盖积分器、时间步与当前约束方案的匹配、
截断半径的排序关系、CHARMM force-switch 要求，以及溶质到周期面的净距离。协议被
拒绝时，构建在写出任何 MDP 之前就中止，因此不会留下"改了一半"的参数集。

## 7. 常见问题

### 7.1 脂质显示不可用

这表示当前力场/膜后端没有通过该脂质的参数和严格构象门槛。选择界面提示的
替代力场，或更改脂质组成；不要通过刷新或旧检查点绕过后端验证。

### 7.2 膜脂方向或膜核心异常

重新查看 Membrane Check 质量报告。头部应朝水相，尾部相对，上下叶贴合且周期
边界无大空隙。若报告与 Viewer 不一致，请保留 Task ID 并停止后续构建。

### 7.3 蛋白自动朝向不合理

自动朝向是近似算法。使用 Manual Adjustment，结合已知跨膜段、胞内外结构和
实验信息调整，然后 Check 保存当前预览。

### 7.4 上下水层看起来不同

Z padding 从脂质外表面计算。先检查 Viewer 透视和膜是否居中，再看保存的盒尺寸
和界面到盒边距离；伸出的蛋白不应成为 padding 原点。

### 7.5 Ion Viewer 不显示水或离子聚集

确认查看的是 Final Structure Review 中的最终体系，而非旧预览。统计水
数应非零，离子应位于被替换水位点。保留 Task ID 和截图，不要继续 Build。

### 7.6 ZIP 缺少 MDP 或运行脚本

干纯膜会有意省略。湿体系检查 Simulation Parameters 中是否启用阶段和
`write_mdp`；若仍缺少，请检查 Build 日志及包内 `README.txt`。

### 7.7 Build 排队或耗时

Build 不重建坐标，但拓扑一致性、索引/MDP/脚本生成和压缩仍需时间。服务器满载
时会先排队。保存 Task ID，通过队列状态或恢复 URL 返回任务。

### 7.8 Task 不存在

Task 可能已过保留期或 ID 输入错误。任务私有自定义脂质也随任务清理，不能转移
到另一 Task。

### 7.9 pH 被拒绝，或构建对端基发出警告

见 1.3 节。pH 3.50–8.00 之间构建继续并给出实际占比；超出该范围构建中止。
根据实际实验构建体与研究目标选择游离端基或受支持的封端。截短本身不自动授权
为每条链封端；不要通过改变分子身份来规避未支持的端基状态。

### 7.10 水模型被判定为 expert-unvalidated

该组合的拓扑文件存在，但既不是力场默认，也未被回归测试覆盖。要么改用力场默认
水模型，要么在确认该组合适用于你的体系后设置
`allow_unvalidated_water_model: true`。无论选哪种，判定结果都会写入导出的元数据。

### 7.11 报告了尺寸限制

1.4 节列出了各项边界，报错信息会指出超限的具体数值。注意通过环境变量抬高的上传
限制会被**钳到硬上限**，并在服务日志中记录，所以实际生效值未必等于请求值。

### 7.12 自动安装无法构建 GROMACS

确认已安装 CMake、C++17 编译器和 Python `venv`，并有足够磁盘空间及网络访问。
如本机 CUDA 工具链导致失败，可用 `GMXBUILDER_GROMACS_FORCE_CPU=1` 重试。若使用
管理员提供的兼容版本，通过 `--gmx-bin` 指定其 `gmx`。不要指定早于 2026.0 的
GROMACS，因为随附的 Amber ff14SB 移植需要较新的预处理行为。

## 8. 获取帮助与准确能力

```bash
gmxbuilder --help
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder lipid-library status

curl -fsS http://127.0.0.1:7788/health
curl -fsS http://127.0.0.1:7788/api/options
curl -fsS http://127.0.0.1:7788/api/task-types
```

力场、脂质、小分子和修饰的边界见
[科学兼容性](SCIENTIFIC_COMPATIBILITY.zh-CN.md)。提交问题时请提供版本、
步骤、完整错误信息和不含敏感结构内容的截图/日志片段。Task ID 仅应私下提供给
可信管理员，不应贴到公开问题或截图中。

## 附录 A：部署配置

- 使用非 root 服务账户。默认安装只监听 `127.0.0.1:7788`。
- `local` / `trusted-lan` 的无认证非回环监听需要显式 unsafe opt-in，只能用于受保护的私有网络。
- 公网认证模式为 `GMXBUILDER_DEPLOYMENT_MODE=public`，配置强 Basic/Bearer 凭据。
- 公网匿名模式为 `GMXBUILDER_DEPLOYMENT_MODE=public-anonymous`，要求受管理存储与计算隔离；保持 `GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=0`。
- 两种公网模式都配置明确的 HTTPS `GMXBUILDER_CORS_ORIGINS` 和可信代理地址 `GMXBUILDER_TRUSTED_PROXIES`；代理必须覆盖外来转发头。设置空 `FORWARDED_ALLOW_IPS=`，保留真实连接对端供应用校验。
- 配置独立的高强度 `GMXBUILDER_ADMIN_TOKEN`。Task ID、操作 ID 和令牌均不应写入公开日志。
- `GET /health/ready` 用于就绪检查；首页可访问或 `/health/live` 成功不代表配额存储可写。
- 受管理安装的默认上限为每操作 16 GiB 内存、每任务 2 GiB 存储、总存储 100 GiB、创建起 24 小时有效期。实际值见 `/api/resource-policy`；普通开发服务器不宣称具备生产隔离。

安装、迁移与维护步骤见[匿名资源配置](anonymous-resources.md)和
[Web 就绪与恢复](WEB_RELIABILITY.md)。这些服务操作不会授权或启动离线脂质模拟队列。
