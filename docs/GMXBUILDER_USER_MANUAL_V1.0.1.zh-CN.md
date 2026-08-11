# GMXBUILDER 用户手册

<p><a href="GMXBUILDER_USER_MANUAL_V1.0.1.md">English</a> · <strong>简体中文</strong></p>

| 项目 | 内容 |
|---|---|
| 文档版本 | V1.0.1 |
| 适用软件 | GMXBUILDER v0.8.6 或更高版本 |
| 编写人 | Haochen Yang |
| 发布日期 | 2026-08-11 |
| 文档状态 | 正式发布 |

## 变更日志

| 文档版本 | 日期 | 变更内容 | 编写人 |
|---|---|---|---|
| V1.0.1 | 2026-08-11 | 更新无 Task ID 路由、Task ID 复制、Martini 3 工作流、CLI/API、输出结构及部署边界 | Haochen Yang |
| V1.0.0 | 2026-07-26 | 初次发布 | Haochen Yang |

## 1. 认识 GMXBUILDER

GMXBUILDER 为 GROMACS 准备膜蛋白、纯脂质双分子层和溶液相体系。网页、CLI
和 HTTP API 共用同一套模块与验证规则。

### 1.1 可用工作流

| 工作流 | 用途 |
|---|---|
| Bilayer Builder | 处理并定向膜蛋白，构建膜，加水和离子 |
| Pure Bilayer System | 构建无蛋白双分子层；可选择干膜或湿体系 |
| Solvator | 构建无膜的蛋白、标准线性 DNA/RNA、蛋白–核酸或含配体水溶液体系 |
| Martini 3 Builder | 独立构建标准蛋白水相、平面膜或蛋白–膜粗粒化体系 |

页面显示但灰色禁用的工作流不属于当前可用能力。以主页和
`GET /api/task-types` 的运行时结果为准。

### 1.2 Check、Viewer 和 Build

每次 **Check** 都会保存一个 Task 独有的坐标检查点。回退并重新 Check 上游步骤
后，下游旧检查点会失效，必须重新确认。

原子级湿体系最终 Build 读取 Ion Check 的坐标；干纯膜读取 Membrane Check 的坐标。
Martini 3 读取用户已经在 Final CG System Viewer 中确认的 `cg_system` 检查点。
Build 不会重新运行结构处理、定向、铺膜、加水或离子放置，只完成拓扑、索引、
MDP、运行脚本和 ZIP 打包。因此 Check 后 Viewer 和下载包共享同一坐标来源。

### 1.3 科学边界

- 力场必须属于明确兼容的同一参数家族，不能因为文件能 include 就任意混用。
- 自动质子化、净电荷建议、蛋白朝向和默认 MDP 都需要结合研究体系复核。
- “构建成功”表示体系可进入最小化和平衡，不表示已经完成生产级平衡或采样。
- 不支持或不完整的脂质、修饰、配体参数会明确报错或禁用，不会静默忽略。

## 2. 安装与启动

### 2.1 环境要求

- Linux，Python 3.10 或更高版本；
- Git LFS，用于获取随版本提供的脂质库资产；
- 可执行的 GROMACS；只有需要 GPU 时才要求 CUDA 版 GROMACS；
- 新 GAFF2 分子参数化需要单独安装 AmberTools/ACPYPE，项目不包含这些程序。
- 推荐使用 `uv` 按 `uv.lock` 中的版本和发行文件 SHA-256 安装依赖。

### 2.2 一键安装本地服务

在仓库根目录运行：

```bash
./install-local.sh
```

脚本交互式选择监听 IP、部署模式、端口、CPU 核心和并发任务数，建立虚拟环境、
安装依赖与预构建资产，并启动加固的用户级服务。默认仅监听
`127.0.0.1:7788`。选择非回环地址时必须明确使用 `trusted-lan`；该模式没有
最终用户登录，只适合受信私有网络。

### 2.3 手动安装

```bash
git clone https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER
git lfs install
git lfs pull

uv sync --frozen --no-dev
source .venv/bin/activate

gmxbuilder --version
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
```

`prebuilt-assets install` 校验归档后只安装缓存中缺失的文件，不覆盖已存在的
较新条目。公开仓库和公开 Release 资产不要求 GitHub Token。

如果没有 `uv`，一键安装会明确警告后使用 pip 非锁定安装；该方式适合开发，
不应作为可复现发布环境的证据。

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

如需在受信局域网监听所有地址：

```bash
GMXBUILDER_DEPLOYMENT_MODE=trusted-lan \
  gmxbuilder serve --host 0.0.0.0 --port 7788
```

缺省 `local` 模式会拒绝非回环监听。公网模式必须按附录 A 配置全局认证、TLS
反向代理、HTTPS 来源和可信代理地址。

## 3. Web 网页端

### 3.1 Task ID 和网址

选择 Bilayer Builder 后，地址形如 `/BilayerBuilder/Step1`，后续步骤使用
`StepN`。地址只包含模块和步骤，不包含 Task ID；刷新会安全返回首页。Pure
Bilayer System、Solvator 和 Coarse-Grained Builder 使用各自路由。

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

- Amber 蛋白与 Lipid21 或整张膜统一 GAFF2；GAFF2 配体需要整数净电荷；
- CHARMM 蛋白与 CHARMM 脂质；配体需要精确内置模板或匹配的 CGenFF 文件；
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

设置上、下叶组成、比例和盒尺寸/蛋白周围 padding。非对称膜可以使用不同组成；
每个叶片都必须形成完整比例。不可用脂质仍可显示，但会标明当前力场下的原因和
可用替代力场。

Check 后检查 Viewer 和质量报告：所有头部应朝溶剂、上下叶尾部相对、疏水核心
贴合、蛋白周围无严重冲突、XY 周期边界密封且没有大面积空隙。

Amber + GAFF2 下可以提交任务私有自定义脂质 SMILES。与内置库同一 canonical
identity 的分子会被拒绝。参数化和显式溶剂 NPT 预平衡期间页面阻塞；通过全部
门槛后才会出现在 Custom Lipids 中，只能由当前 Task 使用，任务清理时一并删除。

#### Step 6 — Solvent & Box

`Z Padding` 表示从上下脂质分子外表面分别向水相延伸的距离，与蛋白外形无关。
蛋白如果伸出太远，程序会要求增大 padding，而不是让上下水层不对称。膜体系
X/Y 由 Membrane Check 锁定。

设置 padding 和 overlap scale 后点击 Check。Viewer 盒线应包围并居中显示完整
体系，统计表中的水数和盒尺寸来自已保存检查点。

#### Step 7 — Ions

设置盐浓度、中和选项和离子种类。三种算法都替换完整水分子：均匀随机、周期
静电势和 Metropolis Monte Carlo。离子使用被替换水氧坐标，不应在盒角形成非物理
堆积。

点击 Check Ion Counts 后，在页面指定位置查看 Complete Simulation System。
Viewer 应同时显示蛋白/其他溶质、膜、水、离子和盒。确认组分、电中性和空间
分布后，点击 Viewer 下方的确认按钮才能继续。

#### Step 8 — Simulation Parameters 与 Build

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
直接设置每叶脂质数与上下叶组成。Membrane Check 使用与 Bilayer Builder 相同的
朝向、贴合、冲突和周期密封质量门槛。

勾选水和离子后继续 Solvent & Box、Ions 和 Simulation Parameters。取消溶剂化
则从 Membrane Check 导出干膜；为防止误把干膜当作水相生产体系，干膜包不生成
MDP 和 `run_md.sh`。

### 3.4 Solvator

Solvator 执行 Input Structure、Force Field、Structure Processing、Solvent & Box、
Ions、Simulation Parameters 和 Build，不包含膜与 Orientation。box padding 从
溶质在六个方向的外形计算；压力耦合缺省为 isotropic。

标准线性 DNA/RNA 会显示为链而不是小分子。当前必须选择 CHARMM36m；Structure
Processing 使用 GROMACS 原生数据库补氢、设置 5′/3′ 羟基末端、连接 O3′–P
聚合物键并核对链净电荷。纯 DNA/RNA、蛋白–核酸复合物和带有已提供 CGenFF
参数的非共价小分子均可进入水化和离子步骤。修饰核苷、共价配体、金属配位、
Amber 核酸参数及膜内核酸当前会明确阻断，不能以普通小分子方式绕过。

### 3.5 Martini 3 Builder

这是独立粗粒化流水线，不与 Amber/CHARMM 原子级参数混合。首先选择水相或
平面膜；水相必须上传标准蛋白，膜模式可取消蛋白以构建纯膜。蛋白映射时选择
folded、tm_helix 或 disordered，并明确决定 Elastic Network。膜组成按上下叶
分别输入 `NAME:ratio`。

Final CG System Check 会一次性组装实际导出的完整体系，并检查净电荷、盐浓度、
脂质朝向、头基间距、双侧水层和适用的跨膜范围。先在 Viewer 中检查蛋白的生物学
朝向，再勾选确认；确认不会重建或移动坐标。Simulation Parameters 使用 Martini
专用 Reaction-Field、20 fs 生产步长和按环境选择的压力耦合。

首版不支持配体、PTM、糖链、核酸、自定义 CG 分子、曲面膜和 backmapping；这些
对象会明确阻断而不是被删除。精确边界和脂质列表可查询
`GET /api/coarse-grained/capabilities`。

## 4. CLI

### 4.1 命令发现

始终以当前安装版本的帮助为准：

```bash
gmxbuilder --help
gmxbuilder build --help
gmxbuilder serve --help
gmxbuilder lipid-library --help
gmxbuilder prebuilt-assets --help
gmxbuilder coarse-grained --help
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
| `coarse-grained` | 串行构建 Martini 3 水相、膜或蛋白–膜体系 |

Martini 3 混合膜示例：

```bash
gmxbuilder coarse-grained \
  --mode bilayer \
  --upper POPC:3 --upper CHOL:1 \
  --lower POPE:1 --lower POPG:1 \
  --box-xy 12 --box-z 14 --salt 0.15 \
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
    hardware:
      mode: thread-mpi
      cpu_threads: 8
      mpi_ranks: 2
      use_gpu: true
      gpu_count: 1
      gpu_ids: [0]
      gmx_command: gmx

  export:
    write_mdp: true
```

执行：

```bash
gmxbuilder build --config build.yaml
gmxbuilder build --config build.yaml --output ./other-output
```

`--output` 覆盖 YAML 的 `output_dir`，不修改源文件。

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
`prod_iters`、`hardware` 和 `schema_version`。MDP 物理参数写在对应阶段中，
不能使用旧的 global override。

显式 `dt` 必须同时写 `dt_unit: fs` 或 `dt_unit: ps`。`hardware` 只配置运行脚本。
省略阶段配置时，程序按体系和力场生成缺省协议；用户仍需在正式模拟前复核。

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
  -d '{"task_type":"coarse-grained"}'
```

纯膜按 `input → cg_model → cg_mapping → cg_environment → cg_solvation →
cg_system` 依次调用 `/api/step/{task_id}/{step}`；`input` 使用
`{"include_protein":false,"environment":"bilayer"}`。蛋白体系先通过
`/api/upload-pdb` 将文件附加到同一 Task。Final Viewer 检查后调用
`POST /api/step/{task_id}/cg_system/confirm`，再提交 `/api/build`。所有布尔值
必须使用 JSON `true` 或 `false`。

## 6. 输出包与运行

### 6.1 湿体系结构

```text
input.gro
input.pdb                    # 可选；超过 PDB 格式容量时省略
topol.top
index.ndx
run_md.sh
README.txt
<topol.top 引用的力场、蛋白、脂质、配体、水和离子参数>
mdp/
  mini.mdp
  equili_<n>.mdp             # 只生成启用阶段
  production.mdp             # 单一且未分段的生产定义（如适用）
  production_<n>.mdp         # 只生成启用的生产分段
```

原子级参数文件位于 ZIP 根目录，不保证存在 `toppar/`。准确文件名取决于体系组分；
`topol.top` 是唯一权威 include 清单。`index.ndx` 按实际体系提供适用的
`System`、`SOLU`、`MEMB`、`SOLV`、`SOLU_MEMB` 组。

干 Pure Bilayer 包不包含水、离子、MDP 或 `run_md.sh`。

Martini 3 包使用 `toppar/` 保存固定参数和蛋白 ITP，并额外包含
`manifest.json` 与 `CITATIONS.json`。湿 CG 体系包含 `mdp/` 和 `run_md.sh`；
干 CG 膜只导出坐标和拓扑。

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

运行前先阅读包内 `README.txt`，检查 `topol.top`、`index.ndx`、总电荷、盒尺寸、
膜/溶剂/离子分布及每阶段 MDP。

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

确认查看的是 Ion Check 下方的 Complete Simulation System，而非旧预览。统计水
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
Task ID、步骤、完整错误信息和不含敏感结构内容的截图/日志片段。

## 附录 A：部署安全提示

- 以非 root 用户运行；任务目录只允许服务账号访问。
- 公网部署保持 loopback 监听，设置 `GMXBUILDER_DEPLOYMENT_MODE=public`；应用会
  强制 HTTPS 和全局认证。
- 配置 `GMXBUILDER_CORS_ORIGINS=https://实际域名`、
  `GMXBUILDER_TRUSTED_PROXIES=代理地址/CIDR`，并在 secrets 文件中设置成对的
  `GMXBUILDER_AUTH_USER`/`GMXBUILDER_AUTH_PASSWORD`，或至少 32 字符的
  `GMXBUILDER_ACCESS_TOKEN`。转发头只接受来自这些可信代理的值。
- 配置高强度 `GMXBUILDER_ADMIN_TOKEN`；不要把 Task ID 或管理员令牌写入日志。
- 使用 [deploy](../deploy/) 中的 systemd/nginx 模板；应用的 SQLite 限流和反向
  代理限流应同时保留。
- 使用 `uv sync --frozen --no-dev` 安装锁定环境；发布审计环境使用
  `uv sync --frozen --all-extras` 后运行 `uv run pip-audit --local`。
- 定期清理过期任务并审查依赖、Git LFS 资产及第三方许可证。
