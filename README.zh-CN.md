<p align="center">
  <img src="src/gmxbuilder/web/static/assets/gmxbuilder-logo.png" alt="GMXBUILDER 标志" width="520">
</p>

<h1 align="center">GMXBUILDER</h1>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

<p align="center"><strong>基于检查点的 GROMACS 分子模拟体系构建工具</strong></p>

GMXBUILDER 用于准备膜蛋白、脂质双分子层和溶液相分子动力学体系的
GROMACS 输入包。网页端采用分步检查点流程，用户在 Viewer 中确认的坐标
就是最终分配拓扑和导出时使用的坐标。

## 支持的工作流

- **Bilayer Builder**：处理并定向膜蛋白，构建兼容的单组分或混合脂质膜，
  随后进行溶剂化和离子添加。
- **Pure Bilayer System**：构建不含蛋白的双分子层，可选择是否添加水和
  离子。干膜导出仅包含坐标与拓扑。
- **Solvator**：在不构建膜的情况下，为蛋白、标准线性 DNA/RNA、蛋白–核酸、
  蛋白–配体或核酸–配体体系建立水盒。
- **Martini 3 Builder**：以独立流水线映射标准蛋白，并构建 Martini 3 水相、
  平面膜或蛋白–膜粗粒化体系。

每次 Check 都会生成任务独立的坐标检查点。最终构建只对最后确认的坐标分配
拓扑并打包，不会重新构建膜、溶剂或离子。

力场、水模型、脂质和分子参数后端会按照兼容性动态过滤。请查询当前安装，
不要假定每种分子都能与每种力场组合：

```bash
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
```

## 环境要求

- Linux 和 Python 3.10 或更高版本。
- Git LFS，用于获取随版本提供的脂质资产。
- 可正常执行的 GROMACS，用于拓扑检查和后续模拟。
- 仅在需要 GPU 计算时要求支持 CUDA 的 GROMACS。
- 参数化新的 GAFF2 分子时，需要独立安装 AmberTools/ACPYPE；项目不打包
  这些可执行程序。
- 使用 Martini 3 工作流时，需要安装包声明的固定版本 Martinize2/Vermouth、
  COBY 和 MDTraj。
- 推荐安装 `uv`，使用 `uv.lock` 完成精确且带哈希校验的环境安装。

## 安装

如需交互式完成本地安装并创建持久用户服务，执行：

```bash
./install-local.sh
```

脚本默认仅监听 `127.0.0.1:7788`，使用检测到的 CPU 核心数的一半，并选择接近
“每 4 个已分配核心对应一个并发任务槽”的可整除槽位数。脚本会建立项目虚拟
环境、安装依赖与预构建资产，并启动 `gmxbuilder.service`。同一任务的步骤仍然
严格串行；计算得到的每任务线程份额只用于当前活动步骤内部。

非回环地址必须明确选择 `trusted-lan` 模式。该模式没有最终用户登录，只能放在
受信私有网络防火墙之后。

也可以继续使用以下手动安装方式：

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

`prebuilt-assets install` 会先校验归档，再向用户缓存安装缺失文件，不会覆盖
已经存在的缓存文件。
公开仓库及其公开 Release 资产无需 GitHub 访问令牌。

## 网页端

启动本地服务：

```bash
gmxbuilder serve
```

访问 <http://127.0.0.1:7788/>。如需监听其他网络接口：

```bash
GMXBUILDER_DEPLOYMENT_MODE=trusted-lan \
  gmxbuilder serve --host 0.0.0.0 --port 7788
```

缺省 `local` 模式会拒绝非回环监听。公网模式还会强制全局 Basic/Bearer 认证、
HTTPS 来源和显式可信的 TLS 反向代理。

服务启动时会发现 GROMACS，并检查所选 GPU 配置是否能够实际执行。可显式
限制资源：

```bash
gmxbuilder serve \
  --cpu-cores <分配的总线程数> \
  --task-threads <每任务线程上限> \
  --max-builds <并发任务数> \
  --gpu-count <GPU数量>
```

`--task-threads` 必须能整除 `--cpu-cores`。同一个 Task 的检查点步骤始终严格
串行；每任务线程上限只用于当前步骤内部经过等价性验证的数值计算核及外部
工具。最终打包槽位占满时，任务会进入可在服务重启后恢复的队列，网页会显示
队列位置和预计开始时间。

如需指定物理 GPU 或调整顺序，请在启动前设置 `CUDA_VISIBLE_DEVICES`。
使用 `--gpu-count 0` 可强制仅使用 CPU。

### 网页工作流

Bilayer Builder 的用户步骤为：

1. Input Structure
2. Force Field
3. Structure Processing
4. Protein Orientation
5. Membrane Builder
6. Solvent & Box
7. Ions
8. Simulation Parameters 与导出

每次点击 **Check** 都会保存检查点。最终导出从最后确认的坐标检查点分配
拓扑并生成压缩包，不会重新构建膜、水或离子。

Input Structure Check 会记录能够无歧义识别的修饰残基名，并将其还原为标准
母体残基。只有当所选力场具备经过验证且原子完整的补丁时，Step 3 才会自动
建议对应修饰；补丁还必须具有完整的键合几何参数。新增重原子按照所选力场的
平衡键长和键角构建，并在接受检查点前验证局部几何与严重原子重叠。存在歧义、
无法识别或当前力场不可用的修饰会保留为明确警告，要求用户复核，程序不会
自行猜测。

内置修饰矩阵按力场分别维护，包含经过验证的磷酸化、去酰胺化和部分原生模板
修饰。R 型甲硫氨酸亚砜、trans-(2S,4R) 羟脯氨酸与明确构型的羟赖氨酸会被
数值构建并检查手性。受支持的 Amber 蛋白力场还提供独立的成对 CYS 二硫键
流程，SG–SG 键会跨检查点保存并写入最终拓扑。权威列表以
`/api/patches?force_field=...` 和
`/api/crosslink-capabilities?force_field=...` 为准。对存在不同化学含义的同名
残基采取保守处理；缺少完整参数的糖基化、长链脂化和 CHARMM 二硫键保持不可用，
不会生成近似拓扑。

## 命令行

查看当前安装实际提供的命令：

```bash
gmxbuilder --help
gmxbuilder build --help
gmxbuilder serve --help
```

常用查询命令：

```bash
gmxbuilder info --pdb protein.pdb
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder list-modules
gmxbuilder lipid-library status
gmxbuilder coarse-grained --help
```

### Martini 3 示例

构建并交互确认一个溶剂化的非对称混合膜：

```bash
gmxbuilder coarse-grained \
  --mode bilayer \
  --upper POPC:3 --upper CHOL:1 \
  --lower POPE:1 --lower POPG:1 \
  --box-xy 12 --box-z 14 --salt 0.15 \
  --threads 8 --mpi-ranks 1 --gpu-ids 0 \
  --output ./martini-system
```

蛋白–膜体系增加 `--pdb protein.pdb`；水相蛋白使用 `--mode solution
--pdb protein.pdb`。首版只接受标准蛋白残基，并明确拒绝配体、PTM、糖链、
核酸、自定义 CG 分子、曲面膜和 backmapping。当前安装的权威能力列表由
`GET /api/coarse-grained/capabilities` 返回。

### YAML 构建示例

以下示例适用于不保留小分子配体的膜蛋白：

```yaml
system_name: membrane_system
output_dir: ./output
seed: 42

modules:
  input:
    pdb: ./protein.pdb

  forcefield:
    name: amber14sb
    lipid_ff: lipid21
    ligand_ff: none
    water_model: tip3p

  structure:
    pH: 7.0
    prepare_standard_termini: true

  orient:
    method: ppm

  membrane:
    lipid_type: POPC
    box_padding: 2.0

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
gmxbuilder build --config build.yaml --output ./another-output
```

`--output` 会覆盖顶层 `output_dir`。混合膜应使用经过验证的组成格式：

```yaml
membrane:
  lipid_composition:
    upper:
      - {name: POPC, ratio: 70}
      - {name: POPE, ratio: 30}
    lower:
      - {name: POPC, ratio: 70}
      - {name: POPS, ratio: 30}
```

所有脂质都必须被所选脂质后端支持。保留的小分子同样需要兼容的参数后端；
选择 GAFF2 时还必须明确提供整数净电荷。

`simparams.hardware` 只配置生成的 `run_md.sh`，不会改变 MDP 物理参数。CPU
线程数必须能被 MPI rank 数整除，启用的 GPU 编号必须唯一。程序会按体系
类别和力场家族选择缺省协议，同时保留每个导出阶段的显式编辑能力。完整参数
契约见[用户手册](docs/GMXBUILDER_USER_MANUAL_V1.0.1.zh-CN.md)。

## HTTP API

服务运行后可访问交互式 API 文档：

- Swagger UI：<http://127.0.0.1:7788/docs>
- OpenAPI：<http://127.0.0.1:7788/openapi.json>

常用发现接口：

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 服务与运行环境状态 |
| `GET` | `/health/live` | 不含主机信息的最小匿名存活检查 |
| `GET` | `/api/hardware` | 不含服务器路径的硬件能力摘要 |
| `GET` | `/api/task-types` | 可用工作流 |
| `GET` | `/api/options` | 力场、水、脂质和离子选项 |

创建 Solvator 任务：

```bash
API=http://127.0.0.1:7788

curl -sS -X POST "$API/api/upload-pdb" \
  -F "file=@protein.pdb" \
  -F "task_type=solvator"
```

后续请求始终使用返回的 `task_id`。Step API 采用检查点机制，必须按工作流
顺序调用：

```bash
TASK_ID=<返回的任务ID>

curl -sS -X POST "$API/api/step/$TASK_ID/input" \
  -H "Content-Type: application/json" \
  -d '{"config":{}}'

curl -sS -X POST "$API/api/step/$TASK_ID/forcefield" \
  -H "Content-Type: application/json" \
  -d '{"config":{"name":"amber14sb","lipid_ff":"none","ligand_ff":"none","water_model":"tip3p"}}'

curl -sS -X POST "$API/api/step/$TASK_ID/structure" \
  -H "Content-Type: application/json" \
  -d '{"config":{"pH":7.0,"prepare_standard_termini":true}}'

curl -sS -X POST "$API/api/step/$TASK_ID/solvation" \
  -H "Content-Type: application/json" \
  -d '{"config":{"box_padding":2.0}}'

curl -sS -X POST "$API/api/step/$TASK_ID/ions" \
  -H "Content-Type: application/json" \
  -d '{"config":{"cation":"NA","anion":"CL","concentration":0.15,"neutralize":true,"ion_method":"random"}}'
```

所需 Check 检查点存在后，`/api/build` 只执行拓扑分配和打包：

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
curl -fL "$API/api/task/$TASK_ID/download" -o result.zip
```

离开页面前请保存 Task ID。服务重启后会从磁盘恢复 queued/running 的最终
打包任务；重新打开 Task 时会跳转到第一个尚未完成的可见步骤。

Bilayer API 客户端还需要在溶剂化之前执行 `orient` 和 `membrane`。
可通过 `GET /api/steps/{task_id}` 查看已有检查点。`GET /api/tasks` 是管理
接口，需要提供 `X-Admin-Token` 请求头。

## 输出包

原子级 ZIP 使用扁平参数目录。典型原子级湿体系包含：

```text
input.gro
input.pdb                    # 可选；超过 PDB 格式限制时省略
topol.top
index.ndx
run_md.sh
README.txt
forcefield.itp
ffbonded.itp
ffnonbonded.itp
<力场数据库文件>
topol_Protein_chain_A.itp    # 仅为实际存在的蛋白链生成
<脂质或配体>.itp            # 仅在体系包含相应分子时生成
<水模型>.itp
<离子参数>.itp
mdp/
  mini.mdp
  equili_<n>.mdp             # 仅包含启用的平衡阶段
  production.mdp             # 单一且未分段的生产阶段（如适用）
  production_<n>.mdp         # 便于断点续跑的生产分段
```

实际参数文件名由力场、水模型、脂质、配体、蛋白链和离子共同决定。
`topol.top` 是权威 include 清单，`index.ndx` 只包含当前体系适用的组。

干燥的 Pure Bilayer 导出会有意省略溶剂、离子、MDP 和 `run_md.sh`。

Martini 3 压缩包将版本化力场和蛋白 ITP 放在 `toppar/` 中，并包含
`manifest.json` 与 `CITATIONS.json`。湿 CG 体系包含 `mdp/` 和 `run_md.sh`；
干燥 CG 膜属于坐标/拓扑几何包，会有意省略两者。

运行湿体系：

```bash
unzip result.zip -d simulation
cd simulation
chmod +x run_md.sh
./run_md.sh
```

导出坐标是能量最小化和平衡的输入体系，并不表示已经完成生产级平衡。

## 部署

公网部署必须让 GMXBUILDER 继续绑定回环地址，并放在 TLS 反向代理后，设置
`GMXBUILDER_DEPLOYMENT_MODE=public`。公网模式如果缺少 HTTPS 来源、可信代理
地址以及强 Basic 凭据或至少 32 字符的 Bearer Token，会拒绝启动或处理请求。
转发的客户端地址和协议头只接受来自显式可信代理的值。

[deploy/](deploy/) 提供加固的 systemd、nginx 和 secrets 示例。应用限流使用
私有 SQLite 数据库，可跨进程重启保存；反向代理示例另有网络边缘限流。一键安装
生成的用户服务也启用文件系统、权限、内核和系统调用隔离，同时保留 CUDA/
GROMACS 设备访问。

## 文档

- [文档中心](docs/README.zh-CN.md)
- [用户手册 V1.0.1](docs/GMXBUILDER_USER_MANUAL_V1.0.1.zh-CN.md)
  （[PDF](docs/GMXBUILDER_USER_MANUAL_V1.0.1.zh-CN.pdf)）
- [科学兼容性与能力边界](docs/SCIENTIFIC_COMPATIBILITY.zh-CN.md)

## 许可证与第三方数据

GMXBUILDER 原创代码采用 MIT 许可证。力场数据、生成参数和外部工具仍受
各自许可证、引用要求和科学适用范围约束。重新分发前请阅读
[THIRD_PARTY_NOTICES.zh-CN.md](THIRD_PARTY_NOTICES.zh-CN.md)。
