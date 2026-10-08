# GMXBUILDER 科学兼容性与能力边界

<p><a href="SCIENTIFIC_COMPATIBILITY.md">English</a> · <strong>简体中文</strong></p>

本文定义哪些组合可以生成以及“通过构建”不代表什么。安装版本的权威能力来自运行时注册表；本文不维护容易过期的脂质或修饰数量。

## 1. 查询当前能力

```bash
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder lipid-library status
```

服务端对应的发现接口包括：

```text
GET /api/options
GET /api/patches?force_field=<name>
GET /api/crosslink-capabilities?force_field=<name>
GET /api/terminal-capabilities?force_field=<name>
GET /api/lipid-library-status?lipid_name=<name>&force_field=<name>&lipid_ff=<backend>
GET /api/coarse-grained/capabilities
```

界面中列出一个名称不等于它能与当前力场组合。灰色禁用项及后端错误会说明可用替代方案。

## 2. 力场家族

| 蛋白家族 | 膜 | 保留小分子 | 状态 |
|---|---|---|---|
| CHARMM36m / CHARMM36 | 已验证的 CHARMM36 脂质参数 | 精确/模块化 CHARMM 模板，或用户上传的匹配 CGenFF MOL2+STR | 逐分子身份、净电荷和 penalty 验证 |
| Amber14SB / Amber99SB / Amber99SB-ILDN | 逐脂质优先 Lipid21；缺失种类在混合模型验证后使用 GAFF2 | GAFF2 + AM1-BCC | GAFF2 小分子必须提供整数净电荷 |
| OPLS-AA | 仅精确安装的 OPLS 脂质后端 | 仅精确安装的 OPLS 参数 | 当前膜目录没有通用兼容后端 |

明确拒绝 CHARMM 蛋白与 GAFF 膜/配体、Amber 蛋白与 CHARMM/CGenFF 膜/配体，以及任何产生多个冲突 `[ defaults ]` 的组合。文件能被 include 不代表交叉家族参数在科学上兼容。

水模型与最终力场组合一起在 Force Field 步骤锁定，不能在后续 Solvation 步骤静默替换。水模型策略 1.0 将组合分为 `recommended`、经过回归的 `supported`、`expert-unvalidated` 和 `prohibited`。打包中存在 `.itp` 只代表技术前提满足，不自动证明科学兼容。专家未验证组合必须通过 API/CLI 显式设置 `allow_unvalidated_water_model=true`；该确认会记录到任务元数据，但不会把组合升级为“已验证”。

## 3. 脂质与预平衡库

Amber 膜逐种脂质优先使用精确 Lipid21，缺失种类才分配 GAFF2。整膜 GAFF2 保留为显式选项。普通用户使用混合模型初始化前，GAFF2 客体必须已有在 90:10 Lipid21 POPC 宿主体系中采样、并由当前参数验证通过的构象；离线 V4 流程可生成这些证据。参数可组合、预处理成功不等于新的混合膜已获物理验证。CHARMM36 与 CHARMM36m 保留不同的构象库身份。

Lipid21 的每个 1–4 对显式携带 LJ 系数、电荷及静电缩放（GROMACS pairs function 2），不依赖母力场的全局缩放。CHARMM plasmalogen 扩展只作用于分子内的烯醚局部结构；不能通过共享类型名称全局覆盖普通脂质的二面角。

严格库 schema 4 要求参数源文件、相关导出代码及采样宿主的内容指纹。缺少指纹或指纹变化的历史构象不再有效，也不能在修改后的参数下接续旧轨迹。1.0.0 分发 V4 脂质库：37 个已准入的初始化构象库，共 134,000 个构象。具体组合和证据边界见[发布支持矩阵](release-support.zh-CN.md)。不兼容的历史资产仍不可用。

严格库条目只有同时满足下列条件才可运行时使用：

- 参数家族、schema、canonical molecular identity、拓扑/原子顺序签名及参数内容指纹匹配；
- 来自显式溶剂、半各向同性 NPT 工作流；
- 构象数量和元数据完整；
- APL、DHH、朝向和疏水核心质量门槛通过。

几何 bootstrap 构象不随软件发布，也不会被标记为预平衡库。某个拓扑存在但构象质量门槛失败时，该力场/脂质组合保持不可用，并显示其他通过验证的力场；程序不会用近似链长或同名分子替代。

界面逐条显示验证范围。**初始化构象**通过构建准入，但不证明整体膜平衡或面积收敛；**预平衡构象**还须满足记录的重复采样、平稳性、平台、重复间一致性及定量面积门槛。任何一种都不能替代新组装系统的平衡；单独修改 `ready` 或 `area_converged` 标志不能升级范围。

DAPG、DLIPG、DMPG、DOPG、DPPG、PAPG、POPG、SOPG 八种 PG 使用天然 R,S 模型。已有 Amber Lipid21 PG 模板由官方 **PGS** 头基模块的坐标和系数再生，PGR 坐标种子不是所选异构体。CHARMM 磷脂酰肌醇模板匹配注册的质子位置：POP2/PAPI/SAPI 使用 P4 质子化的 PI(4,5)P2，SOP2 使用 P3 质子化的 PI(3,4)P2，净电荷均为 -4。LYSPG 按 [EC 2.3.2.3](https://enzyme.expasy.org/EC/2.3.2.3) 使用天然 PG 的 3′-O-L-赖氨酰取代并保留头基构型。TMCL/TOCL 的相同磷脂酰臂使中央甘油碳不具手性；立体完整性检查归一化等价阴离子磷酸共振表示，同时保留真正碳不对称性的检查。这些身份与参数检查只确定构建候选；每个候选仍须独立通过模拟证据和构建准入才能选择。

发布归档是“已验证子集”，不承诺注册表中的每个兼容条目都已经通过预平衡。安装前会校验归档 SHA-256、严格库 schema，并逐条确认其中的构象库可由当前运行时加载。尚未完成或未通过生产质量门槛的组合不会进入归档，并在界面中保持不可用。

随版本提供的资产用以下命令校验和安装，具体条目数以命令输出为准：

```bash
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
gmxbuilder lipid-library status
```

管理员可使用 `gmxbuilder lipid-library status` 查看全局库状态。短时 `--test-mode` 结果只用于烟雾测试，不通过生产运行时质量门槛。

### 新脂质申请

网页不接受新脂质的提交、在线计算或重试。脂质选择器中的“Contact administrator”会提示从首页或公告板获取管理员邮箱，并在邮件中提供脂质名称、包含适用立体化学及形式电荷的 SMILES、所需力场。邮箱由维护者自行维护，不在构建器中硬编码。管理员审核、离线参数化并验证后再安排提供相应脂质。

已完成的历史任务私有脂质仍可读取和使用；未完成的旧计算不会在服务重启后自动恢复，任务会提示联系管理员或使用已安装脂质另建任务，不把未验证结果当作可用。

## 4. 小分子

- GAFF2 要求用户确认每个保留分子的整数净电荷。自动建议只用于辅助，不能替代
  对 pH、互变异构体、盐形式和配位状态的化学判断。
- GAFF2 可以补氢，但生成结果不得改变输入重原子身份和顺序。
- CHARMM 默认使用[本地兼容后端](charmm_compat/README.md)，区分精确模板与需显式确认的研究域。
  CGenFF 导入是另一选项，需要上传针对同一化学结构的 MOL2 和 STR 输出；界面会
  分别说明文件用途。高 penalty 参数需要外部量化计算验证/重拟合，不能自动
  宣称可靠。
- 金属配位、共价配体、反应中间体及耦合质子化不是通用自动参数化能力。

## 5. 核酸

标准线性 DNA/RNA 目前仅支持 Solvator 工作流，可选择以下两种力场：

| 力场 | DNA | RNA | 非共价小分子 |
|---|---|---|---|
| `charmm36m` | CHARMM36 | CHARMM36 | 本地 CHARMM 或匹配的 CGenFF MOL2+STR |
| `amber14sb_ol24` | OL24 | ff99bsc0+χOL3 | 自动 GAFF2 / AM1-BCC |

每条链按聚合物处理，使用对应力场的原生残基数据库生成 5′/3′ 羟基末端、氢原子、O3′–P 连接、键合项和积分链电荷。蛋白–DNA、蛋白–RNA 以及含有兼容非共价配体的复合物均可构建。该原生处理会将上传的核酸坐标替换为补全氢原子的 `pdb2gmx` 坐标；Step 3 Viewer 是继续流程前必须进行的坐标复核点。

`amber14sb_ol24` 在安装时下载、校验 SHA-256 并合并构成，不随代码直接分发；它在内置 ff14SB 上只补充核酸部分，要求 GROMACS 2026 或更高版本。未安装时不会显示为可用。非核酸的既有 ff14SB 参数保持一致；残基命名例外是 `RA` 在合并力场中表示 RNA 腺苷，不能再作为原 ff14SB 的镭离子名称。

断裂骨架、环状链、共价 DNA/RNA 杂合链及修饰或非标准核苷酸会明确阻断。膜内核酸及 Martini 核酸当前不可用。游离的核苷酸类配体仍进入小分子流程，不会被静默连接到核酸聚合物。

## 6. 蛋白质质子化、端基和修饰

PROPKA 建议是给定静态结构下的离散状态分配，不是 constant-pH MD，也不联合求解膜电势、配体质子化、金属配位或多构象耦合。催化位点、埋藏氢键网络和辅因子必须人工复核。

游离蛋白端基当前使用标准 NH3+/COO− 模板，未实现中性端基微观状态。边界由构建器自身的模型 pKa（N 端 8.0、C 端 3.5）推导而来而非人为选定，分为三档：pH 4.45--7.05 之间两种标准态占比均不低于 90%，构建静默通过；pH 3.50--8.00 之间标准态仍是多数物种，构建继续并报告实际占比——因为分子动力学必须为每个可滴定基团指定一个离散状态，惯例是指定优势态；超出 pKa 之后标准带电态成为少数物种，此时该指定是错误而非近似，构建中止。封端会改变真实化学构建体，只适用于确实带相应封端基团的分子，不能替代尚未支持的中性游离端基，也不能用作绕过 pH 拒绝的办法。程序不会把尚未实现的游离端基微观状态静默宣称为正确。ACE/NME 只有在目标力场包含完整模板时才会作为显式 cap residue 插入；缺少完整模板的端基选项保持禁用。

单残基修饰按目标力场原生模板开放，而不是跨力场复用。代表能力包括：

- Amber14SB 的单负/双负 Ser、Thr、Tyr 磷酸化；
- CHARMM36m 的磷酸化、若干 Lys/Arg/Cys 修饰、Tyr/Ser 修饰，以及明确构型的
  R-methionine sulfoxide、trans-(2S,4R)-hydroxyproline 和 hydroxylysine；
- Amber 蛋白家族的原生 hydroxyproline；
- 所有打包蛋白力场的 ASN/GLN 去酰胺化；
- Amber 蛋白家族的成对 CYS→CYX 二硫键模型，包括经过距离验证的跨链连接。

每个开放项必须满足：化学身份/电荷/立体构型唯一，原子增删和局部几何完整，RTP/HDB/bonded/non-bonded 参数齐全，checkpoint 后身份不变，并通过目标力场真实 `gmx grompp`。手性项还进行数值有向体积检查。

以下类别保持显式不可用：缺少完整参数的糖基化和长链脂化、化学名称冲突的 PCA/MLY/MYR 等近似模板、没有原生模板的单甲基 Arg/Cys 变体，以及尚未实现力场原生成对 patch 的 CHARMM 二硫键。相近名称或相似化学结构不能作为支持证据。

## 7. Martini 3 粗粒化边界

Martini 3 是独立分辨率和独立参数体系，不与上表的 Amber/CHARMM/OPLS 原子级分子拼接。首版使用固定的 Martini 3.0.0 资产、Martinize2/Vermouth 0.15.0 和 COBY 1.0.14。入口拆分为 Martini 3 Solvent 与 Martini 3 Bilayer：前者支持标准蛋白水相体系；后者支持指定每叶精确整数脂质数量的平面纯膜、混合/对称或非对称膜及可选标准蛋白。两者使用普通 W 水与 NA/CL，脂质权威列表由 capability API 返回。

膜体系具有独立 Orientation 步骤，采用与原子级膜流程一致的 PPM-like 能量/ 跨膜片段审查，并允许保存精确手动变换。周期盒根据确认后的分子包络、padding 和请求的膜尺寸自动推导，用户不需要输入可能截断定位蛋白的 Box Z。

折叠蛋白可使用 Elastic Network；单跨膜螺旋允许关闭；无序蛋白禁止通用网络。自动检查可以证明拓扑自洽、净电荷、盐浓度、双层朝向和几何范围，但不能判定蛋白真正的生物学内外朝向，用户必须确认 Final Viewer。

配体、PTM、糖链、核酸、任意自定义 CG 分子、混合分辨率、复杂曲面、Gō/OLIVES 及 backmapping 均不支持。输入审计会阻断，不能将原子级模块的近似参数带入 CG。当前安装的权威脂质列表、后端版本和功能边界由 `GET /api/coarse-grained/capabilities` 返回，操作流程见 [用户手册](USER_MANUAL.zh-CN.md)。

## 8. 构建质量与责任边界

通过 GMXBUILDER 表示：输入、坐标检查点、拓扑、盒、索引和 MDP 通过当前自动检查，代表体系可以进入能量最小化和平衡。它不证明任意混合比例、温度、相态、pH、蛋白构象或修饰已经达到实验级正确性，也不证明生产采样收敛。

当前原子级拓扑未实现 HMR 或虚拟位点，因此受支持的成键约束下时间步长上限为 2 fs，未约束成键时上限为 1 fs。高级 MDP override 会在合并到最终文件之后再次校验，不能绕过时间步长、约束、CHARMM 非键参数、cutoff 或周期盒边界。最短周期盒高必须大于最大 `rlist/rvdw/rcoulomb` 的两倍；溶质到周期面的距离必须覆盖该最大 cutoff（周期膜只检查 Z）。初始速度使用由构建 seed 域分离派生的正整数 seed，并写入 MDP 和包内来源信息，不再输出隐式随机的 `gen-seed=-1`。

生产前至少应检查体系总电荷、膜 APL/厚度/上下叶朝向和空隙、蛋白朝向、溶剂层、离子位置、配体电荷/参数 penalty，并按包内脚本完成能量最小化和逐级平衡。建议对关键体系进行重复轨迹及独立实验/文献对照。

仓库不附带长轨迹、checkpoint 或一次性 campaign 工具。自动化构建测试、`grompp` 和短程运行检查不应被解读为对任意体系的生产级验证。具体研究体系仍需由用户设计独立重复、收敛检查和实验/文献对照。

## 9. 主要参考资料

- [GROMACS force-field overview](https://manual.gromacs.org/documentation/current/user-guide/force-fields.html)
- [GROMACS topology format and defaults](https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html)
- [Lipid21 validation](https://pubmed.ncbi.nlm.nih.gov/34286854/)
- [GAFF](https://pubmed.ncbi.nlm.nih.gov/15116359/)
- [CGenFF](https://pmc.ncbi.nlm.nih.gov/articles/PMC2888302/)
- [CHARMM36 lipid validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922408/)
- [GROMACS pdb2gmx input databases](https://manual.gromacs.org/documentation/current/reference-manual/topologies/pdb2gmx-input-files.html)
- [RCSB Chemical Component Dictionary](https://www.rcsb.org/ligand)
- [Martini 3](https://doi.org/10.1038/s41592-021-01098-3)
- [Martini 3 tutorials](https://cgmartini.nl/docs/tutorials/Martini3/tutorials.html)


### 0.9.124 参数与诊断说明

CHARMM 的 PPCPL/PPEPL 采用 West 2020 原始 PLA18 参数，缩短乙烯醚脂链为 P-16:0；PPCPL 头基来自原生 POPC。0.9.124 修正了 C23/H3R/H3S 与 C33/H3X/H3Y 之间的电荷分配错误。旧版这两类 CHARMM 轨迹需要按修正参数重新构建；总电荷为零不足以证明逐原子电荷正确。原始文件地址、哈希与引用见英文版对应小节。

实验 CHARMM 模型继续要求显式启用；最大匹配半径和单个电荷样本不构成独立佐证。两个同系物留出样本不能证明跨骨架或物理有效性。V4 局部构象准入、面积诊断和物相验证分别报告；现有温度、POP2 精确批准及生产准入策略保持既定语义。离子报告同时给出请求浓度和含中和离子的实际计数浓度，后者以初始可替换水分子数乘以 0.0299 nm³ 估计溶剂体积，不是平衡态测量。

自动 GAFF 安装改用 AmberTools 24.8、ACPYPE 2023.10.27 与 Open Babel 3.1.1 的 Linux x86_64 完整工件锁；此前声明的 26.0 组合存在官方仓库依赖冲突。不会覆盖当前运行环境；未建立经核验锁文件的平台需使用另行验证的运行环境。
