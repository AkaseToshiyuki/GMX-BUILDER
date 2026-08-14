# GMXBUILDER 科学兼容性与能力边界

<p><a href="SCIENTIFIC_COMPATIBILITY.md">English</a> · <strong>简体中文</strong></p>

本文定义哪些组合可以生成以及“通过构建”不代表什么。安装版本的权威能力来自
运行时注册表；本文不维护容易过期的脂质或修饰数量。

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

界面中列出一个名称不等于它能与当前力场组合。灰色禁用项及后端错误会说明可用
替代方案。

## 2. 力场家族

| 蛋白家族 | 膜 | 保留小分子 | 状态 |
|---|---|---|---|
| CHARMM36m / CHARMM36 | 已验证的 CHARMM36 脂质参数 | 精确/模块化 CHARMM 模板，或用户上传的匹配 CGenFF MOL2+STR | 逐分子身份、净电荷和 penalty 验证 |
| Amber14SB / Amber99SB / Amber99SB-ILDN | 整张膜统一使用 Lipid21 或验证通过的 GAFF2 | GAFF2 + AM1-BCC | GAFF2 小分子必须提供整数净电荷 |
| OPLS-AA | 仅精确安装的 OPLS 脂质后端 | 仅精确安装的 OPLS 参数 | 当前膜目录没有通用兼容后端 |

明确拒绝 CHARMM 蛋白与 GAFF 膜/配体、Amber 蛋白与 CHARMM/CGenFF 膜/配体，
以及任何产生多个冲突 `[ defaults ]` 的组合。文件能被 include 不代表交叉家族
参数在科学上兼容。

水模型与最终力场组合一起在 Force Field 步骤锁定。可选列表来自实际打包的
力场数据，不能在后续 Solvation 步骤静默替换。

## 3. 脂质与预平衡库

Amber 膜按“精确 Lipid21 → 完整膜统一 GAFF2 → 不可用”的顺序解析。混合膜
不能逐分子混用 Lipid21 和 GAFF2。CHARMM36 与 CHARMM36m 使用分别标记的严格
库，即使某些分子的脂质参数相同，也不会把缓存身份混为一谈。

严格库条目只有同时满足下列条件才可运行时使用：

- 参数家族、schema、canonical molecular identity 和拓扑/原子顺序签名匹配；
- 来自显式溶剂、半各向同性 NPT 工作流；
- 构象数量和元数据完整；
- APL、DHH、朝向和疏水核心质量门槛通过。

几何 bootstrap 构象不随软件发布，也不会被标记为预平衡库。某个拓扑存在但
构象质量门槛失败时，该力场/脂质组合保持不可用，并显示其他通过验证
的力场；程序不会用近似链长或同名分子替代。

发布归档是“已验证子集”，不承诺注册表中的每个兼容条目都已经通过预平衡。安装
前会校验归档 SHA-256、严格库 schema，并逐条确认其中的构象库可由当前运行时
加载。尚未完成或未通过生产质量门槛的组合不会进入归档，并在界面中保持不可用。

随版本提供的资产用以下命令校验和安装，具体条目数以命令输出为准：

```bash
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
gmxbuilder lipid-library status
```

管理员可使用 `gmxbuilder lipid-library build/queue/status` 维护全局库。短时
`--test-mode` 结果只用于烟雾测试，不通过生产运行时质量门槛。

### 任务私有自定义脂质

自定义脂质当前只开放 Amber + GAFF2。RDKit 使用立体化学 canonical identity
和 InChIKey 拒绝内置库的同一分子。接受后，任务在 GAFF2/AM1-BCC 参数化和
显式溶剂 NPT 预平衡完成前阻塞；只有质量门槛通过后才出现在当前 Task 的
Custom Lipids 中。参数和构象不会进入全局库，任务过期时一并删除。

## 4. 小分子

- GAFF2 要求用户确认每个保留分子的整数净电荷。自动建议只用于辅助，不能替代
  对 pH、互变异构体、盐形式和配位状态的化学判断。
- GAFF2 可以补氢，但生成结果不得改变输入重原子身份和顺序。
- CHARMM 小分子需要上传针对同一化学结构的 CGenFF MOL2 和 STR 输出；界面会
  分别说明文件用途。高 penalty 参数需要外部量化计算验证/重拟合，不能自动
  宣称可靠。
- 金属配位、共价配体、反应中间体及耦合质子化不是通用自动参数化能力。

## 5. 核酸

标准线性 DNA/RNA 目前仅支持 Solvator、CHARMM36m 与 CHARMM TIP3P。每条链按
聚合物处理，并使用随软件提供的 GROMACS/CHARMM36 数据库生成 5′/3′ 羟基末端、
氢原子、O3′–P 连接、键合项和积分链电荷。蛋白–DNA、蛋白–RNA 以及含有兼容
非共价 CHARMM 配体的复合物均可构建。
该原生处理会将上传的核酸坐标替换为补全氢原子的 `pdb2gmx` 坐标；Step 3
Viewer 是继续流程前必须进行的坐标复核点。

断裂骨架、环状链、共价 DNA/RNA 杂合链及修饰或非标准核苷酸会明确阻断。
Amber 核酸模型、膜内核酸及 Martini 核酸当前不可用。游离的核苷酸类配体仍进入
小分子流程，不会被静默连接到核酸聚合物。

## 6. 蛋白质质子化、端基和修饰

PROPKA 建议是给定静态结构下的离散状态分配，不是 constant-pH MD，也不联合
求解膜电势、配体质子化、金属配位或多构象耦合。催化位点、埋藏氢键网络和
辅因子必须人工复核。

标准 NH3+/COO− 端基可用。ACE/NME 只有在目标力场包含完整模板时才会作为显式
cap residue 插入；缺少完整模板的端基选项保持禁用。

单残基修饰按目标力场原生模板开放，而不是跨力场复用。代表能力包括：

- Amber14SB 的单负/双负 Ser、Thr、Tyr 磷酸化；
- CHARMM36m 的磷酸化、若干 Lys/Arg/Cys 修饰、Tyr/Ser 修饰，以及明确构型的
  R-methionine sulfoxide、trans-(2S,4R)-hydroxyproline 和 hydroxylysine；
- Amber 蛋白家族的原生 hydroxyproline；
- 所有打包蛋白力场的 ASN/GLN 去酰胺化；
- Amber 蛋白家族的成对 CYS→CYX 二硫键模型，包括经过距离验证的跨链连接。

每个开放项必须满足：化学身份/电荷/立体构型唯一，原子增删和局部几何完整，
RTP/HDB/bonded/non-bonded 参数齐全，checkpoint 后身份不变，并通过目标力场真实
`gmx grompp`。手性项还进行数值有向体积检查。

以下类别保持显式不可用：缺少完整参数的糖基化和长链脂化、化学名称冲突的
PCA/MLY/MYR 等近似模板、没有原生模板的单甲基 Arg/Cys 变体，以及尚未实现
力场原生成对 patch 的 CHARMM 二硫键。相近名称或相似化学结构不能作为支持
证据。

## 7. Martini 3 粗粒化边界

Martini 3 是独立分辨率和独立参数体系，不与上表的 Amber/CHARMM/OPLS 原子级
分子拼接。首版使用固定的 Martini 3.0.0 资产、Martinize2/Vermouth 0.15.0 和
COBY 1.0.14。入口拆分为 Martini 3 Solvent 与 Martini 3 Bilayer：前者支持
标准蛋白水相体系；后者支持指定每叶精确整数脂质数量的平面纯膜、混合/对称或
非对称膜及可选标准蛋白。两者使用普通 W 水与 NA/CL，脂质权威列表由
capability API 返回。

膜体系具有独立 Orientation 步骤，采用与原子级膜流程一致的 PPM-like 能量/
跨膜片段审查，并允许保存精确手动变换。周期盒根据确认后的分子包络、padding
和请求的膜尺寸自动推导，用户不需要输入可能截断定位蛋白的 Box Z。

折叠蛋白可使用 Elastic Network；单跨膜螺旋允许关闭；无序蛋白禁止通用网络。
自动检查可以证明拓扑自洽、净电荷、盐浓度、双层朝向和几何范围，但不能判定
蛋白真正的生物学内外朝向，用户必须确认 Final Viewer。

配体、PTM、糖链、核酸、任意自定义 CG 分子、混合分辨率、复杂曲面、Gō/OLIVES
及 backmapping 均不支持。输入审计会阻断，不能将原子级模块的近似参数带入 CG。
当前安装的权威脂质列表、后端版本和功能边界由
`GET /api/coarse-grained/capabilities` 返回，操作流程见
[用户手册](GMXBUILDER_USER_MANUAL_V1.0.2.zh-CN.md)。

## 8. 构建质量与责任边界

通过 GMXBUILDER 表示：输入、坐标检查点、拓扑、盒、索引和 MDP 通过当前自动
检查，代表体系可以进入能量最小化和平衡。它不证明任意混合比例、温度、相态、
pH、蛋白构象或修饰已经达到实验级正确性，也不证明生产采样收敛。

生产前至少应检查体系总电荷、膜 APL/厚度/上下叶朝向和空隙、蛋白朝向、溶剂
层、离子位置、配体电荷/参数 penalty，并按包内脚本完成能量最小化和逐级平衡。
建议对关键体系进行重复轨迹及独立实验/文献对照。

仓库不附带长轨迹、checkpoint 或一次性 campaign 工具。自动化构建测试、
`grompp` 和短程运行检查不应被解读为对任意体系的生产级验证。具体研究体系
仍需由用户设计独立重复、收敛检查和实验/文献对照。

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
