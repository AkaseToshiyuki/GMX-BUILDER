# 第三方声明

<p><a href="THIRD_PARTY_NOTICES.md">English</a> · <strong>简体中文</strong></p>

本译文用于帮助阅读；许可证条款、上游声明或译文之间存在差异时，以随文件提供的
英文许可证和上游原文为准。

GMXBUILDER 将原创调度代码与科学数据格式、力场移植数据及可选外部程序组合使用。
项目 MIT 许可证只适用于 GMXBUILDER 原创代码和文档，不替代任何上游许可证、
引用要求或使用条件。

## 力场数据

公开源码分发包含下列 GROMACS 来源的 Amber 与 OPLS 数据及其来源文件，但不分发
CHARMM36/CHARMM36m。安装器会在安装 Python 依赖前，从 MacKerell Lab 官方地址
下载两个固定 GROMACS 移植版本，校验 SHA-256，应用项目声明的 overlay 后安装到
本地目录。

所有不随公开仓库再分发的运行资产都在 `scripts/external_assets.json` 中列明官方
来源页、直接 HTTPS 下载地址、归档根目录、必需文件和固定 SHA-256。无人值守的
`install-local.sh` 会在安装 Python 依赖前处理该清单，不需要用户在浏览器中手动
下载或放置文件。

- `src/gmxbuilder/data/forcefields/amber14sb.ff`
- `src/gmxbuilder/data/forcefields/amber99sb.ff`
- `src/gmxbuilder/data/forcefields/charmm36m`（本地安装）
- `src/gmxbuilder/data/forcefields/charmm36`（本地安装）
- `src/gmxbuilder/data/forcefields/oplsaa.ff`

必须保留 `forcefield.itp`、`forcefield.doc` 及相关源文件头中的来源和科学引用。
再次分发前，分发者必须核实每个力场文件的上游再分发条款。
本声明本身不授予再分发权。公开部署必须使用
`scripts/install_external_assets.py`；更改固定来源或校验和属于需要审查的科学
变更，不能作为自动升级处理。

## 预构建脂质资产

`src/gmxbuilder/data/prebuilt_assets/` 包含显式溶剂、半各向同性 NPT 脂质构象，
以及通过 GAFF2/AM1-BCC 工具生成的 GROMACS 拓扑与缓存。安装器可从清单固定的
公开 HTTPS 媒体地址直接获取 Git LFS 负载，因此用户不需要 Git LFS 或访问令牌。
归档不包含 AmberTools、
ACPYPE 或 GROMACS 可执行代码；这些程序仍是外部依赖。生成参数仍需遵守对应
力场引用要求，并针对目标体系完成科学复核。

AmberTools 的组件许可证并不完全相同，请查阅官方分发和许可证：
<https://ambermd.org/AmberTools.php>。GROMACS 不随本项目打包，其许可证与引用
指南见 <https://www.gromacs.org/>。

## Martini 3 粗粒化工作流

`src/gmxbuilder/data/martini3/` 包含来自 Martini Force Field Initiative 的固定
版本相互作用和分子拓扑。清单记录上游提交和 SHA-256。保留的上游
`LICENSE.txt` 为 Apache-2.0，分发时必须与数据共同保留：
<https://github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters>。

粗粒化工作流依赖外部 Python 包 Vermouth/Martinize2、COBY 和 MDTraj；其许可证
及引用要求仍以上游为准。生成包内的 `CITATIONS.json` 提供模型和工具引用。

## 安装的 Python 依赖

安装器按照 `uv.lock` 解析锁定的软件包文件，不把依赖源码复制到本仓库。主要科学依赖在
其已安装包元数据中声明如下许可证；完整许可证和捆绑组件声明仍保存在对应安装包
中。

| 依赖 | 声明的许可证 |
|---|---|
| NumPy / SciPy | BSD-3-Clause 系列；NumPy 另含捆绑组件声明 |
| RDKit | BSD-3-Clause |
| OpenMM | BSD-like |
| PDBFixer | MIT |
| Vermouth / Martinize2 | Apache-2.0 |
| COBY | Apache-2.0 |
| MDTraj | LGPL-2.1-or-later |
| Matplotlib | 与 PSF 兼容的 Matplotlib 许可证 |

其他网页和打包依赖同样从锁定的软件包文件安装并保留上游元数据。若重新分发完整虚拟环境，
必须保留每个依赖的声明；本项目不授予替代条款。

## 用户上传的参数文件

用户上传的外部小分子 MOL2/STR 包只保存在对应 Task 目录，不属于仓库或预构建
发布资产。

## 浏览器可视化库

GMXBUILDER 随软件提供固定的本地浏览器库，避免 Viewer 在运行时执行可变 CDN
代码：

- 3Dmol.js 2.5.5，BSD-3-Clause；
- SmilesDrawer 2.0.3，MIT。

对应上游许可证保存在各版本目录中；
`src/gmxbuilder/web/static/vendor/ASSET_MANIFEST.json` 记录 npm 完整性值与已安装
JavaScript 文件的 SHA-256。
