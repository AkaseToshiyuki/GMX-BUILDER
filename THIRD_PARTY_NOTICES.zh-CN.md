# 第三方声明

<p><a href="THIRD_PARTY_NOTICES.md">English</a> · <strong>简体中文</strong></p>

本译文用于帮助阅读；许可证条款、上游声明或译文之间存在差异时，以随文件提供的
英文许可证和上游原文为准。

GMXBUILDER 将原创调度代码与科学数据格式、力场移植数据及可选外部程序组合使用。
项目 MIT 许可证只适用于 GMXBUILDER 原创代码和文档，不替代任何上游许可证、
引用要求或使用条件。

## 力场数据

以下目录包含力场数据或转换后的移植版本：

- `src/gmxbuilder/data/forcefields/amber14sb.ff`
- `src/gmxbuilder/data/forcefields/amber99sb.ff`
- `src/gmxbuilder/data/forcefields/charmm36m`
- `src/gmxbuilder/data/forcefields/charmm36`
- `src/gmxbuilder/data/forcefields/oplsaa.ff`

必须保留 `forcefield.itp`、`forcefield.doc` 及相关源文件头中的来源和科学引用。
在私有研究仓库之外分发前，仓库所有者必须核实每个力场文件的上游再分发条款。
本声明本身不授予再分发权。

## 预构建脂质资产

`src/gmxbuilder/data/prebuilt_assets/` 包含显式溶剂、半各向同性 NPT 脂质构象，
以及通过 GAFF2/AM1-BCC 工具生成的 GROMACS 拓扑与缓存。归档不包含 AmberTools、
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
