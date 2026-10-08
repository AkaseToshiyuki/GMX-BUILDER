# GMXBUILDER 1.0.0

<p><a href="https://github.com/AkaseToshiyuki/GMX-BUILDER/blob/v1.0.0/docs/RELEASE_NOTES_1.0.0.md">English</a> · <strong>简体中文</strong></p>

发布日期：2026-10-08。GMXBUILDER 提供五类 GROMACS 引导式构建工作流：全原子蛋白膜
系统、纯脂双层、溶剂化系统、Martini 3 脂双层/蛋白膜系统和 Martini 3 蛋白水相系统。

- Web、CLI 与 HTTP API 提供能力查询、逐步检查、最终结构确认和模拟输入包下载。
- 受管理的 Web 执行支持异步操作、任务恢复、有界队列和可配置资源限额；本地、带认证
  公开和匿名公开部署分别定义访问策略。
- 资产 v7 分发 37 个已准入脂质/后端初始化构象库、134,000 个构象和 67 个
  GAFF2/AM1-BCC 参数缓存。
- 精确 Lipid21、GAFF2、本地 CHARMM/CGenFF 和 Martini 路径执行各自的分子身份、
  参数及力场兼容性契约。
- 中英文手册更新安装前提、当前 API 行为、输出文件和科学边界。

支持的组合以[支持矩阵](https://github.com/AkaseToshiyuki/GMX-BUILDER/blob/v1.0.0/docs/release-support.zh-CN.md)为准。其余 299 个注册脂质/来源组合
尚未作为已准入构象库分发。初始化构象不证明膜平衡、新混合物的物理有效性或生产轨迹
收敛；蛋白质、核酸、配体、自定义脂质及研究模型限制见[科学指南](https://github.com/AkaseToshiyuki/GMX-BUILDER/blob/v1.0.0/docs/SCIENTIFIC_COMPATIBILITY.zh-CN.md)。
GROMACS 预处理及软件回归结果不能替代针对具体系统的物理验证。

安装见[下载与安装指南](https://github.com/AkaseToshiyuki/GMX-BUILDER/blob/v1.0.0/docs/RELEASE.zh-CN.md)。升级保留较新的有效构象库缓存，升级后应
检查实际可用状态。安装不启动分子动力学，也不启动离线构象库生产队列。
