# GMXBUILDER 1.0.0 文档

<p><a href="README.md">English</a> · <strong>简体中文</strong></p>

本目录是 1.0.0 的用户与部署管理员文档。当前安装的能力、校验结果和 OpenAPI 模式决定实际可用范围；灰色选项或未验收的脂质不属于已支持组合。

| 文档 | 内容 |
|---|---|
| [用户手册](USER_MANUAL.zh-CN.md)（[PDF](USER_MANUAL.zh-CN.pdf)） | 五类工作流、安装、CLI、异步 API、最终确认、输出与故障排查 |
| [发布文件](RELEASE.zh-CN.md) | 1.0.0 下载项、校验值与目录说明 |
| [发布说明](RELEASE_NOTES_1.0.0.zh-CN.md) | 正式版功能和边界 |
| [支持清单](release-support.zh-CN.md) | V4 脂质库的 37 个组合及初始化验证范围 |
| [科学兼容性](SCIENTIFIC_COMPATIBILITY.zh-CN.md) | 力场、脂质、配体、修饰、核酸和物理限制 |
| [CHARMM 配体](charmm_compat/README.md) | 本地识别、完整模板、实验模型和 CGenFF 导入 |
| [界面与任务恢复](frontend-appearance.md) | 深色/经典界面及检查点恢复 |
| [匿名资源配置](anonymous-resources.md) | 安装前置条件、任务隔离、配额与过期 |
| [Web 就绪与恢复](WEB_RELIABILITY.md) | 代理配置、健康检查与存储维护 |
| [离线 V4 队列](V4_QUEUE.md) | 管理员主动运行的模拟与证据维护 |

简要安装见[项目 README](../README.zh-CN.md)。服务中的 `/docs` 与 `/openapi.json` 提供 HTTP 接口模式。许可证与上游数据声明见 [许可证说明](../LICENSING.zh-CN.md)和[第三方声明](../THIRD_PARTY_NOTICES.zh-CN.md)。
