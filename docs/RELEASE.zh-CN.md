# 下载与安装 GMXBUILDER 1.0.0

<p><a href="RELEASE.md">English</a> · <strong>简体中文</strong></p>

[1.0.0 发布页](https://github.com/AkaseToshiyuki/GMX-BUILDER/releases/tag/v1.0.0) 提供 Python wheel、源码发行包、中英文 PDF 手册、发布说明和 `SHA256SUMS`。完整科学运行环境建议从固定标签的源码安装：

```bash
git clone --branch v1.0.0 --depth 1 https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER
./install-local.sh
```

安装器可通过 HTTPS 获取清单固定的脂质归档，无需 Git LFS 或访问令牌。若需要显式拉取源码中的全部 LFS 文件，请安装 Git LFS 后在仓库内执行 `git lfs install` 和 `git lfs pull`。引导依赖、服务模式、资源限额与手动安装见[用户手册](USER_MANUAL.zh-CN.md)。完整锁定的 GAFF 运行环境当前面向 Linux x86-64；受管理的 Web 存储还需要 FUSE。

## 文件与目录

| 下载文件 | 用途 |
| --- | --- |
| `gmxbuilder-1.0.0-py3-none-any.whl` | Python 软件包和内置 V4 脂质库 |
| `gmxbuilder-1.0.0.tar.gz` | 含安装器、脚本和用户文档的源码发行包 |
| `USER_MANUAL.pdf`、`USER_MANUAL.zh-CN.pdf` | 本次发布的双语 PDF 手册 |
| `RELEASE_NOTES_1.0.0.md`、`RELEASE_NOTES_1.0.0.zh-CN.md` | 功能与科学边界 |
| `SHA256SUMS` | 下载文件的 SHA256 |

在下载目录运行 `sha256sum -c SHA256SUMS` 核验。wheel 安装 Python 代码和打包数据，不会单独安装 GROMACS、AmberTools、系统/FUSE 依赖或单独授权的力场；这些依赖应由匹配版本的源码安装器准备。GitHub 自动生成的源码 ZIP/tar 可能仅含 LFS 指针；离线保留内置资产时，应使用已拉取 LFS 的 Git 克隆或发布页附加的源码发行包。

源码目录中，`src/gmxbuilder/` 存放应用，`docs/` 存放用户文档，`scripts/` 存放安装辅助脚本，`deploy/` 存放服务示例。本地发行文件统一放入 `dist/1.0.0/`。运行缓存与任务存储独立于该目录；修改存储路径前请查阅手册。安装过程不启动 MD。

另见[发布说明](RELEASE_NOTES_1.0.0.zh-CN.md)、[脂质支持矩阵](release-support.zh-CN.md)、[科学边界](SCIENTIFIC_COMPATIBILITY.zh-CN.md)和[第三方声明](../THIRD_PARTY_NOTICES.zh-CN.md)。
