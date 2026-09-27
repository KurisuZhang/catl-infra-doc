# AI Infra 学习文档

大模型推理优化课程：推理基础、模型压缩、推理引擎、服务化部署、分布式推理、综合实战与 vLLM 源码解析。

- [在线阅读](https://kurisuzhang.github.io/catl-infra-doc/)
- [课程目录](<推理模型优化/《大厂AI Intra训练营二期：推理模型优化》.md>)

## Markdown 如何变成网站

课程 Markdown 和图片是源文件。VitePress 读取这些文件，将正文转换成 HTML，套上统一的导航、样式和搜索界面，输出一个静态网站。GitHub Pages 负责把生成的网站提供给浏览器访问。

```mermaid
flowchart LR
  A[本地 Markdown 和图片] -->|git push| B[GitHub main 分支]
  B --> C[GitHub Actions]
  C -->|npm ci / npm test| D[VitePress 构建]
  D -->|npm run docs:build| E[.vitepress/dist 静态文件]
  E --> F[GitHub Pages 在线阅读]
```

例如，`推理模型优化/第1课：大模型推理技术挑战.md` 会生成同名的 `.html` 页面。页面中的 Markdown 标题变成网页标题，代码块获得语法高亮，相对课程链接转换为网页链接。课程内容不会被 AI 改写。

| 文件 | 作用 |
| --- | --- |
| `index.md` | 网站首页的标题、简介和按钮 |
| `.vitepress/config.mjs` | 网站地址、导航、中文搜索和编辑入口 |
| `.vitepress/course.mjs` | 从现有目录生成课程列表；构建时补充缺少的主标题、规范飞书代码块标记 |
| `.vitepress/theme/` | 默认文档主题及蓝色配色 |
| `package.json` / `package-lock.json` | 构建命令、依赖版本及安装锁文件 |
| `.github/workflows/deploy.yml` | GitHub 上的自动构建、检查和发布步骤 |

本地执行 `npm run docs:build` 时，产物也会出现在 `.vitepress/dist/`，可用 `npm run docs:preview` 查看。产物和 `node_modules/` 不提交进 Git，GitHub Actions 会从源文件重新安装依赖并构建。根目录 README 是仓库使用说明，不作为课程网页发布。

每次发布的进度、执行命令和错误信息都可以在 [GitHub Actions](https://github.com/KurisuZhang/catl-infra-doc/actions/workflows/deploy.yml) 中查看。选择一次运行，再展开 `build` 或 `deploy` 中的步骤即可。

## 本地预览

安装 Node.js 22 或更新的 LTS 版本，然后运行：

```bash
npm ci
npm run docs:dev
```

使用命令输出的本地地址访问网站。验证生产版本：

```bash
npm test
npm run docs:build
npm run docs:preview
```

## 更新文档

直接编辑“推理模型优化”目录内的 Markdown 文件，也可以通过网站的“在 GitHub 编辑此页”按钮修改。
推送到 `main` 后，GitHub Actions 自动构建并发布；拉取请求只进行检查。

新增课程时，将文件放入同一目录，并在课程目录文档的相应模块中添加 `[标题](<文件名.md>)` 链接。
首页卡片、侧边导航和上下篇顺序均从课程目录生成，无需维护另一份列表。修改目录结构后，重启本地开发服务。

图片使用相对于文章的路径（例如 `images/图片.jpeg`）；保留 Markdown 文件名中的中文和空格即可。
构建过程会为缺少一级标题的课程补充文件名标题，并规范飞书导出的纯文本代码块标记，不修改源文档。

## GitHub Pages 发布

仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
随后推送 `main`，或在 **Actions → Build and deploy documentation → Run workflow** 手动发布。

站点基础路径为 `/catl-infra-doc/`。如果仓库改名，需要同步修改 `.vitepress/config.mjs` 中的基础路径和 GitHub 链接。
发布失败时先查看 Actions 日志，修复问题后重新推送；上一次成功部署的网站会保留。
