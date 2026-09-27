import { defineConfig } from 'vitepress';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { courseDirectory, overviewFile, overviewLink, parseCatalog, courseMarkdown } from './course.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const groups = parseCatalog(readFileSync(`${root}${courseDirectory}/${overviewFile}`, 'utf8'));
const listed = new Set(groups.flatMap(group => group.items.map(item => item.link.split('/').at(-1))));
const files = readdirSync(`${root}${courseDirectory}`).filter(file => file.endsWith('.md') && file !== overviewFile);
for (const file of files) if (!listed.has(file)) throw new Error(`请先在课程目录中添加：${file}`);
for (const file of listed) if (!files.includes(file)) throw new Error(`课程目录指向不存在的文件：${file}`);

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI Infra 学习文档',
  description: '从推理基础到模型压缩、推理引擎与服务部署，系统学习大模型推理优化。',
  base: '/catl-infra-doc/',
  cleanUrls: false,
  srcExclude: ['README.md', 'node_modules/**', 'tests/**'],
  lastUpdated: true,
  markdown: { config: md => md.use(courseMarkdown) },
  transformPageData(page) {
    if (page.relativePath === 'index.md') {
      page.frontmatter.hero.actions[0].link = groups[0].items[0].link;
      page.frontmatter.hero.actions[1].link = overviewLink;
      page.frontmatter.features = groups.map((group, index) => ({
        icon: String(index + 1).padStart(2, '0'),
        title: group.text,
        details: `${group.items.length} 篇课程 · ${group.items.slice(0, 3).map(item => item.text.replace(/^.*?：/, '')).join(' / ')}`,
        link: group.items[0].link,
        linkText: '开始学习'
      }));
    }
  },
  themeConfig: {
    nav: [{ text: '首页', link: '/' }, { text: '课程概览', link: overviewLink }],
    sidebar: [{ text: '课程概览', link: overviewLink }, ...groups.map(group => ({ ...group, collapsed: true }))],
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    sidebarMenuLabel: '课程导航',
    returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '外观',
    darkModeSwitchTitle: '切换到深色模式',
    lightModeSwitchTitle: '切换到浅色模式',
    skipToContentLabel: '跳转到正文',
    lastUpdated: { text: '最后更新', formatOptions: { dateStyle: 'medium' } },
    editLink: {
      pattern: ({ relativePath }) => `https://github.com/KurisuZhang/catl-infra-doc/edit/main/${relativePath.split('/').map(encodeURIComponent).join('/')}`,
      text: '在 GitHub 编辑此页'
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/KurisuZhang/catl-infra-doc' }],
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          options: {
            tokenize: text => Array.from(new Intl.Segmenter('zh-CN', { granularity: 'word' }).segment(text)).filter(part => part.isWordLike).map(part => part.segment.toLowerCase())
          },
          searchOptions: { prefix: true, fuzzy: 0.2 }
        },
        locales: { root: { translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            displayDetails: '显示详细内容', resetButtonTitle: '清除搜索', backButtonTitle: '关闭搜索',
            noResultsText: '未找到相关结果',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        } } }
      }
    },
    footer: { message: '从原理到实战，系统学习大模型推理优化。' }
  }
});
