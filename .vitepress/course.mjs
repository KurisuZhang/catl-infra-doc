import { basename } from 'node:path';

export const courseDirectory = '推理模型优化';
export const overviewFile = '《大厂AI Intra训练营二期：推理模型优化》.md';
export const overviewLink = `/${courseDirectory}/${overviewFile}`;

// The original directory document is the single source of course ordering.
export function parseCatalog(source) {
  const groups = [];
  const seen = new Set();
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim().replace(/^#{1,6}\s+/, '');
    if (/^(模块[一二三四五六七八九十]+：|附录：)/.test(line)) {
      groups.push({ text: line, items: [] });
      continue;
    }
    if (!groups.length || !line.startsWith('[')) continue;
    const match = line.match(/^\[([^\]]+)\]\((?:<([^>]+)>|([^\s)]+))\)$/);
    if (!match) throw new Error(`目录链接格式无效：${line}`);
    const target = decodeURIComponent(match[2] || match[3]);
    if (!target.endsWith('.md') || target.includes('/') || target.includes('\\')) {
      throw new Error(`课程目录必须使用同目录本地 Markdown 链接：${target}`);
    }
    if (seen.has(target)) throw new Error(`目录中存在重复课程：${target}`);
    seen.add(target);
    groups.at(-1).items.push({ text: match[1], link: `/${courseDirectory}/${target}` });
  }
  if (!groups.length || groups.some(group => !group.items.length)) throw new Error('课程目录缺少模块或课程');
  return groups;
}

// Operate on parsed Markdown tokens, never on the contents of code blocks.
export function courseMarkdown(md) {
  // These are addresses of services the reader runs locally, not site routes.
  md.core.ruler.push('course-local-services', state => {
    for (const token of state.tokens) {
      for (const child of token.children || []) {
        if (child.type === 'link_open' && /^https?:\/\/localhost(?=[:/]|$)/.test(child.attrGet('href') || '')) {
          child.attrSet('target', '_blank');
          child.attrSet('rel', 'noreferrer noopener');
        }
      }
    }
  });
  md.core.ruler.after('block', 'course-format', state => {
    const file = (state.env.path || '').replaceAll('\\', '/');
    if (!file.includes(`/${courseDirectory}/`)) return;
    for (const token of state.tokens) {
      if (token.type === 'fence' && /^plain(?:&#x20;|\s)+text$/i.test(token.info.trim())) token.info = 'text';
    }
    if (state.tokens.some(token => token.type === 'heading_open' && token.tag === 'h1')) return;
    const open = new state.Token('heading_open', 'h1', 1);
    const title = new state.Token('inline', '', 0);
    title.content = basename(file, '.md');
    title.children = [];
    const close = new state.Token('heading_close', 'h1', -1);
    state.tokens.unshift(open, title, close);
  });
}
