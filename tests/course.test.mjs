import assert from 'node:assert/strict';
import test from 'node:test';
import { parseCatalog, courseMarkdown } from '../.vitepress/course.mjs';

test('catalog preserves module order, practical lessons and filenames with spaces', () => {
  const source = '模块一：基础\n\n[第1课：概览](<第1课：概览.md>)\n模块二：实战\n[第2课【实战】：KV Cache](<第2课【实战】：KV Cache.md>)';
  assert.deepEqual(parseCatalog(source), [
    { text: '模块一：基础', items: [{ text: '第1课：概览', link: '/推理模型优化/第1课：概览.md' }] },
    { text: '模块二：实战', items: [{ text: '第2课【实战】：KV Cache', link: '/推理模型优化/第2课【实战】：KV Cache.md' }] }
  ]);
});

test('catalog rejects cloud links rather than silently losing lessons', () => {
  assert.throws(() => parseCatalog('模块一：基础\n[第1课](https://example.com/lesson)'), /本地/);
});

test('catalog rejects duplicate lessons', () => {
  assert.throws(() => parseCatalog('模块一：基础\n[一](<一.md>)\n[一](<一.md>)'), /重复/);
});

// A real Markdown parser exercises the renderer boundary, including code fences.
test('adds a course title without mistaking Python comments for headings', async () => {
  const { default: MarkdownIt } = await import('markdown-it');
  const md = new MarkdownIt().use(courseMarkdown);
  const html = md.render('```python\n# comment\nprint("{{ value }}")\n```', { path: '/repo/推理模型优化/第1课：概览.md' });
  assert.match(html, /^<h1>第1课：概览<\/h1>/);
  assert.match(html, /# comment/);
  assert.match(html, /\{\{ value \}\}/);
});

test('normalizes exported fence labels and leaves code bytes intact', async () => {
  const { default: MarkdownIt } = await import('markdown-it');
  const md = new MarkdownIt().use(courseMarkdown);
  const tokens = md.parse('```plain&#x20;text\n# heading\n<model>\n```', { path: '/repo/推理模型优化/示例.md' });
  const fence = tokens.find(t => t.type === 'fence');
  assert.equal(fence.info, 'text');
  assert.equal(fence.content, '# heading\n<model>\n');
});

test('keeps existing real titles and does not alter non-course pages', async () => {
  const { default: MarkdownIt } = await import('markdown-it');
  const md = new MarkdownIt().use(courseMarkdown);
  assert.equal(md.render('# 原标题\n\n内容', { path: '/repo/推理模型优化/示例.md' }), '<h1>原标题</h1>\n<p>内容</p>\n');
  assert.equal(md.render('首页', { path: '/repo/index.md' }), '<p>首页</p>\n');
});

test('local service examples open externally without altering lesson links', async () => {
  const { default: MarkdownIt } = await import('markdown-it');
  const md = new MarkdownIt().use(courseMarkdown);
  const html = md.render('<http://localhost:30000/docs> [课程](<第1课.md>)', { path: '/repo/推理模型优化/示例.md' });
  assert.match(html, /href="http:\/\/localhost:30000\/docs" target="_blank" rel="noreferrer noopener"/);
  assert.match(html, /<a href="[^\"]+\.md">课程<\/a>/);
});
