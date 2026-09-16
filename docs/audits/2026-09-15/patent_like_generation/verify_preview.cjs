const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

async function main() {
  const gallery = process.argv[2];
  assert(gallery, 'Supply the generated audit/index.html path');
  const browser = await chromium.launch({ headless: true });
  const report = [];
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.goto(pathToFileURL(path.resolve(gallery)).href);
      await page.waitForFunction(() => [...document.images].every(img => img.complete && img.naturalWidth > 0));
      const actual = await page.evaluate(() => ({
        images: document.images.length,
        articles: document.querySelectorAll('article').length,
        scrollWidth: document.documentElement.scrollWidth,
        innerWidth: window.innerWidth,
      }));
      assert.equal(actual.images, 30);
      assert.equal(actual.articles, 30);
      assert(actual.scrollWidth <= actual.innerWidth, 'Horizontal overflow');
      assert.deepEqual(errors, []);
      await page.screenshot({ path: path.join(__dirname, `gallery_${viewport.width}.png`) });
      report.push({ viewport, ...actual, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  await fs.writeFile(path.join(__dirname, 'browser_checks.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
