const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

async function main() {
  const root = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const name = process.argv[4] || 'final';
  fs.mkdirSync(output, {recursive: true});
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'review.json'), 'utf8'));
  const report = {gallery: root, summary: manifest.summary, page_errors: [], cases_checked: 0};
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const context = await browser.newContext({viewport: {width: 1600, height: 1000}});
    const page = await context.newPage();
    page.on('pageerror', error => report.page_errors.push(String(error)));
    const waitImages = () => page.waitForFunction(() => [...document.querySelectorAll('.drawing')]
      .every(image => image.complete && image.naturalWidth > 0));
    await page.goto(pathToFileURL(path.join(root, 'index.html')).href);
    await waitImages();
    assert.equal(await page.locator('#sample option').count(), manifest.figures.length);
    assert.equal(await page.locator('.pane').count(), 3);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    const originalZoom = await page.evaluate(() => zoom);
    await page.locator('#zoom-in').click();
    assert.ok(await page.evaluate(() => zoom) > originalZoom);
    const rect = await page.locator('.viewport').first().boundingBox();
    await page.mouse.move(rect.x+100, rect.y+100);
    await page.mouse.down();
    await page.mouse.move(rect.x+145, rect.y+120);
    await page.mouse.up();
    assert.ok(await page.evaluate(() => Math.abs(panX)+Math.abs(panY)) > 0);
    await page.locator('#fit').click();
    assert.deepEqual(await page.evaluate(() => [zoom, panX, panY]), [1, 0, 0]);
    await page.locator('#next').click();
    assert.equal(await page.locator('#sample').inputValue(), '1');
    for (let i=0; i<manifest.figures.length; i++) {
      await page.locator('#sample').selectOption(String(i));
      await waitImages();
      const expected = manifest.figures[i];
      const expectedViews = [expected.original, expected.current.views[expected.current.initial_view],
        expected.previous.views[expected.previous.initial_view]].filter(Boolean);
      assert.equal(await page.locator('.drawing').count(), expectedViews.length);
      const dimensions = await page.locator('.drawing').evaluateAll(nodes =>
        nodes.map(image => [image.naturalWidth, image.naturalHeight]));
      assert.deepEqual(dimensions, expectedViews.map(view => view.size));
      assert.equal(await page.locator('.pane').first().locator('.status').innerText(), expected.patent+' / '+expected.sketch);
      report.cases_checked++;
    }
    await page.locator('#filter').selectOption('missing');
    assert.equal(await page.locator('#sample option').count(), manifest.figures.length-manifest.summary.previous_outputs);
    await page.locator('#filter').selectOption('all');
    await page.locator('#search').fill('NO_SUCH_PATENT');
    assert.equal(await page.locator('#no-results').isVisible(), true);
    assert.equal(await page.locator('#sample').isDisabled(), true);
    await page.locator('#search').fill('');
    await page.locator('#sample').selectOption('0');
    await page.locator('#rating').selectOption('needs_review');
    await page.locator('#notes').fill('Browser verification, isolated test profile');
    await page.reload();
    await waitImages();
    assert.equal(await page.locator('#rating').inputValue(), 'needs_review');
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#download-feedback').click();
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), 'patent-review-feedback.csv');
    await download.saveAs(path.join(output, name+'_feedback_test.csv'));
    await page.locator('#rating').selectOption('');
    await page.locator('#notes').fill('');
    await page.screenshot({path: path.join(output, name+'_desktop.png'), fullPage: true});
    for (const viewport of [{width:1280,height:800}, {width:390,height:844}]) {
      await page.setViewportSize(viewport);
      await waitImages();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const boxes = await page.locator('.pane').evaluateAll(nodes => nodes.map(node => {
        const head=node.querySelector('.pane-head').getBoundingClientRect();
        const view=node.querySelector('.viewport').getBoundingClientRect();
        return {headBottom:head.bottom,viewTop:view.top};
      }));
      assert.ok(boxes.every(box => box.viewTop >= box.headBottom));
      await page.screenshot({path:path.join(output, name+'_'+viewport.width+'.png'),fullPage:true});
    }
    assert.deepEqual(report.page_errors, []);
    report.status = 'pass';
  } catch (error) {
    report.status = 'fail';
    report.error = String(error.stack || error);
    throw error;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, name+'_browser.json'), JSON.stringify(report, null, 2)+'\n');
  }
  console.log(JSON.stringify({status: report.status, cases_checked: report.cases_checked, completed: manifest.summary.completed}));
}
main().catch(error => {console.error(error); process.exitCode=1;});
