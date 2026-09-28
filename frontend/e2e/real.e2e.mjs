/**
 * End-to-end checks against the REAL backend (Qwen3-0.6B). NOT part of the app build.
 *
 *   ./run.sh &                                  # serves UI + API on :8000 (or use vite + ACTLENS_API)
 *   PW_DIR=/path/to/dir/with/node_modules node e2e/real.e2e.mjs [outDir] [baseUrl]
 *
 * Loops over every activation the model offers and checks it loads and paints, then exercises head jump,
 * tooltips, the distribution panel (all modes), Across layers, exports and dark mode.
 * Exit code 1 if any check fails, a request fails (>=400) or the page logs a console error.
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(process.env.PW_DIR ? path.join(process.env.PW_DIR, "package.json") : import.meta.url);
const { chromium } = require("playwright-core");
const OUT = path.resolve(process.argv[2] ?? "e2e-out-real");
const BASE = process.argv[3] ?? "http://127.0.0.1:8000/";
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
fs.mkdirSync(OUT, { recursive: true });

let failed = 0;
const ok = (name, cond, extra = "") => {
  if (!cond) failed++;
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });

async function session(scheme, fn) {
  const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 }, deviceScaleFactor: 1, acceptDownloads: true, colorScheme: scheme });
  const page = await ctx.newPage();
  const problems = [];
  page.on("console", (m) => { if (m.type() === "error") problems.push(`[console.error] ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`[pageerror] ${e.message}`));
  page.on("response", (r) => { if (r.status() >= 400) problems.push(`[http ${r.status()}] ${r.url().replace(BASE, "/")}`); });
  await page.goto(BASE);
  await page.waitForSelector(".pill.ready", { timeout: 90000 });
  await page.selectOption(".prompt-panel select", "fact-eiffel");
  await page.click("button.primary");
  await page.waitForSelector(".tokens .tok");
  await page.waitForSelector(".selbar");
  await fn(page, problems);
  ok(`[${scheme}] no console errors / failed requests`, problems.length === 0, problems.slice(0, 4).join(" | "));
  await ctx.close();
}

const actSelect = (page, id) => page.selectOption('.selbar select[aria-label="Activation"]', id);
const shot = (page, scheme, name) => page.screenshot({ path: path.join(OUT, `${scheme}-${name}.png`) });
const win = (page) => page.locator(".windowbar input.num").evaluateAll((els) => els.map((e) => Number(e.value)));

/** Wait until the heatmap (or attention grid) stopped loading; returns ms taken. */
async function settled(page) {
  const t = Date.now();
  await page.waitForFunction(() => !document.querySelector(".heatmap-badge"), null, { timeout: 20000 });
  await page.waitForTimeout(150);
  return Date.now() - t;
}

/** Number of distinct colours in a canvas (0 = blank). */
const distinctColours = (page, sel) =>
  page.locator(sel).first().evaluate((cv) => {
    const c = cv.getContext("2d");
    const { data } = c.getImageData(0, 0, cv.width, cv.height);
    const seen = new Set();
    for (let i = 0; i < data.length && seen.size < 50; i += 4 * 37) seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
    return seen.size;
  });

await session("light", async (page, problems) => {
  const opts = await page.locator('.selbar select[aria-label="Activation"] option').evaluateAll((els) => els.map((e) => e.value));
  ok("activation picker lists all 20 activations", opts.length === 20, `got ${opts.length}`);
  const groups = await page.locator('.selbar select[aria-label="Activation"] optgroup').evaluateAll((els) => els.map((e) => e.label));
  ok("picker is grouped Residual / Attention / MLP", JSON.stringify(groups) === JSON.stringify(["Residual", "Attention", "MLP"]), JSON.stringify(groups));

  // every activation loads and paints
  const timings = {};
  for (const id of opts) {
    const before = problems.length;
    await actSelect(page, id);
    timings[id] = await settled(page);
    if (id === "attn_pattern") {
      ok(`${id}: 16 head thumbnails`, (await page.locator(".thumb").count()) === 16);
    } else {
      const n = await distinctColours(page, ".figure canvas");
      ok(`${id}: heatmap paints`, n > 8, `${n} colours, ${timings[id]} ms`);
    }
    ok(`${id}: no request/console errors`, problems.length === before, problems.slice(before).join(" | "));
  }
  console.log("first-view latency (ms):", JSON.stringify(timings));

  // head-structured activation: separators, tooltip, head jump
  await actSelect(page, "q_rope");
  await settled(page);
  const hj = page.locator(".headjump");
  ok("head jump control shown for q_rope", (await hj.count()) === 1);
  await hj.locator("input.num").fill("3");
  await hj.locator("input.num").press("Enter");
  await settled(page);
  const w = await win(page);
  ok("head jump sets window to head 3 (channels 384–512)", w[2] === 384 && w[3] === 512, JSON.stringify(w));
  const box = await page.locator(".figure canvas").first().boundingBox();
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.4);
  await page.waitForTimeout(250);
  const tip = (await page.locator(".tooltip").innerText().catch(() => "")).replace(/\n/g, " | ");
  ok("tooltip names head, dim and channel", /head 3/.test(tip) && /channel/.test(tip), tip);
  await shot(page, "light", "q_rope-head3");

  // zoom out to all heads → separators visible
  await page.locator('.windowbar .presets button:has-text("all")').last().click();
  await settled(page);
  await shot(page, "light", "q_rope-allheads");

  // distribution panel with a brush + cursor
  await actSelect(page, "swiglu");
  await settled(page);
  const b2 = await page.locator(".figure canvas").first().boundingBox();
  await page.mouse.click(b2.x + b2.width * 0.5, b2.y + b2.height * 0.5);
  await page.keyboard.down("Shift");
  await page.mouse.move(b2.x + b2.width * 0.3, b2.y + b2.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(b2.x + b2.width * 0.6, b2.y + b2.height * 0.6, { steps: 6 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await settled(page);
  const panel = page.locator('[data-testid="distribution-panel"]');
  ok("distribution panel present", (await panel.count()) === 1);
  for (const mode of ["Values", "Per channel", "Per token"]) {
    await panel.getByRole("tab", { name: mode }).click();
    await page.waitForTimeout(700);
    ok(`distribution mode "${mode}" renders without error`, (await panel.locator(".dist-error").count()) === 0 && (await panel.locator("canvas").count()) >= 1, await panel.getAttribute("data-mode"));
    if (mode === "Values") {
      for (const scope of ["window", "selection", "channel", "token", "layer"]) {
        const btn = panel.locator(`.dist-chips[aria-label="Scope"] button:text-is("${scope}")`);
        if ((await btn.count()) && (await btn.isEnabled())) {
          await btn.click();
          await page.waitForTimeout(400);
          ok(`values scope "${scope}" ok`, (await panel.locator(".dist-error").count()) === 0);
        } else ok(`values scope "${scope}" available`, false, "button missing/disabled");
      }
    }
  }
  await panel.getByRole("tab", { name: "Per channel" }).click();
  await page.waitForTimeout(700);
  await shot(page, "light", "swiglu-per-channel");
  const topBtn = panel.locator(".dist-top button, .top button").first();
  if (await topBtn.count()) {
    const cursorBefore = await page.locator(".selsummary").innerText().catch(() => "");
    await topBtn.click();
    await page.waitForTimeout(600);
    ok("clicking a top channel does not error", (await panel.locator(".dist-error").count()) === 0, cursorBefore.slice(0, 40));
  } else ok("per-channel top list has clickable rows", false);
  await panel.getByRole("tab", { name: "Per token" }).click();
  await page.waitForTimeout(600);
  await shot(page, "light", "swiglu-per-token");

  // Across layers → click a cell → back to Layer at that layer
  await page.click('.selbar .seg button:has-text("Across layers")');
  await settled(page);
  await shot(page, "light", "across-layers");
  const b3 = await page.locator(".figure canvas").first().boundingBox();
  const layerBefore = await page.locator(".selbar .layer-name").innerText();
  await page.mouse.click(b3.x + b3.width * 0.7, b3.y + b3.height * 0.5);
  await settled(page);
  const modeNow = await page.locator(".selbar .seg button.on").innerText();
  const layerAfter = await page.locator(".selbar .layer-name").innerText();
  ok("clicking a cell opens that layer in Layer mode", modeNow.includes("Layer") && !modeNow.includes("Across") && layerAfter !== layerBefore, `${layerBefore} -> ${layerAfter}, mode=${modeNow}`);

  // exports
  for (const kind of ["PNG", "PDF"]) {
    const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.locator(`.export button:has-text("${kind}")`).first().click()]);
    const f = path.join(OUT, dl.suggestedFilename());
    await dl.saveAs(f);
    const buf = fs.readFileSync(f);
    ok(`${kind} export`, kind === "PNG" ? buf.slice(1, 4).toString() === "PNG" && buf.length > 20000 : buf.slice(0, 5).toString() === "%PDF-", `${dl.suggestedFilename()} ${(buf.length / 1024) | 0}KB`);
  }

  // attention: grid → detail → back, stepping layers
  await actSelect(page, "attn_pattern");
  await settled(page);
  await shot(page, "light", "attn-grid");
  await page.locator(".thumb").nth(3).click();
  await settled(page);
  await shot(page, "light", "attn-detail");
  await page.click('.selbar button[title="Next layer"]');
  await settled(page);
  ok("attention detail survives layer step", (await page.locator(".heatmap-error").count()) === 0);
});

await session("dark", async (page) => {
  await actSelect(page, "q_norm");
  await settled(page);
  await shot(page, "dark", "q_norm");
});

await browser.close();
console.log(failed ? `\n${failed} check(s) FAILED` : "\nall checks passed");
process.exit(failed ? 1 : 0);
