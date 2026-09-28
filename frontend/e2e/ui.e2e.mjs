/**
 * UI end-to-end checks against the mock backend (./ui-mock-backend.mjs). NOT part of the app build.
 *
 *   cd frontend && PORT=5202 npx vite --host 127.0.0.1 &          # dev server (the mock replaces /api)
 *   PW_DIR=/path/to/dir/with/node_modules node e2e/ui.e2e.mjs [outDir] [baseUrl]
 *
 * Needs `playwright-core` (resolved from PW_DIR, or normally) and Google Chrome (CHROME env overrides the macOS path).
 * Screenshots (light + dark) and downloaded exports land in outDir (default ./e2e-out, gitignored by convention).
 * Exit code 1 if any check fails or the page logged a console error.
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { installMock } from "./ui-mock-backend.mjs";

const require = createRequire(process.env.PW_DIR ? path.join(process.env.PW_DIR, "package.json") : import.meta.url);
const { chromium } = require("playwright-core");
const OUT = path.resolve(process.argv[2] ?? "e2e-out");
const BASE = process.argv[3] ?? "http://127.0.0.1:5202/";
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
  const logs = [];
  page.on("console", (m) => { if (m.type() === "error") logs.push(`[console.error] ${m.text()}`); });
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  const mock = await installMock(page);
  await page.goto(BASE);
  await page.waitForSelector(".pill.ready", { timeout: 20000 });
  await page.selectOption(".prompt-panel select", "mock-1");
  await page.click("button.primary");
  await page.waitForSelector(".tokens .tok");
  await page.waitForSelector(".selbar");
  await page.waitForTimeout(600);
  await fn(page, mock);
  ok(`[${scheme}] no console errors`, logs.length === 0, logs.slice(0, 3).join(" | "));
  await ctx.close();
}

const win = (page) => page.locator(".windowbar input.num").evaluateAll((els) => els.map((e) => Number(e.value)));
const layerName = (page) => page.locator(".selbar .layer-name").innerText();
const actSelect = (page, id) => page.selectOption('.selbar select[aria-label="Activation"]', id);
const settle = (page, ms = 700) => page.waitForTimeout(ms);
const shot = (page, scheme, name) => page.screenshot({ path: path.join(OUT, `${scheme}-${name}.png`) });

async function download(page, buttonText) {
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 15000 }), page.click(`.export button:has-text("${buttonText}")`)]);
  const file = path.join(OUT, dl.suggestedFilename());
  await dl.saveAs(file);
  return { name: dl.suggestedFilename(), buf: fs.readFileSync(file) };
}

await session("light", async (page, mock) => {
  const S = "light";
  // ---- defaults ----
  ok("selection bar has activation picker, slider, mode toggle", (await page.locator(".selbar select").count()) === 1 && (await page.locator(".selbar input[type=range]").count()) === 1 && (await page.locator(".selbar .seg button").count()) === 2);
  ok("no tab bar", (await page.locator(".tabs").count()) === 0);
  ok("default act is resid_post", (await page.inputValue('.selbar select[aria-label="Activation"]')) === "resid_post");
  ok("default layer is mid-network", (await layerName(page)) === "14", await layerName(page));
  const groups = await page.locator('.selbar select optgroup').evaluateAll((els) => els.map((e) => e.label));
  ok("optgroups Residual/Attention/MLP", groups.join(",") === "Residual,Attention,MLP", groups.join(","));
  const optTexts = await page.locator('.selbar select option').allInnerTexts();
  ok("options show channel counts / head shapes", optTexts.some((t) => t.includes("1024 ch")) && optTexts.some((t) => t.includes("16×128")) && optTexts.some((t) => t.includes("16 heads")), optTexts.slice(0, 2).join(" | "));
  ok("no head control for resid_post", (await page.locator(".headjump").count()) === 0);
  const w0 = await win(page);
  ok("default window 64 tokens x 128 channels", JSON.stringify(w0) === JSON.stringify([0, 64, 0, 128]), JSON.stringify(w0));
  ok("distribution panel slot rendered", (await page.locator(".side").innerText()).length > 0);
  await shot(page, S, "01-resid_post");

  // ---- head-structured activation ----
  await actSelect(page, "q");
  await settle(page);
  const wq = await win(page);
  ok("q: window is head 0 (channels 0-128), tokens kept", JSON.stringify(wq.slice(0, 4)) === JSON.stringify([0, 64, 0, 128]), JSON.stringify(wq));
  ok("q: layer kept when switching activation", (await layerName(page)) === "14");
  ok("q: head control visible with value 0", (await page.locator(".headjump").count()) === 1 && wq[4] === 0, JSON.stringify(wq));
  const colLabels = await page.evaluate(() => document.querySelector(".figure canvas") !== null);
  ok("q: canvas present", colLabels);
  await shot(page, S, "02-q-head0");

  // hover tooltip
  const fig = page.locator(".figure canvas").first();
  const box = await fig.boundingBox();
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.4);
  await page.waitForTimeout(250);
  const tip = await page.locator(".tooltip").innerText().catch(() => "");
  ok("q: tooltip says 'head H, dim D (channel N)'", /head \d+, dim \d+ \(channel \d+\)/.test(tip), tip.replace(/\n/g, " / "));
  await shot(page, S, "03-q-tooltip");

  // show all 2048 channels -> head separators visible; heads are ~ 50px wide
  await page.click('.windowbar .presets:nth-of-type(2) button:has-text("all")').catch(async () => {
    await page.locator('.windowbar .presets').nth(1).locator('button:has-text("all")').click();
  });
  await settle(page, 900);
  const wAll = await win(page);
  ok("q: 'all' preset shows every channel", wAll[2] === 0 && wAll[3] === 2048, JSON.stringify(wAll));
  await shot(page, S, "04-q-all-heads-separators");

  // head jump
  const headBox = page.locator(".headjump input.num");
  await headBox.fill("5");
  await headBox.press("Enter");
  await settle(page);
  let w = await win(page);
  ok("head jump to 5 -> channels 640-768", w[2] === 640 && w[3] === 768, JSON.stringify(w));
  await page.click('.headjump button[aria-label="Next head"]');
  await settle(page, 400);
  w = await win(page);
  ok("head ▶ -> head 6 (768-896)", w[2] === 768 && w[3] === 896 && w[4] === 6, JSON.stringify(w));
  await page.click('.headjump button[aria-label="Previous head"]');
  await page.click('.headjump button[aria-label="Previous head"]');
  await settle(page, 400);
  w = await win(page);
  ok("head ◀ ◀ -> head 4 (512-640)", w[2] === 512 && w[3] === 640 && w[4] === 4, JSON.stringify(w));
  ok("token window unchanged by head jumps", w[0] === 0 && w[1] === 64);
  await shot(page, S, "05-q-head4");
  // k has only 8 heads
  await actSelect(page, "k");
  await settle(page);
  ok("k: 8 heads x 128", (await page.locator(".headjump").innerText()).includes("of 8 × 128"), await page.locator(".headjump").innerText());

  // ---- per-activation memory ----
  await actSelect(page, "q");
  await settle(page, 400);
  ok("q remembers its head window", JSON.stringify((await win(page)).slice(2, 4)) === JSON.stringify([512, 640]), JSON.stringify(await win(page)));
  await page.locator(".windowbar input.num").nth(0).fill("10");
  await page.locator(".windowbar input.num").nth(0).press("Enter");
  await page.locator(".windowbar input.num").nth(1).fill("40");
  await page.locator(".windowbar input.num").nth(1).press("Enter");
  await settle(page, 300);
  await page.selectOption('label:has-text("Order") select', "absmax");
  await settle(page, 300);
  await actSelect(page, "gate");
  await settle(page);
  w = await win(page);
  ok("gate: token window carried over (10-40), fresh channel window", w[0] === 10 && w[1] === 40 && w[2] === 0 && w[3] === 128, JSON.stringify(w));
  ok("gate: default order is |max| ranked", (await page.inputValue('label:has-text("Order") select')) === "absmax");
  ok("gate: no head control", (await page.locator(".headjump").count()) === 0);
  await shot(page, S, "06-gate-ranked");
  await actSelect(page, "q");
  await settle(page, 300);
  ok("back to q: own order (absmax) restored", (await page.inputValue('label:has-text("Order") select')) === "absmax");
  ok("back to q: head window restored", JSON.stringify((await win(page)).slice(2, 4)) === JSON.stringify([512, 640]));
  // head jump in ranked order switches back to natural
  await page.locator(".headjump input.num").fill("2");
  await page.locator(".headjump input.num").press("Enter");
  await settle(page, 400);
  ok("head jump forces natural order", (await page.inputValue('label:has-text("Order") select')) === "natural");
  ok("…and lands on head 2", JSON.stringify((await win(page)).slice(2, 4)) === JSON.stringify([256, 384]));

  // ---- layer stepping ----
  await page.click('.selbar button[title="Next layer"]');
  ok("▶ button steps layer", (await layerName(page)) === "15", await layerName(page));
  await page.locator(".selbar input[type=range]").fill("3");
  await settle(page, 400);
  ok("slider sets layer", (await layerName(page)) === "3");
  const sliceReq = mock.requests.filter((r) => r.path.endsWith("/slice")).pop();
  ok("slice requests use `act` and the layer", sliceReq && sliceReq.q.act === "q" && sliceReq.q.layer === "3" && !("site" in sliceReq.q), JSON.stringify(sliceReq?.q));

  // ---- cursor / brush ----
  await page.locator(".selbar input[type=range]").fill("14");
  await settle(page, 400);
  const box2 = await page.locator(".figure canvas").first().boundingBox();
  await page.mouse.click(box2.x + box2.width * 0.4, box2.y + box2.height * 0.3);
  await settle(page, 300);
  const sel1 = await page.locator(".selsummary").innerText();
  ok("click places cursor (summary shows head/dim)", /cursor: token #\d+.*head \d+, dim \d+ \(channel \d+\)/.test(sel1), sel1.replace(/\n/g, " "));
  await page.keyboard.down("Shift");
  await page.mouse.move(box2.x + box2.width * 0.3, box2.y + box2.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box2.x + box2.width * 0.6, box2.y + box2.height * 0.75, { steps: 6 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await settle(page, 400);
  await shot(page, S, "07-q-cursor-brush");
  await page.click(".selsummary button");
  ok("clear button removes selection summary", (await page.locator(".selsummary").count()) === 0);
  await page.mouse.click(box2.x + box2.width * 0.5, box2.y + box2.height * 0.5);
  await page.keyboard.press("Escape");
  await settle(page, 200);
  ok("Esc clears cursor", (await page.locator(".selsummary").count()) === 0);

  // ---- exports (token view) ----
  const png = await download(page, "PNG");
  ok("PNG export is a PNG", png.buf.subarray(0, 4).toString("hex") === "89504e47" && png.buf.length > 5000, `${png.name} ${png.buf.length}B`);
  const pdf = await download(page, "PDF");
  ok("PDF export is a PDF", pdf.buf.subarray(0, 5).toString() === "%PDF-" && pdf.buf.length > 5000, `${pdf.name} ${pdf.buf.length}B`);
  ok("export file name carries act and layer", /q_L14/.test(png.name), png.name);

  // ---- across layers ----
  await actSelect(page, "resid_post");
  await settle(page, 300);
  await page.click('.selbar .seg button:has-text("Across layers")');
  await settle(page, 900);
  ok("across: statistic picker offers single channel", (await page.locator('label:has-text("Statistic per token") option').allInnerTexts()).includes("single channel"));
  await shot(page, S, "08-across-norm");
  await page.selectOption('label:has-text("Statistic per token") select', "dim");
  await settle(page, 600);
  ok("across: channel input for single channel", (await page.locator('label:has-text("channel") input.num').count()) === 1);
  await page.selectOption('label:has-text("Statistic per token") select', "kurtosis");
  await settle(page, 600);
  await shot(page, S, "09-across-kurtosis");
  const box3 = await page.locator(".figure canvas").first().boundingBox();
  await page.mouse.move(box3.x + box3.width * 0.5, box3.y + box3.height * 0.3);
  await settle(page, 300);
  ok("across: hovering a token row shows its trajectory", (await page.locator(".side .traj").count()) === 1);
  const csv = await page.locator(".side .card-title").first().innerText();
  console.log("      hovered:", csv.replace(/\n/g, " "));
  await page.mouse.click(box3.x + box3.width * 0.75, box3.y + box3.height * 0.5);
  await settle(page, 700);
  const modeOn = await page.locator(".selbar .seg button.on").innerText();
  ok("across click -> back to Layer mode", modeOn === "Layer", modeOn);
  const lyr = Number(await layerName(page));
  ok("across click set the layer (right side of the map -> late layer)", lyr > 14, String(lyr));
  ok("across click placed a token cursor", (await page.locator(".selsummary").count()) === 1, await page.locator(".selsummary").innerText().catch(() => ""));
  await shot(page, S, "10-after-across-click");

  // ---- attention ----
  await actSelect(page, "attn_pattern");
  await settle(page, 1200);
  ok("attn: across button disabled", await page.locator('.selbar .seg button:has-text("Across layers")').isDisabled());
  ok("attn: 16 head thumbnails for the layer", (await page.locator(".thumb").count()) === 16);
  const lyrAttn = await layerName(page);
  ok("attn: shared layer slider keeps the layer", lyrAttn === String(lyr), lyrAttn);
  await shot(page, S, "11-attn-grid");
  await page.selectOption('label:has-text("Sort heads by") select', "entropy");
  await settle(page, 300);
  const g = await download(page, "PNG");
  ok("attn grid PNG export", g.buf.subarray(0, 4).toString("hex") === "89504e47" && g.buf.length > 5000, `${g.name} ${g.buf.length}B`);
  await page.click('.selbar button[title="Next layer"]');
  await settle(page, 700);
  ok("attn: ▶ moves the layer and reloads the grid", (await layerName(page)) === String(lyr + 1));
  const gridReq = mock.requests.filter((r) => r.path.endsWith("/attn")).pop();
  ok("attn: request layer matches slider", gridReq && Number(gridReq.q.layer) === lyr + 1, JSON.stringify(gridReq?.q));
  await page.locator(".thumb").nth(2).click();
  await settle(page, 1000);
  ok("attn: click thumbnail opens head detail", (await page.locator(".figure canvas").count()) >= 1 && (await page.locator('.toolbar > label:has-text("Head") select').count()) === 1);
  await shot(page, S, "12-attn-detail");
  const d = await download(page, "PDF");
  ok("attn detail PDF export", d.buf.subarray(0, 5).toString() === "%PDF-", `${d.name} ${d.buf.length}B`);
  // layer x head map click
  const mm = await page.locator(".mini-map canvas").boundingBox();
  await page.mouse.click(mm.x + mm.width * 0.6, mm.y + mm.height * 0.4);
  await settle(page, 900);
  ok("attn: layer x head map click changes layer", (await layerName(page)) !== String(lyr + 1), await layerName(page));
  await shot(page, S, "13-attn-map-click");
  await page.click('button:has-text("all heads")');
  await settle(page, 300);
  // back to a token activation: layout intact
  await actSelect(page, "swiglu");
  await settle(page, 900);
  await shot(page, S, "14-swiglu");
  ok("mode is Layer after leaving attention", (await page.locator(".selbar .seg button.on").innerText()) === "Layer");
});

await session("dark", async (page) => {
  const S = "dark";
  await actSelect(page, "q_rope");
  await page.locator(".windowbar .presets").nth(1).locator('button:has-text("all")').click();
  await settle(page, 900);
  await shot(page, S, "01-q_rope-heads");
  await page.click('.selbar .seg button:has-text("Across layers")');
  await settle(page, 900);
  await shot(page, S, "02-across");
  await page.click('.selbar .seg button:has-text("Layer")');
  await actSelect(page, "attn_pattern");
  await settle(page, 1200);
  await shot(page, S, "03-attn-grid");
  await page.locator(".thumb").nth(1).click();
  await settle(page, 900);
  await shot(page, S, "04-attn-detail");
});

await browser.close();
console.log(failed ? `\n${failed} check(s) FAILED` : "\nall checks passed");
process.exit(failed ? 1 : 0);
