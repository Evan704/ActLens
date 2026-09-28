// Visual / behavioural check of the distribution panel against mocked endpoints.
//   PORT=5203 npx vite --host 127.0.0.1 &      # dev server serving e2e/distribution-harness.html
//   PW_DIR=/path/to/dir/with/node_modules node e2e/distribution.mjs [outDir=e2e-out-dist] [baseUrl=http://127.0.0.1:5203]
// Needs playwright-core (resolved from PW_DIR, or normally) and Google Chrome (CHROME env overrides the macOS path).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { handle } from "./mock-backend.mjs";

const require = createRequire(process.env.PW_DIR ? path.join(process.env.PW_DIR, "package.json") : import.meta.url);
const { chromium } = require("playwright-core");
const OUT = path.resolve(process.argv[2] ?? "e2e-out-dist");
const BASE = process.argv[3] ?? "http://127.0.0.1:5203";
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
let failures = 0;
const ok = (name, cond, extra = "") => { if (!cond) failures++; console.log(`${cond ? "PASS" : "FAIL"}  ${name} ${extra}`); };

async function session(scheme) {
  const ctx = await browser.newContext({ viewport: { width: 760, height: 1100 }, deviceScaleFactor: 2, acceptDownloads: true, colorScheme: scheme });
  const page = await ctx.newPage();
  const logs = [];
  const requests = [];
  page.on("console", (m) => { if (["error", "warning"].includes(m.type()) && !/status of 500/.test(m.text())) logs.push(`[${m.type()}] ${m.text()}`); });
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  await page.route("**/api/run/*/{stats,axis_stats}*", async (route) => {
    const u = new URL(route.request().url());
    requests.push(u.pathname.split("/").pop() + "?" + u.searchParams.toString());
    await new Promise((r) => setTimeout(r, 120)); // visible loading state
    const { status, body } = handle(u.pathname, u.searchParams);
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto(`${BASE}/e2e/distribution-harness.html`);
  await page.waitForSelector('[data-testid="distribution-panel"] .hist canvas');
  return { ctx, page, logs, requests };
}

const panel = (page) => page.locator('[data-testid="distribution-panel"]');
const settle = (page) => page.waitForFunction(() => !document.querySelector(".dist-busy")?.textContent);
const shot = (page, name) => panel(page).screenshot({ path: `${OUT}/${name}.png` });
const chip = (page, label) => panel(page).locator(".dist-chips button", { hasText: new RegExp(`^${label.replace(/[|]/g, "\\|")}$`) });
const mode = (page, label) => panel(page).locator(".dist-seg button", { hasText: label }).click();

for (const scheme of ["light", "dark"]) {
  console.log(`--- ${scheme}`);
  const { ctx, page, logs, requests } = await session(scheme);
  const tag = (n) => `${scheme}-${n}`;
  await settle(page);

  // Values mode, window scope
  ok("values query uses act, not site", requests.some((r) => r.startsWith("stats?") && r.includes("act=resid_pre") && !r.includes("site=")), requests[0]);
  ok("selection/channel/token scopes disabled without prerequisites", (await Promise.all(["selection", "channel", "token"].map((s) => chip(page, s).isDisabled()))).every(Boolean));
  const kurt = await panel(page).locator(".kv > div", { hasText: "kurtosis" }).innerText();
  console.log("kurtosis cell:", kurt.replace(/\s+/g, " "));
  await shot(page, tag("1-values-window"));

  // scope switching
  await chip(page, "layer").click(); await settle(page);
  ok("layer scope requests the whole layer", requests.at(-1).includes("t1=48") && requests.at(-1).includes("d1=256"), requests.at(-1));
  await shot(page, tag("2-values-layer"));
  await page.click("#sel"); await page.click("#cur");
  await chip(page, "selection").click(); await settle(page);
  ok("selection scope requests brushed region", requests.at(-1).includes("d0=30") && requests.at(-1).includes("d1=50"), requests.at(-1));
  await chip(page, "channel").click(); await settle(page);
  ok("channel scope: d1=d0+1 over all tokens", requests.at(-1).includes("t0=0&t1=48&d0=37&d1=38"), requests.at(-1));
  await chip(page, "token").click(); await settle(page);
  ok("token scope: t1=t0+1 over all channels", requests.at(-1).includes("t0=5&t1=6") && requests.at(-1).includes("d1=256"), requests.at(-1));
  await shot(page, tag("3-values-token"));

  // largest |value| list picks a token
  await panel(page).locator(".dist-top button").first().click();
  ok("values top list calls onPickToken", (await page.evaluate(() => window.__calls)).some((c) => c.startsWith("onPickToken(")));

  // clearing the cursor resets the scope to window
  await page.click("#cur"); await settle(page);
  ok("scope falls back to window when the cursor disappears", (await chip(page, "window").getAttribute("class"))?.includes("on") === true);

  // log / clip toggles
  await chip(page, "layer").click(); await settle(page);
  const before = await panel(page).locator("canvas").evaluate((c) => c.toDataURL().length);
  await panel(page).getByLabel("log count").uncheck();
  const after = await panel(page).locator("canvas").evaluate((c) => c.toDataURL().length);
  ok("log toggle redraws the histogram", before !== after);
  await shot(page, tag("4-values-linear"));
  await panel(page).getByLabel("log count").check();
  await panel(page).getByLabel(/clip p0/).check(); await settle(page);
  ok("clip refetches with clip=true", requests.at(-1).includes("clip=true"), requests.at(-1));
  const note = await panel(page).locator(".hist-note").innerText();
  ok("clip shows how many values are outside", /outside/.test(note), note);
  await shot(page, tag("5-values-clipped"));
  await panel(page).getByLabel(/clip p0/).uncheck(); await settle(page);

  // hover tooltip
  const cv = panel(page).locator(".hist canvas");
  const box = await cv.boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.waitForSelector(".hist-tip");
  ok("hover tooltip shows range and count", /\[.*\) · /.test(await panel(page).locator(".hist-tip").innerText()));
  await page.mouse.move(0, 0);

  // export
  const [dl] = await Promise.all([page.waitForEvent("download"), panel(page).getByRole("button", { name: "PNG" }).click()]);
  const path = `${OUT}/${tag("export")}.png`;
  await dl.saveAs(path);
  const sz = fs.statSync(path).size;
  ok("PNG export downloads a non-trivial image", sz > 10_000 && dl.suggestedFilename().endsWith("_hist.png"), `${dl.suggestedFilename()} ${sz}B`);

  // Per channel
  await mode(page, "Per channel"); await settle(page);
  ok("per-channel uses axis_stats with axis=channel", requests.at(-1).startsWith("axis_stats?") && requests.at(-1).includes("axis=channel") && requests.at(-1).includes("stat=absmax"), requests.at(-1));
  await shot(page, tag("6-per-channel-absmax"));

  // inside the selected channel / token
  const inside = panel(page).locator('[data-testid="inside-panel"]');
  ok("per-channel: hint without a cursor", (await inside.innerText()).includes("Click a cell"), (await inside.innerText()).slice(0, 60));
  ok("per-channel: overall histogram stays", (await panel(page).locator(".hist canvas").count()) === 1);
  await page.click("#cur"); await settle(page);
  ok("per-channel: cursor channel requests /stats over one column", requests.some((r) => r.startsWith("stats?") && r.includes("d0=37") && r.includes("d1=38")), requests.at(-1));
  await inside.locator(".hist canvas").waitFor();
  ok("per-channel: inside histogram drawn next to the overall one", (await panel(page).locator(".hist canvas").count()) === 2);
  ok("per-channel: overall legend has the selected marker", (await panel(page).locator(".hist-legend .m-selected").count()) === 1);
  await shot(page, tag("6b-per-channel-inside"));
  await page.click("#cur"); await settle(page);
  ok("per-channel: hint returns when the cursor is cleared", (await inside.innerText()).includes("Click a cell"));
  const first = (await panel(page).locator('[data-testid="top-list"] button').first().innerText()).replace(/\s+/g, " ");
  ok("outlier channel 37 tops the |max| list", first.includes("ch 37"), first);
  await panel(page).locator('[data-testid="top-list"] button').first().click();
  ok("channel top list calls onPickChannel(rank)", (await page.evaluate(() => window.__calls)).includes("onPickChannel(37)"));
  await chip(page, "kurtosis").click(); await settle(page);
  ok("stat picker refetches", requests.at(-1).includes("stat=kurtosis"), requests.at(-1));
  await shot(page, tag("7-per-channel-kurtosis"));
  await chip(page, "layer").click(); await settle(page);
  ok("region scope layer", requests.at(-1).includes("d1=256"), requests.at(-1));
  await page.click("#order"); await settle(page); // order=absmax => rank != id
  const ranked = (await panel(page).locator('[data-testid="top-list"] button').first().innerText()).replace(/\s+/g, " ");
  ok("ranked order labels original id and rank", ranked.includes("ch 37") && ranked.includes("rank"), ranked);
  await panel(page).locator('[data-testid="top-list"] button').first().click();
  ok("ranked pick sends the rank, not the id", (await page.evaluate(() => window.__calls)).includes("onPickChannel(0)"));
  await page.click("#head"); await settle(page);
  ok("head-structured channel labels", /h0·37|h0·/.test(await panel(page).locator('[data-testid="top-list"]').innerText()));
  await shot(page, tag("8-per-channel-heads"));
  await page.click("#head"); await page.click("#order"); await settle(page);

  // Per token
  await mode(page, "Per token"); await settle(page);
  ok("per-token uses axis=token", requests.at(-1).includes("axis=token"), requests.at(-1));
  const t0 = (await panel(page).locator('[data-testid="top-list"] button').first().innerText()).replace(/\s+/g, " ");
  ok("token labels show index and token", /#0 .*bos/.test(t0), t0);
  await panel(page).locator('[data-testid="top-list"] button').first().click();
  ok("token top list calls onPickToken(token)", (await page.evaluate(() => window.__calls)).includes("onPickToken(0)"));
  await chip(page, "norm").click(); await settle(page);
  await page.click("#cur"); await settle(page);
  await inside.locator(".hist canvas").waitFor();
  ok("per-token: cursor token requests /stats over one row", requests.some((r) => r.startsWith("stats?") && r.includes("t0=5&t1=6") && r.includes("d1=256")), requests.at(-1));
  ok("per-token: inside histogram drawn", (await panel(page).locator(".hist canvas").count()) === 2);
  await shot(page, tag("9a-per-token-inside"));
  await page.click("#cur"); await settle(page);
  await shot(page, tag("9-per-token-norm"));
  await panel(page).locator("summary").click();
  await shot(page, tag("10-per-token-details"));

  // prop changes refetch
  const n = requests.length;
  await page.click("#layer"); await settle(page);
  ok("layer change refetches", requests.length > n && requests.at(-1).includes("layer=4"), requests.at(-1));

  // stale state: slow request keeps previous data
  await page.route("**/api/run/*/axis_stats*", async (route) => { await new Promise((r) => setTimeout(r, 800)); route.fallback(); });
  await chip(page, "std").click();
  await page.waitForSelector(".dist.is-stale");
  ok("previous data kept (stale) while refetching", (await panel(page).locator(".hist canvas").count()) === 1);
  await shot(page, tag("11-stale"));
  await settle(page);
  await page.unroute("**/api/run/*/axis_stats*");

  // error state
  await page.click("#act");
  await page.waitForSelector(".dist-error");
  ok("error state shows the backend message", /out of memory/.test(await panel(page).locator(".dist-error").innerText()));
  await shot(page, tag("12-error"));
  await page.click("#act"); // back to a good act
  await panel(page).locator(".dist-error button").click().catch(() => {});
  await settle(page);

  ok("no console errors/warnings", logs.length === 0, logs.join(" | "));
  await ctx.close();
}
await browser.close();
console.log(failures ? `${failures} FAILED` : "ALL PASSED");
process.exit(failures ? 1 : 0);
