export function downloadBlob(blob: Blob, filename: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

export async function savePng(canvas: HTMLCanvasElement, filename: string): Promise<void> {
  const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, "image/png"));
  if (!blob) throw new Error("Canvas is too large to export as PNG; try a lower scale");
  downloadBlob(blob, filename);
}

/** One-page PDF sized to the figure. The figure is embedded as a high-resolution image. */
export async function savePdf(canvas: HTMLCanvasElement, logicalW: number, logicalH: number, filename: string, title: string) {
  const { jsPDF } = await import("jspdf");
  const pxToPt = 0.75;
  const w = logicalW * pxToPt;
  const h = logicalH * pxToPt;
  const doc = new jsPDF({ unit: "pt", format: [w, h], orientation: w >= h ? "landscape" : "portrait", compress: true });
  doc.setProperties({ title, creator: "ActLens" });
  doc.addImage(canvas.toDataURL("image/png"), "PNG", 0, 0, w, h, undefined, "FAST");
  doc.save(filename);
}

export function slug(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
}
