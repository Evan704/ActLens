/** Make a token readable in a label: visible whitespace, no control characters, bounded length. */
export function fmtToken(s: string, max = 18): string {
  let out = s
    .replace(/\n/g, "↵")
    .replace(/\t/g, "⇥")
    .replace(/ /g, "·")
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u001f\u007f]/g, "�");
  if (out === "") out = "∅";
  const chars = Array.from(out);
  if (chars.length > max) out = chars.slice(0, max - 1).join("") + "…";
  return out;
}
