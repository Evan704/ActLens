/** ActLens mark: a lens over a 2x2 heatmap. Keep in sync with public/favicon.svg. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <circle cx="14" cy="14" r="8" fill="none" stroke="#fff" strokeWidth="2.6" />
      <path d="M20 20l6.5 6.5" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" />
      <rect x="9.6" y="9.6" width="3.7" height="3.7" rx=".6" fill="#fff" />
      <rect x="14.7" y="9.6" width="3.7" height="3.7" rx=".6" fill="#fff" fillOpacity=".35" />
      <rect x="9.6" y="14.7" width="3.7" height="3.7" rx=".6" fill="#fff" fillOpacity=".35" />
      <rect x="14.7" y="14.7" width="3.7" height="3.7" rx=".6" fill="#fff" fillOpacity=".7" />
    </svg>
  );
}
