/**
 * Document icon loader component.
 * Uses the doc.svg inline with a pulsing animation for consistent branding.
 */

interface DocLoaderProps {
  /** Size variant: "sm" (16px), "md" (24px), or "lg" (40px). Defaults to "md". */
  size?: "sm" | "md" | "lg";
  /** Additional CSS classes */
  className?: string;
}

const sizes = {
  sm: { width: 16, height: 20 },
  md: { width: 24, height: 31 },
  lg: { width: 40, height: 51 },
} as const;

export function DocLoader({ size = "md", className = "" }: DocLoaderProps) {
  const { width, height } = sizes[size];

  return (
    <div className={`inline-flex items-center justify-center ${className}`}>
      <svg
        width={width}
        height={height}
        viewBox="0 0 360 460"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="animate-pulse-brutal"
        aria-label="Loading..."
        role="img"
      >
        <path
          d="M340 130H230V20"
          stroke="currentColor"
          strokeWidth="20"
          strokeMiterlimit="10"
          strokeLinecap="round"
        />
        <path
          d="M350 450H10V10H230L350 130V450Z"
          stroke="currentColor"
          strokeWidth="20"
          strokeMiterlimit="10"
          strokeLinecap="round"
        />
        <path d="M280 200H80V220H280V200Z" fill="currentColor" className="animate-line-1" />
        <path d="M280 320H80V340H280V320Z" fill="currentColor" className="animate-line-2" />
        <path d="M240 260H80V280H240V260Z" fill="currentColor" className="animate-line-3" />
      </svg>
    </div>
  );
}
