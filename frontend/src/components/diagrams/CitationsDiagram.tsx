/**
 * Diagram showing paper with references (left) and citations (right).
 * Static PNG image showing snowball search.
 */
export function CitationsDiagram() {
  return (
    <div
      className="my-4 py-2"
      role="img"
      aria-label="Diagram showing paper with references on left and citations on right"
    >
      <img
        src="/snowball.png"
        alt="Citation snowball diagram"
        className="w-full h-auto max-w-lg"
        loading="lazy"
      />
    </div>
  );
}
