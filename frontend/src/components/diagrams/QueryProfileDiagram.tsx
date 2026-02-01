/**
 * Simple diagram showing query → profile transformation.
 * Static PNG image.
 */
export function QueryProfileDiagram() {
  return (
    <div
      className="my-4 py-2"
      role="img"
      aria-label="Diagram showing how a user query becomes a query profile"
    >
      <img
        src="/query_profile.png"
        alt="Query to profile transformation diagram"
        className="w-full h-auto max-w-lg"
        loading="lazy"
      />
    </div>
  );
}
