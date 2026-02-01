/**
 * Diagram showing query augmentation and search in OpenAlex/arXiv.
 * Static PNG image.
 */
export function QueryAugmentDiagram() {
  return (
    <div
      className="my-4 py-2"
      role="img"
      aria-label="Diagram showing how a query is augmented into multiple search queries"
    >
      <img
        src="/query_augmented.png"
        alt="Query augmentation diagram"
        className="w-full h-auto max-w-lg"
        loading="lazy"
      />
    </div>
  );
}
