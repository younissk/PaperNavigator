/**
 * Infrastructure diagram showing Azure cloud architecture.
 * Static PNG image.
 */
export function InfrastructureDiagram() {
  return (
    <div
      className="my-6 py-4"
      role="img"
      aria-label="Diagram showing PaperPilot's Azure cloud infrastructure"
    >
      <img
        src="/infra.png"
        alt="PaperPilot Azure infrastructure diagram"
        className="w-full h-auto"
        loading="lazy"
      />
    </div>
  );
}
