import { InfrastructureDiagram } from "./diagrams";

// Brutalist shadow styles
const brutalShadow = { boxShadow: "3px 3px 0 #F3787A" };

const HIGHLIGHTS = [
  {
    title: "frontend",
    description:
      "react frontend with vite and tailwind css. why github pages? because azure cant get its shit together and they disallow it for european students... I already sent a ticket and am waiting for a response.",
  },
  {
    title: "serverless & scalable",
    description:
      "azure functions and cosmos db scale automatically. If nobody uses this projet, I wont pay anything for upkeep",
  },
  {
    title: "secure secrets management",
    description:
      "api keys and credentials stored in key vault with managed identity access — no secrets in code.",
  },
  {
    title: "infrastructure as code",
    description:
      "all resources defined in bicep templates. Really cool system tbh.",
  },
  {
    title: "async job processing",
    description:
      "service bus queues decouple api requests from long-running pipeline jobs.",
  },
];

/**
 * Infrastructure section explaining the Azure cloud architecture.
 * Brutalist design matching other home page sections.
 */
export function InfrastructureSection() {
  return (
    <section id="infrastructure" className="py-16 px-4">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 text-center mb-2 text-shadow-brutal lowercase">
          infrastructure
        </h2>
        <p className="text-gray-600 text-center mb-8 lowercase">
          built on azure with infrastructure-as-code
        </p>

        <div
          className="bg-white border-2 border-black p-6"
          style={brutalShadow}
        >
          <div className="space-y-6">
            {/* Intro text */}
            <p className="text-gray-600 lowercase">
              paper navigator runs on a fully serverless azure stack. every
              resource is defined in bicep templates and deployed via github
              actions — making infrastructure changes reviewable, reproducible,
              and auditable.
            </p>

            {/* Architecture Diagram */}
            <div className="border-2 border-black bg-gray-50 p-4">
              <h3 className="text-sm font-bold text-black lowercase mb-2">
                architecture overview
              </h3>
              <InfrastructureDiagram />
            </div>

            {/* Highlights */}
            <h3 className="text-xl font-bold text-black lowercase mt-6">
              highlights
            </h3>
            <ul className="list-none p-0 m-0">
              {HIGHLIGHTS.map((item) => (
                <li
                  key={item.title}
                  className="py-4 border-b border-black last:border-b-0"
                >
                  <strong className="text-black">{item.title}</strong>
                  <p className="text-gray-600 text-sm mt-1">
                    {item.description}
                  </p>
                </li>
              ))}
            </ul>

            {/* Tech badges */}
            <div className="flex flex-wrap gap-2 pt-4">
              {[
                "azure functions",
                "cosmos db",
                "service bus",
                "key vault",
                "bicep",
                "github actions",
              ].map((tech) => (
                <span
                  key={tech}
                  className="px-3 py-1 bg-black text-white text-xs font-bold lowercase"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
