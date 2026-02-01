import { useEffect, useState, useMemo } from "react";
import { useParams, useSearchParams, useNavigate, Link } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { SEO, PaperCard, ProgressIndicator, DocLoader } from "@/components";
import { useAllResults, usePipelineStatus } from "@/hooks";
import type { ReportData } from "@/lib/types";

// Brutalist shadow styles
const brutalShadow = { boxShadow: "3px 3px 0 #F3787A" };

/**
 * Build a citation map from report data.
 * Extracts all paper IDs and assigns sequential numbers.
 */
function buildCitationMap(report: ReportData): Map<string, number> {
  const paperIds = new Set<string>();
  
  // Extract from introduction
  const introMatches = report.introduction.matchAll(/\[([A-Za-z0-9_:-]+)\]/g);
  for (const match of introMatches) {
    paperIds.add(match[1]);
  }
  
  // Extract from current research
  for (const item of report.current_research) {
    const summaryMatches = item.summary.matchAll(/\[([A-Za-z0-9_:-]+)\]/g);
    for (const match of summaryMatches) {
      paperIds.add(match[1]);
    }
    for (const id of item.paper_ids) {
      paperIds.add(id);
    }
  }
  
  // Extract from open problems
  for (const problem of report.open_problems) {
    const textMatches = problem.text.matchAll(/\[([A-Za-z0-9_:-]+)\]/g);
    for (const match of textMatches) {
      paperIds.add(match[1]);
    }
    for (const id of problem.paper_ids) {
      paperIds.add(id);
    }
  }
  
  // Extract from conclusion
  const conclusionMatches = report.conclusion.matchAll(/\[([A-Za-z0-9_:-]+)\]/g);
  for (const match of conclusionMatches) {
    paperIds.add(match[1]);
  }
  
  // Create numbered map
  const citationMap = new Map<string, number>();
  let counter = 1;
  for (const id of paperIds) {
    citationMap.set(id, counter++);
  }
  
  return citationMap;
}

/**
 * Get URL for a paper ID.
 * Handles OpenAlex (W...) and Semantic Scholar (S2:...) IDs.
 */
function getPaperUrl(paperId: string): string {
  // OpenAlex Work IDs start with W
  if (paperId.startsWith("W")) {
    return `https://openalex.org/${paperId}`;
  }
  
  // Semantic Scholar IDs start with S2:
  if (paperId.startsWith("S2:")) {
    const s2Id = paperId.substring(3); // Remove "S2:" prefix
    return `https://www.semanticscholar.org/paper/${s2Id}`;
  }
  
  // Fallback: try Google Scholar search
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(paperId)}`;
}

/**
 * Helper function to format citations in text with numbered pills.
 * Converts [W123456789] to clickable numbered pill links.
 */
function formatCitationsWithNumbers(text: string, citationMap: Map<string, number>): string {
  return text.replace(/\[([A-Za-z0-9_:-]+)\]/g, (_match, paperId) => {
    const number = citationMap.get(paperId) ?? "?";
    const url = getPaperUrl(paperId);
    return `<a href="${url}" target="_blank" rel="noopener" class="citation-pill" title="${paperId}">${number}</a>`;
  });
}

/**
 * Report page component - brutalist design.
 */
export default function ReportPage() {
  const { queryId } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const jobId = searchParams.get("job");

  const { results, metadata, isLoading, error, notFound } = useAllResults(queryId);
  const reportData = results?.report ?? null;
  const papers = results?.snowball?.papers ?? [];

  // Build citation map for numbered references
  const citationMap = useMemo(() => {
    if (!reportData) return new Map<string, number>();
    return buildCitationMap(reportData);
  }, [reportData]);

  // Poll pipeline status if job is provided
  const { data: pipelineStatus } = usePipelineStatus(jobId);

  const [finalizingSince, setFinalizingSince] = useState<number | null>(null);

  const shouldFinalize =
    !!jobId && pipelineStatus?.status === "completed" && !reportData;
  const finalizingTimedOut =
    finalizingSince !== null && Date.now() - finalizingSince >= 45_000;
  const isFinalizing = shouldFinalize && !finalizingTimedOut;

  // When the pipeline completes, keep the `job` param until the report artifact is readable.
  // Otherwise we can briefly render "report not available" due to cached /all results.
  useEffect(() => {
    if (shouldFinalize) {
      setFinalizingSince((prev) => prev ?? Date.now());
    }
  }, [shouldFinalize]);

  useEffect(() => {
    if (!isFinalizing || !queryId) return;

    const tick = () => {
      void queryClient.invalidateQueries({ queryKey: ["results", queryId] });
      void queryClient.invalidateQueries({ queryKey: ["metadata", queryId] });
    };

    tick();
    const intervalId = window.setInterval(tick, 1500);
    return () => window.clearInterval(intervalId);
  }, [isFinalizing, queryId, queryClient]);

  useEffect(() => {
    if (pipelineStatus?.status === "completed" && jobId && reportData && queryId) {
      setFinalizingSince(null);
      navigate(`/report/${queryId}`, { replace: true });
    }
  }, [pipelineStatus?.status, jobId, reportData, queryId, navigate]);

  // Display title
  const queryTitle = reportData?.query || queryId?.replace(/_/g, " ") || "Research Report";
  const pageTitle = `Research Report: ${queryTitle}`;

  // Meta description
  const introText = reportData?.introduction || "";
  const metaDescription =
    introText.length > 160
      ? introText.substring(0, 157) + "..."
      : introText || `Research report on ${queryTitle}`;

  // Canonical URL
  const canonicalUrl = `${window.location.origin}/report/${queryId}`;

  // JSON-LD structured data
  const jsonLd = reportData
    ? {
        "@context": "https://schema.org",
        "@type": "ScholarlyArticle",
        headline: pageTitle,
        description: metaDescription,
        author: {
          "@type": "Organization",
          name: "Paper Navigator",
        },
        datePublished: metadata?.report_generated_at || reportData.generated_at,
        dateModified: metadata?.last_updated || reportData.generated_at,
        mainEntityOfPage: {
          "@type": "WebPage",
          "@id": canonicalUrl,
        },
        about: {
          "@type": "Thing",
          name: queryTitle,
        },
        numberOfPages: reportData.total_papers_used,
      }
    : undefined;

  // 404 state
  if (notFound && !jobId) {
    return (
      <>
        <SEO title="Report Not Found" noindex />
        <div className="text-center py-12 px-4">
          <h1 className="text-3xl font-bold text-black text-shadow-brutal lowercase">
            report not found
          </h1>
          <p className="text-gray-600 mt-2 lowercase">
            the report for "{queryId?.replace(/_/g, " ")}" does not exist.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 mt-6 justify-center">
            <Link
              to="/"
              className="btn btn-brutal lowercase"
              style={brutalShadow}
            >
              create new report
            </Link>
            <Link
              to="/queries"
              className="btn border-2 border-black bg-white text-black hover:bg-gray-50 lowercase"
            >
              view existing reports
            </Link>
          </div>
        </div>
      </>
    );
  }

  // Error state
  if (error) {
    return (
      <>
        <SEO title="Error" noindex />
        <div className="container container-lg py-12 px-4">
          <div className="p-4 border-2 border-black border-l-4 border-l-red-500 bg-white">
            <strong className="lowercase">error loading report:</strong> {error.message}
          </div>
        </div>
      </>
    );
  }

  // Loading state (no job, just fetching)
  if (isLoading && !jobId) {
    return (
      <>
        <SEO title="Loading..." noindex />
        <div className="flex justify-center items-center min-h-[400px]">
          <div className="text-center">
            <DocLoader size="lg" className="mx-auto mb-4" />
            <p className="text-gray-600 lowercase">loading report...</p>
          </div>
        </div>
      </>
    );
  }

  // Generating state (has job, polling)
  if (!reportData && jobId && pipelineStatus) {
    // Check for pipeline failure
    if (pipelineStatus.status === "failed") {
      return (
        <>
          <SEO title="Generation Failed" noindex />
          <div className="container container-lg py-12 px-4">
            <div className="p-4 border-2 border-black border-l-4 border-l-red-500 bg-white">
              <strong className="lowercase">report generation failed:</strong>{" "}
              {pipelineStatus.error || "unknown error"}
            </div>
            {pipelineStatus.alerts && pipelineStatus.alerts.length > 0 && (
              <div className="stack stack-sm mt-4">
                {pipelineStatus.alerts.slice(-3).map((alert, idx) => (
                  <div
                    key={`${alert.ts ?? "alert"}-${idx}`}
                    className={`p-3 border-2 border-black bg-white ${
                      alert.level === "error" ? "border-l-4 border-l-red-500" : ""
                    }`}
                  >
                    <strong className="lowercase">{alert.level || "warning"}:</strong>{" "}
                    {alert.message}
                  </div>
                ))}
              </div>
            )}
            <Link
              to="/"
              className="btn border-2 border-black bg-white text-black hover:bg-gray-50 mt-4 lowercase inline-block"
            >
              try again
            </Link>
          </div>
        </>
      );
    }

    if (pipelineStatus.status === "completed" && isFinalizing) {
      return (
        <>
          <SEO title="Finalizing Report..." noindex />
          <div className="flex justify-center items-center min-h-[400px]">
            <div className="text-center">
              <DocLoader size="lg" className="mx-auto mb-4" />
              <p className="text-gray-600 lowercase">
                finalizing report artifacts...
              </p>
              <p className="text-gray-500 text-sm mt-2 lowercase">
                this can take a few seconds after completion.
              </p>
            </div>
          </div>
        </>
      );
    }

    if (pipelineStatus.status !== "completed") {
      return (
        <>
          <SEO title="Generating Report..." noindex />
          <ProgressIndicator
            status={pipelineStatus}
            queryTitle={queryId?.replace(/_/g, " ")}
          />
        </>
      );
    }
  }

  // Report display
  if (!reportData) {
    const prettyQuery = metadata?.query || queryId?.replace(/_/g, " ") || "this query";
    const paperCount =
      papers.length ||
      results?.snowball?.total_accepted ||
      metadata?.snowball_count ||
      0;

    const hasAnyPapers = papers.length > 0;

    return (
      <>
        <SEO title={`Report Unavailable: ${prettyQuery}`} noindex />
        <div className="container container-lg py-12 px-4">
          <div className="bg-white border-2 border-black p-6" style={brutalShadow}>
            <h1 className="text-3xl font-bold text-black text-shadow-brutal lowercase">
              report not available
            </h1>
            <p className="text-gray-700 mt-2 lowercase">
              {paperCount === 0
                ? "we couldn't find any papers for this query, so there is no report to show."
                : "we found papers for this query, but the report artifact is missing or empty."}
            </p>

            <div className="flex flex-col sm:flex-row gap-3 mt-6">
              <Link to="/" className="btn btn-brutal lowercase" style={brutalShadow}>
                try a new search
              </Link>
              <Link
                to="/queries"
                className="btn border-2 border-black bg-white text-black hover:bg-gray-50 lowercase"
              >
                back to queries
              </Link>
            </div>

            {paperCount === 0 && (
              <div className="mt-6 p-4 border-2 border-black border-l-4 bg-white" style={{ borderLeftColor: "#F3787A" }}>
                <p className="text-sm text-gray-700 lowercase">
                  try broadening the query, adding a study type (e.g., "systematic review"), or removing overly specific terms.
                </p>
              </div>
            )}

            {hasAnyPapers && (
              <div className="mt-8">
                <h2 className="text-black font-bold text-lg mb-3 lowercase">
                  papers found (without report)
                </h2>
                <ul className="list-disc pl-5 space-y-2">
                  {[...papers]
                    .sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0))
                    .slice(0, 10)
                    .map((p) => (
                      <li key={p.paper_id} className="text-sm text-gray-800">
                        <a
                          href={getPaperUrl(p.paper_id)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:underline"
                        >
                          {p.title}
                        </a>
                        {p.year ? <span className="text-gray-500"> ({p.year})</span> : null}
                      </li>
                    ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <SEO
        title={pageTitle}
        description={metaDescription}
        canonical={canonicalUrl}
        jsonLd={jsonLd}
      />

      <div className="max-w-7xl mx-auto py-12 px-4">
        <div className="lg:flex lg:gap-8">
          {/* Desktop TOC Sidebar */}
          <aside className="hidden lg:block lg:w-64 lg:flex-shrink-0">
            <nav
              className="sticky top-20 bg-white p-6 border-2 border-black"
              style={brutalShadow}
            >
              <h3 className="text-base font-bold text-black mb-4 lowercase">table of contents</h3>
              <TableOfContentsLinks reportData={reportData} />
            </nav>
          </aside>

          {/* Main Content */}
          <article className="flex-1 max-w-4xl">
            {/* Header */}
            <header
              className="bg-black text-white p-8 mb-8 border-2 border-black"
              style={brutalShadow}
            >
              <h1 className="text-white text-2xl md:text-3xl mb-4 lowercase">
                research report: <span>{reportData.query}</span>
              </h1>
              <div className="flex gap-2 flex-wrap">
                <span className="inline-flex items-center px-3 py-1 text-sm border border-white text-white lowercase">
                  {new Date(reportData.generated_at).toLocaleDateString()}
                </span>
                <span className="inline-flex items-center px-3 py-1 text-sm border border-white text-white lowercase">
                  {reportData.total_papers_used} papers used
                </span>
              </div>
            </header>

            {/* LLM Disclaimer */}
            <div className="mb-8 p-4 border-2 border-black bg-gray-50 text-sm text-gray-700">
              <strong className="lowercase">disclaimer:</strong>{" "}
              <span className="lowercase">
                this report was generated using LLMs (with citations of papers). due to the nature of LLMs, they can hallucinate and make mistakes.
              </span>
            </div>

            {(reportData.total_papers_used === 0 ||
              (reportData.current_research.length === 0 &&
                reportData.paper_cards.length === 0 &&
                !reportData.introduction.trim())) && (
              <div className="mb-8 p-4 border-2 border-black border-l-4 bg-white" style={{ borderLeftColor: "#F3787A" }}>
                <strong className="lowercase">empty report:</strong>{" "}
                <span className="lowercase">
                  no usable papers were available to generate a full report. try rerunning with a broader query.
                </span>
              </div>
            )}

            {/* Mobile TOC (accordion) */}
            <MobileTOC reportData={reportData} />

            {/* Introduction */}
            <ReportSection id="introduction" title="introduction">
              <p
                className="leading-relaxed"
                dangerouslySetInnerHTML={{ __html: formatCitationsWithNumbers(reportData.introduction, citationMap) }}
              />
            </ReportSection>

            {/* Current Research */}
            <ReportSection id="current-research" title="current research">
              {reportData.current_research.map((item, idx) => (
                <div
                  key={idx}
                  id={`research-${idx}`}
                  className="mb-8 pb-6 border-b border-gray-200 last:mb-0 last:pb-0 last:border-b-0"
                >
                  <h3 className="text-black font-bold text-lg mb-4">{item.title}</h3>
                  <p
                    className="leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: formatCitationsWithNumbers(item.summary, citationMap) }}
                  />
                  {item.paper_ids.length > 0 && (
                    <ReferencesAccordion
                      paperIds={item.paper_ids}
                      paperCards={reportData.paper_cards}
                      citationMap={citationMap}
                    />
                  )}
                </div>
              ))}
            </ReportSection>

            {/* Open Problems */}
            {reportData.open_problems.length > 0 && (
              <ReportSection id="open-problems" title="open problems">
                {reportData.open_problems.map((problem, idx) => (
                  <div
                    key={idx}
                    className="mb-6 p-6 bg-white border-2 border-black border-l-4"
                    style={{ borderLeftColor: "#F3787A" }}
                  >
                    <h3 className="text-black font-bold text-lg mb-4">{problem.title}</h3>
                    <p
                      className="leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: formatCitationsWithNumbers(problem.text, citationMap) }}
                    />
                    {problem.paper_ids.length > 0 && (
                      <div className="mt-4 flex gap-1 flex-wrap items-center">
                        <span className="text-sm text-gray-600 lowercase mr-1">sources:</span>
                        {problem.paper_ids.map((paperId) => {
                          const number = citationMap.get(paperId) ?? "?";
                          const url = getPaperUrl(paperId);
                          return (
                            <a
                              key={paperId}
                              href={url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="citation-pill"
                              title={paperId}
                            >
                              {number}
                            </a>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </ReportSection>
            )}

            {/* Conclusion */}
            <ReportSection id="conclusion" title="conclusion">
              <p
                className="leading-relaxed"
                dangerouslySetInnerHTML={{ __html: formatCitationsWithNumbers(reportData.conclusion, citationMap) }}
              />
            </ReportSection>

            {/* Sources */}
            <ReportSection id="sources" title="sources">
              <p className="text-gray-600 text-sm mb-6 lowercase">
                {papers.length} papers were analyzed for this report. click on any paper
                to view it on openalex.
              </p>

              {reportData.paper_cards.length > 0 && (
                <div className="grid gap-4">
                  {reportData.paper_cards.map((card) => (
                    <PaperCard key={card.id} card={card} />
                  ))}
                </div>
              )}
            </ReportSection>
          </article>
        </div>
      </div>
    </>
  );
}

/**
 * Report section wrapper component - flat layout with brutalist title.
 */
function ReportSection({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="mb-10">
      <h2
        className="text-black font-bold text-xl mb-6 pb-2 border-b-2 border-black lowercase inline-block pr-4"
        style={brutalShadow}
      >
        {title}
      </h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/**
 * Table of Contents links - reusable for desktop and mobile.
 */
function TableOfContentsLinks({ reportData }: { reportData: ReportData }) {
  return (
    <ul className="list-none p-0 m-0">
      <li className="mb-2">
        <a href="#introduction" className="text-gray-700 hover:text-black no-underline hover:underline lowercase">
          introduction
        </a>
      </li>
      <li className="mb-2">
        <a href="#current-research" className="text-gray-700 hover:text-black no-underline hover:underline lowercase">
          current research
        </a>
        {reportData.current_research.length > 0 && (
          <ul className="list-none p-0 ml-4 mt-1">
            {reportData.current_research.map((item, idx) => (
              <li key={idx} className="text-sm mb-1">
                <a href={`#research-${idx}`} className="text-gray-600 hover:text-black no-underline hover:underline">
                  {item.title}
                </a>
              </li>
            ))}
          </ul>
        )}
      </li>
      {reportData.open_problems.length > 0 && (
        <li className="mb-2">
          <a href="#open-problems" className="text-gray-700 hover:text-black no-underline hover:underline lowercase">
            open problems
          </a>
        </li>
      )}
      <li className="mb-2">
        <a href="#conclusion" className="text-gray-700 hover:text-black no-underline hover:underline lowercase">
          conclusion
        </a>
      </li>
      <li>
        <a href="#sources" className="text-gray-700 hover:text-black no-underline hover:underline lowercase">
          sources
        </a>
      </li>
    </ul>
  );
}

/**
 * Mobile TOC - collapsible accordion.
 */
function MobileTOC({ reportData }: { reportData: ReportData }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="lg:hidden mb-8">
      <div className="border-2 border-black bg-white" style={brutalShadow}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full p-4 flex justify-between items-center text-left hover:bg-gray-50 transition-colors"
        >
          <span className="text-base font-bold text-black lowercase">table of contents</span>
          <svg
            className={`w-5 h-5 transition-transform ${isOpen ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {isOpen && (
          <div className="p-4 pt-0 border-t border-black">
            <TableOfContentsLinks reportData={reportData} />
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * References Accordion - collapsible list of referenced papers.
 */
function ReferencesAccordion({
  paperIds,
  paperCards,
  citationMap,
}: {
  paperIds: string[];
  paperCards: Array<{ id: string; title: string }>;
  citationMap: Map<string, number>;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mt-4 border border-black">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-3 flex justify-between items-center bg-white hover:bg-gray-50 transition-colors"
      >
        <span className="text-sm font-medium lowercase">
          referenced papers ({paperIds.length})
        </span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="p-4 border-t border-black bg-white">
          <ul className="list-none p-0 m-0">
            {paperIds.map((paperId) => {
              const card = paperCards.find((c) => c.id === paperId);
              const number = citationMap.get(paperId) ?? "?";
              const url = getPaperUrl(paperId);
              return (
                <li key={paperId} className="mb-2 flex items-baseline gap-2 text-sm last:mb-0">
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="citation-pill flex-shrink-0"
                    title={paperId}
                  >
                    {number}
                  </a>
                  {card && <span className="text-gray-600">{card.title}</span>}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
