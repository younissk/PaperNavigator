// Brutalist shadow styles
const brutalShadow = { boxShadow: "3px 3px 0 #F3787A" };

/**
 * About section for the home page - brutalist design.
 * Displays features and description of Paper Navigator.
 */
export function AboutSection() {
  return (
    <section className="py-16 px-4">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 text-center mb-2 text-shadow-brutal lowercase">
          about paper navigator
        </h2>
        <p className="text-gray-600 text-center mb-8 lowercase">
          academic literature discovery and analysis
        </p>

        <div
          className="bg-white border-2 border-black p-6"
          style={brutalShadow}
        >
          <div className="space-y-4">
            <p className="text-gray-600 lowercase">
              paper navigator helps you discover and analyze academic
              papers through intelligent search, ranking, and
              automated report generation.
              <br />
              <br />

              this was built mainly as a portfolio project and is completely free to use (as long as my azure credits last). if you have any feedback, please go to <a href="https://forms.gle/Nu4sUUeWMSJmCYR28" target="_blank" rel="noopener noreferrer">this form</a>.

              <br />
              <br />
              if you're interested in working with me, you can reach out via <a href="https://younissk.github.io" target="_blank" rel="noopener noreferrer">my website</a>.
            </p>

          </div>
        </div>
      </div>
    </section>
  );
}
