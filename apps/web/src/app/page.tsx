import { ActivitySummary } from "@/components/dashboard/activity-summary";

const metrics = [
  { label: "Health score", value: "85", detail: "+4 this month" },
  { label: "Current streak", value: "5 days", detail: "Keep moving" },
  { label: "Sessions", value: "42", detail: "All time" },
];

export default function HomePage() {
  return (
    <main className="shell">
      <nav className="nav" aria-label="Main navigation">
        <span className="brand">NeuroMotion</span>
        <span className="nav-note">Movement intelligence</span>
      </nav>
      <section className="hero">
        <p className="eyebrow">YOUR MOVEMENT, UNDERSTOOD</p>
        <h1>Build a healthier relationship with movement.</h1>
        <p className="lede">
          Capture a session, understand your form, and turn measurable progress into lasting habits.
        </p>
        <button className="primary-button" type="button">Start an assessment</button>
      </section>
      <section className="metrics" aria-label="Movement summary">
        {metrics.map((metric) => (
          <article className="metric" key={metric.label}>
            <p>{metric.label}</p>
            <strong>{metric.value}</strong>
            <span>{metric.detail}</span>
          </article>
        ))}
      </section>
      <ActivitySummary />
    </main>
  );
}
