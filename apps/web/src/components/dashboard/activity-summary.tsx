const activities = ["Posture check-in", "Mountain pose session", "Weekly mobility review"];

export function ActivitySummary() {
  return (
    <section className="activity" aria-labelledby="activity-heading">
      <h2 id="activity-heading">Recent activity</h2>
      <ul>{activities.map((activity) => <li key={activity}>{activity}</li>)}</ul>
    </section>
  );
}
