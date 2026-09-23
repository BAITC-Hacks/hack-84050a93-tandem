"""Public-data-only planning ledger; never holds a real environment or truth."""


class ScheduleEnvironment:
    def __init__(self, budget, contacts, pilots, cells, beliefs, channels, reference):
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = pilots
        self.cells = cells
        self.priors = {(b.current, b.segment, b.target): b.prior_mean for b in beliefs}
        self.channels = channels
        self.reference = reference
        self.schedule = []

    def run_pilot(self, *, target_tariff, channel, n_customers,
                  filter_current_tariff, filter_arpu_segment):
        n = min(n_customers, self.cells[filter_current_tariff, filter_arpu_segment]["n"],
                self.remaining_contacts)
        info = self.channels[channel]
        cost = float(info["cost_per_contact"])
        if cost:
            n = min(n, int(self.remaining_budget // cost))
        if n < 10 or self.pilots_left <= 0:
            raise ValueError("Infeasible planning action")
        action = dict(target_tariff=target_tariff, channel=channel, n_customers=int(n),
                      filter_current_tariff=filter_current_tariff,
                      filter_arpu_segment=filter_arpu_segment)
        self.schedule.append(action)
        self.remaining_budget -= n * cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        mean = self.priors[filter_current_tariff, filter_arpu_segment, target_tariff]
        # Explicit fantasy observation, only used to allocate the fixed schedule.
        return {"n_customers": int(n), "observed_lift_ratio":
                mean * float(info["conversion_multiplier"]) / self.reference}
