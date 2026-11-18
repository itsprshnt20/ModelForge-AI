"""Recreate the bundled synthetic customer data using Python's seeded RNG."""
import csv
import math
import random
from pathlib import Path

SEED = 42
PATH = Path(__file__).with_name("synthetic_churn.csv")


def generate(path=PATH):
    randomizer = random.Random(SEED)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["customer_id", "tenure_months", "monthly_charge", "support_calls",
                         "contract", "region", "cancellation_ticket", "churn"])
        for row in range(720):
            tenure = randomizer.randint(1, 72)
            charge = round(randomizer.uniform(25, 120), 2)
            calls = randomizer.choices([0, 1, 2, 3, 4, 5], [36, 28, 16, 10, 7, 3])[0]
            contract = randomizer.choices(["Monthly", "Annual", "Two-year"], [55, 30, 15])[0]
            region = randomizer.choice(["North", "South", "East", "West"])
            logit = (-1.25 - 0.032 * tenure + 0.023 * (charge - 60) + 0.40 * calls
                     + (0.45 if contract == "Monthly" else -0.65))
            probability = 1 / (1 + math.exp(-logit))
            churn = randomizer.random() < probability
            writer.writerow([f"SYN-{row+1:04}", tenure, charge if row % 19 else "", calls,
                             contract, region if row % 31 else "",
                             "Filed" if churn else "None recorded", "Yes" if churn else "No"])


if __name__ == "__main__":
    generate()
