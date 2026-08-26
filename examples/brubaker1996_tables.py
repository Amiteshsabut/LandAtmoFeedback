"""Print the published Brubaker and Entekhabi (1996) benchmark values."""

from landfeedback.benchmarks import brubaker1996


def main() -> None:
    tables = {
        "State statistics": brubaker1996.state_statistics(),
        "Main parameters": brubaker1996.main_parameters(),
        "Published reporting matrix": brubaker1996.published_reporting_matrix(),
        "Dry/moist soil-moisture tendency": brubaker1996.conditional_soil_moisture(),
        "Dry/moist soil-temperature tendency": brubaker1996.conditional_soil_temperature(),
    }
    print(brubaker1996.citation())
    for title, table in tables.items():
        print(f"\n{title}\n{'=' * len(title)}")
        print(table.to_string())


if __name__ == "__main__":
    main()

