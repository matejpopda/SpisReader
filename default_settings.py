from pathlib import Path


class Settings:
    # Cut off for extremal values
    percentile: float = 0.05

    # Output path for all graphs and such
    default_output_path: Path = Path("./temp")

    screenshot_size: int = 2

    lazy_loading: bool = True

    reduced_numerical_kernel: bool = True  # Only save final times

    # If set to none, then the pickle is saved in the location where the spis output is
    default_pickle_path: Path | None = Path("./temp")

    @classmethod
    def print_current_settings(cls):
        print("--------------------------------------------------")
        print("Current settings are: ")
        print(f"Default output path is {cls.default_output_path}")
        print(f"Lazy loading is {cls.lazy_loading}")
        print(f"Percentile is {cls.percentile}")
        print(f"Load only final times for some of the SPIS outputs is {cls.reduced_numerical_kernel}")
        print("--------------------------------------------------")
