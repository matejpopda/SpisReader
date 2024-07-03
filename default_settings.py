from pathlib import Path


class Settings:
    percentile: float = 0.05
    default_output_path: Path = Path("./temp")
    screenshot_size: int = 2

    lazy_loading: bool = True

    @classmethod
    def print_current_settings(cls):
        print("--------------------------------------------------")
        print("Current settings are: ")
        print(f"Default output path is {cls.default_output_path}")
        print(f"Lazy loading is {cls.lazy_loading}")
        print("--------------------------------------------------")