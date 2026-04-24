# ===== MUST BE AT VERY TOP =====
import os
import logging

# Disable Ray dashboard & metrics (as much as possible)
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_METRICS_EXPORT_PORT"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_LOG_TO_STDERR"] = "0"

# Reduce logging noise
logging.getLogger("ray").setLevel(logging.ERROR)

# ===== NOW IMPORT RAY =====
import ray


def main():
    print("Starting Ray...")

    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False
    )

    print("Ray initialized.")

    @ray.remote
    def f(x):
        return x * x

    print("Running test tasks...")

    results = ray.get([f.remote(i) for i in range(5)])

    print("Results:", results)

    ray.shutdown()
    print("Finished cleanly.")


if __name__ == "__main__":
    main()