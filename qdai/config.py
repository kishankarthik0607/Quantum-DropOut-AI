"""Central configuration for Quantum_DropOut_AI.

Everything marked PRESERVED is carried over unchanged from the original
main.py so the ML behaviour stays identical.
"""
from pathlib import Path

APP_NAME = "Quantum_DropOut_AI"
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- PRESERVED
TARGET_MAPPING = {
    "Graduate": 1,
    "Dropout": 0,
    "Enrolled": 2,
}
REVERSE_TARGET = {v: k for k, v in TARGET_MAPPING.items()}

# Same lookup order as the original load_data(), plus a project-relative path
# so the app finds ./data/student_dropout_data.csv from any working directory.
DATA_PATHS = [
    "C:/Users/swapn/StudentDropoutPrediction/.venv/data/student_dropout_data.csv",
    "C:/Users/swapn/StudentDropoutPrediction/data/student_dropout_data.csv",
    "./data/student_dropout_data.csv",
    "../data/student_dropout_data.csv",
    str(ROOT / "data" / "student_dropout_data.csv"),
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# ------------------------------------------------------------- PRESENTATION
# Dropout-probability bands used by the risk meter. These are fixed
# presentation thresholds, not calibrated clinical cut-offs, and the UI says so.
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.60

# Groups smaller than this are never used to state a rate comparison.
MIN_GROUP_N = 30

# A feature has to push the dropout probability up by at least this much
# (2 percentage points) before it can trigger an intervention suggestion.
INTERVENTION_MIN_PUSH = 0.02

OUTCOME_COLORS = {
    "Dropout": "#F0605D",
    "Graduate": "#35D0C4",
    "Enrolled": "#9C8CFF",
}
OUTCOME_ORDER = ["Dropout", "Enrolled", "Graduate"]

# Authentication
AUTH_DIR = ROOT / ".auth"
AUTH_FILE = AUTH_DIR / "users.json"
PBKDF2_ITERATIONS = 240_000
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 30

PAGES = ["Overview", "Explore", "Models", "Predict", "Insights"]
