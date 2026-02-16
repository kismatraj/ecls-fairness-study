"""
ECLS Fairness Study Package (2025 State-of-the-Art)
====================================================

Analyzing algorithmic fairness in cognitive development prediction models.

Includes 2025 state-of-the-art:
- ML algorithms: LightGBM, CatBoost, HistGradientBoosting, TabPFN
- Explainability: SHAP, Permutation Importance, PDP/ICE, LIME, Counterfactuals
- Fairness: Bootstrap CI, Intersectional Analysis, Calibration Fairness
"""

__version__ = "0.2.0"

from .data_loader import (
    load_config,
    load_ecls_data,
    handle_missing_values,
    create_race_variable,
    create_ses_variable,
    create_at_risk_indicator,
    prepare_modeling_data,
)

from .models import ModelTrainer, get_feature_importance, create_model_card

from .fairness import (
    FairnessEvaluator,
    ThresholdOptimizer,
    compare_before_after_mitigation,
    generate_fairness_report,
    # 2025 additions
    IntersectionalFairnessAnalyzer,
    FairnessConfidenceIntervals,
    CalibrationFairnessAnalyzer,
    IndividualFairnessAnalyzer,
    generate_comprehensive_fairness_report,
)

from .temporal import (
    TemporalScenario,
    TemporalScenarioResult,
    TemporalGeneralizationAnalyzer,
)

from .descriptives import generate_table1, save_table1
from .latex_tables import generate_all_latex_tables
from .sensitivity import SensitivityAnalyzer, OutcomeComparisonAnalyzer
from .missing_data import (
    AttritionAnalyzer,
    MICEAnalyzer,
    IPWAnalyzer,
    run_missing_data_analysis,
)

from .visualization import (
    plot_roc_curves_by_group,
    plot_calibration_curves_by_group,
    plot_feature_importance,
    create_all_figures,
    # 2025 additions
    plot_fairness_with_ci,
    plot_intersectional_heatmap,
    plot_calibration_error_comparison,
    plot_explanation_comparison,
    create_all_figures_2025,
)

# Optional explainability module (requires extra dependencies)
try:
    from .explainability import (
        SHAPExplainer,  # noqa: F401
        PermutationImportanceAnalyzer,  # noqa: F401
        PartialDependenceAnalyzer,  # noqa: F401
        CounterfactualExplainer,  # noqa: F401
        LIMEExplainer,  # noqa: F401
        compare_explanations,  # noqa: F401
        generate_explainability_report,  # noqa: F401
    )

    HAS_EXPLAINABILITY = True
except (ImportError, OSError):
    HAS_EXPLAINABILITY = False


__all__ = [
    # Data loading
    "load_config",
    "load_ecls_data",
    "handle_missing_values",
    "create_race_variable",
    "create_ses_variable",
    "create_at_risk_indicator",
    "prepare_modeling_data",
    # Models
    "ModelTrainer",
    "get_feature_importance",
    "create_model_card",
    # Fairness
    "FairnessEvaluator",
    "ThresholdOptimizer",
    "compare_before_after_mitigation",
    "generate_fairness_report",
    "IntersectionalFairnessAnalyzer",
    "FairnessConfidenceIntervals",
    "CalibrationFairnessAnalyzer",
    "IndividualFairnessAnalyzer",
    "generate_comprehensive_fairness_report",
    # Visualization
    "plot_roc_curves_by_group",
    "plot_calibration_curves_by_group",
    "plot_feature_importance",
    "create_all_figures",
    "plot_fairness_with_ci",
    "plot_intersectional_heatmap",
    "plot_calibration_error_comparison",
    "plot_explanation_comparison",
    "create_all_figures_2025",
    # Temporal
    "TemporalScenario",
    "TemporalScenarioResult",
    "TemporalGeneralizationAnalyzer",
    # Descriptives & LaTeX
    "generate_table1",
    "save_table1",
    "generate_all_latex_tables",
    # Sensitivity
    "SensitivityAnalyzer",
    "OutcomeComparisonAnalyzer",
    # Missing data
    "AttritionAnalyzer",
    "MICEAnalyzer",
    "IPWAnalyzer",
    "run_missing_data_analysis",
    # Explainability (conditional)
    "HAS_EXPLAINABILITY",
]
