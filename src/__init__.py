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
    prepare_modeling_data
)

from .models import (
    ModelTrainer,
    get_feature_importance,
    create_model_card
)

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
    generate_comprehensive_fairness_report
)

from .temporal import (
    TemporalScenario,
    TemporalScenarioResult,
    TemporalGeneralizationAnalyzer
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
    create_all_figures_2025
)

# Optional explainability module (requires extra dependencies)
try:
    from .explainability import (
        SHAPExplainer,
        PermutationImportanceAnalyzer,
        PartialDependenceAnalyzer,
        CounterfactualExplainer,
        LIMEExplainer,
        compare_explanations,
        generate_explainability_report
    )
    HAS_EXPLAINABILITY = True
except ImportError:
    HAS_EXPLAINABILITY = False


__all__ = [
    # Data loading
    'load_config',
    'load_ecls_data',
    'handle_missing_values',
    'create_race_variable',
    'create_ses_variable',
    'create_at_risk_indicator',
    'prepare_modeling_data',
    # Models
    'ModelTrainer',
    'get_feature_importance',
    'create_model_card',
    # Fairness
    'FairnessEvaluator',
    'ThresholdOptimizer',
    'compare_before_after_mitigation',
    'generate_fairness_report',
    'IntersectionalFairnessAnalyzer',
    'FairnessConfidenceIntervals',
    'CalibrationFairnessAnalyzer',
    'IndividualFairnessAnalyzer',
    'generate_comprehensive_fairness_report',
    # Visualization
    'plot_roc_curves_by_group',
    'plot_calibration_curves_by_group',
    'plot_feature_importance',
    'create_all_figures',
    'plot_fairness_with_ci',
    'plot_intersectional_heatmap',
    'plot_calibration_error_comparison',
    'plot_explanation_comparison',
    'create_all_figures_2025',
    # Temporal
    'TemporalScenario',
    'TemporalScenarioResult',
    'TemporalGeneralizationAnalyzer',
    # Explainability (conditional)
    'HAS_EXPLAINABILITY'
]
