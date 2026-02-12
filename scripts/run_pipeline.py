#!/usr/bin/env python
"""
Main Pipeline Script
====================

Run the complete analysis pipeline or individual steps.

Usage:
    python scripts/run_pipeline.py --config config.yaml
    python scripts/run_pipeline.py --step data_prep
    python scripts/run_pipeline.py --step train_models
    python scripts/run_pipeline.py --step fairness_analysis
    python scripts/run_pipeline.py --step all
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import joblib

# Suppress noisy sklearn/xgboost/lightgbm warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', message='.*use_label_encoder.*')

# Setup logging BEFORE imports to avoid basicConfig being silently ignored
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', mode='w'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_config,
    load_ecls_data,
    handle_missing_values,
    create_race_variable,
    create_ses_variable,
    create_at_risk_indicator,
    create_analytic_sample,
    get_variable_lists,
    prepare_modeling_data
)
from src.models import (
    ModelTrainer,
    get_feature_importance,
    create_model_card
)
from src.fairness import (
    FairnessEvaluator,
    ThresholdOptimizer,
    compare_before_after_mitigation,
    generate_fairness_report,
    generate_comprehensive_fairness_report,
    IntersectionalFairnessAnalyzer,
    FairnessConfidenceIntervals,
    CalibrationFairnessAnalyzer
)
from src.visualization import create_all_figures, create_all_figures_2025

# 2025 State-of-the-Art imports
try:
    from src.explainability import (
        SHAPExplainer,
        PermutationImportanceAnalyzer,
        PartialDependenceAnalyzer,
        generate_explainability_report,
        compare_explanations
    )
    HAS_EXPLAINABILITY = True
except ImportError:
    HAS_EXPLAINABILITY = False
    logger.warning("Explainability module not available. Install shap, lime, dice-ml.")


def step_data_prep(config: dict) -> pd.DataFrame:
    """
    Step 1: Data Preparation
    
    Load, clean, and prepare the analytic sample.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 60)
    
    # Get paths
    raw_path = Path(config['paths']['raw_data'])
    processed_path = Path(config['paths']['processed_data'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Get variable lists
    vars = get_variable_lists(config)
    
    # Check for existing processed data
    processed_file = processed_path / "analytic_sample.parquet"

    if processed_file.exists():
        logger.info(f"Loading existing processed data from {processed_file}")
        df = pd.read_parquet(processed_file)
        logger.info(f"Loaded {len(df):,} records")
        return df

    # Check for extracted parquet (from ASCII parsing)
    extracted_file = processed_path / "ecls_extracted.parquet"
    if extracted_file.exists():
        logger.info(f"Loading extracted data from {extracted_file}")
        df = pd.read_parquet(extracted_file)
        logger.info(f"Loaded {len(df):,} records")
    else:
        # Load raw data
        raw_file = raw_path / config['data']['filename']

        if not raw_file.exists():
            logger.error(f"Raw data file not found: {raw_file}")
            logger.info("Please download ECLS-K:2011 data from:")
            logger.info("https://nces.ed.gov/ecls/dataproducts.asp")
            logger.info("Then run: python scripts/parse_ascii_data.py")
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")

        logger.info(f"Loading raw data from {raw_file}")
        df = load_ecls_data(str(raw_file), columns=vars['all'])
    
    # Handle missing values
    logger.info("Handling missing values...")
    df = handle_missing_values(df, config['data']['missing_codes'])
    
    # Create derived variables
    logger.info("Creating derived variables...")
    df = create_race_variable(df, config['variables']['demographics']['race'])
    df = create_ses_variable(df, config['variables']['demographics']['ses'])
    
    # Create at-risk indicators
    percentile = config['variables']['at_risk_percentile']
    for name, col in config['variables']['outcomes'].items():
        logger.info(f"Creating at-risk indicator for {name}...")
        df = create_at_risk_indicator(df, col, percentile)
    
    # Create analytic sample
    outcome_col = f"{config['variables']['outcomes']['reading']}_at_risk"
    df, stats = create_analytic_sample(df, vars['predictors'], outcome_col)
    
    # Save processed data
    df.to_parquet(processed_file)
    logger.info(f"Saved processed data to {processed_file}")
    
    # Save sample statistics
    stats_file = processed_path / "sample_stats.yaml"
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f)
    
    return df


def step_train_models(config: dict, df: pd.DataFrame) -> dict:
    """
    Step 2: Model Training
    
    Train and evaluate ML models.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    # Paths
    models_path = Path(config['paths']['models'])
    models_path.mkdir(parents=True, exist_ok=True)
    tables_path = Path(config['paths']['tables'])
    tables_path.mkdir(parents=True, exist_ok=True)
    
    # Get variables
    vars = get_variable_lists(config)
    outcome_col = f"{config['variables']['outcomes']['reading']}_at_risk"
    
    # Prepare data
    logger.info("Preparing modeling data...")
    X, y, groups = prepare_modeling_data(
        df, vars['predictors'], outcome_col, 'race_ethnicity'
    )
    
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        random_state=config['model']['random_state'],
        test_size=config['model']['test_size'],
        cv_folds=config['model']['cv_folds']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Also split groups
    groups_train = groups.loc[X_train.index].reset_index(drop=True)
    groups_test = groups.loc[X_test.index].reset_index(drop=True)
    
    # Train models
    logger.info("Training models...")
    model_names = [
        name for name, cfg in config['model']['algorithms'].items()
        if cfg.get('enabled', True)
    ]
    
    cv_results = trainer.train_all_models(X_train, y_train, model_names)
    
    # Evaluate models
    logger.info("Evaluating models on test set...")
    results_df = trainer.evaluate_all_models(X_test, y_test)
    
    # Save results
    results_df.to_csv(tables_path / "model_performance.csv")
    logger.info(f"Model results:\n{results_df}")
    
    # Get predictions from best model
    best_model_name = results_df['auc_roc'].idxmax()
    best_model = trainer.models[best_model_name]
    
    y_pred, y_prob = trainer.get_predictions(best_model, X_test)
    
    # Feature importance
    logger.info("Computing feature importance...")
    importance_df = get_feature_importance(best_model, list(X.columns))
    importance_df.to_csv(tables_path / "feature_importance.csv", index=False)
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_path / f"best_model_{best_model_name}_{timestamp}.joblib"
    trainer.save_model(best_model, str(model_file))
    
    # Create model card
    model_card = create_model_card(
        best_model_name,
        results_df.loc[best_model_name].to_dict(),
        {"cv_results": cv_results[best_model_name]},
        str(models_path / f"model_card_{best_model_name}.json")
    )
    
    return {
        'trainer': trainer,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'groups_test': groups_test,
        'feature_importance': importance_df,
        'results_df': results_df
    }


def step_explainability_analysis(config: dict, model_results: dict) -> dict:
    """
    Step 3a: Explainability Analysis (2025 State-of-the-Art)

    Compute SHAP values, permutation importance, and fairness-aware explanations.
    """
    logger.info("=" * 60)
    logger.info("STEP 3a: EXPLAINABILITY ANALYSIS (2025)")
    logger.info("=" * 60)

    if not HAS_EXPLAINABILITY:
        logger.warning("Explainability module not available. Skipping.")
        return {}

    # Paths
    tables_path = Path(config['paths']['tables'])
    figures_path = Path(config['paths']['figures'])
    reports_path = Path(config['paths']['results']) / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    trainer = model_results['trainer']
    best_model = model_results['best_model']
    X_test = model_results['X_test']
    y_test = model_results['y_test']
    groups = model_results['groups_test']

    # Scale X_test the same way as during training
    X_test_scaled = pd.DataFrame(
        trainer.scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    results = {}

    try:
        # SHAP Analysis
        logger.info("Computing SHAP values...")
        shap_explainer = SHAPExplainer(
            best_model,
            X_test_scaled,
            feature_names=list(X_test.columns)
        )
        shap_explainer.compute_shap_values(X_test_scaled)

        # SHAP feature importance
        shap_importance = shap_explainer.get_feature_importance(top_n=20)
        shap_importance.to_csv(tables_path / "shap_importance.csv", index=False)
        results['shap_importance'] = shap_importance
        logger.info(f"Top SHAP features:\n{shap_importance.head(10)}")

        # SHAP summary plot
        shap_explainer.plot_summary(
            X_test_scaled,
            output_path=str(figures_path / "shap_summary.png")
        )

        # Fairness-aware SHAP
        logger.info("Computing fairness-aware SHAP...")
        fairness_shap = shap_explainer.fairness_aware_shap(X_test_scaled, groups)
        results['fairness_shap'] = fairness_shap

        # Save per-group SHAP importance
        for group, df in fairness_shap.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(tables_path / f"shap_importance_{group}.csv", index=False)

    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        results['shap_importance'] = None
        results['fairness_shap'] = None

    try:
        # Permutation Importance with bootstrap CI
        logger.info("Computing permutation importance with bootstrap CI...")
        perm_analyzer = PermutationImportanceAnalyzer(
            best_model, X_test_scaled, y_test.values
        )
        perm_importance = perm_analyzer.compute_importance(n_repeats=30)
        perm_importance.to_csv(tables_path / "permutation_importance.csv", index=False)
        results['perm_importance'] = perm_importance

        # Bootstrap CI
        perm_bootstrap = perm_analyzer.bootstrap_importance(n_bootstrap=50)
        perm_bootstrap.to_csv(tables_path / "permutation_importance_bootstrap.csv", index=False)
        results['perm_bootstrap'] = perm_bootstrap

        logger.info(f"Top permutation importance:\n{perm_importance.head(10)}")

    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        results['perm_importance'] = None

    # Compare explanations
    if results.get('shap_importance') is not None and results.get('perm_importance') is not None:
        logger.info("Comparing explanation methods...")
        comparison = compare_explanations(
            results['shap_importance'],
            results['perm_importance']
        )
        comparison.to_csv(tables_path / "explanation_comparison.csv", index=False)
        results['explanation_comparison'] = comparison

    logger.info("Explainability analysis complete")
    return results


def step_fairness_analysis(config: dict, model_results: dict) -> dict:
    """
    Step 3b: Fairness Analysis (2025 State-of-the-Art)

    Evaluate algorithmic fairness with confidence intervals,
    intersectional analysis, and calibration fairness.
    """
    logger.info("=" * 60)
    logger.info("STEP 3b: FAIRNESS ANALYSIS (2025)")
    logger.info("=" * 60)

    # Paths
    tables_path = Path(config['paths']['tables'])
    reports_path = Path(config['paths']['results']) / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    y_test = model_results['y_test']
    y_pred = model_results['y_pred']
    y_prob = model_results['y_prob']
    groups = model_results['groups_test']
    X_test = model_results['X_test']

    # Initialize evaluator
    logger.info("Computing fairness metrics...")
    evaluator = FairnessEvaluator(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        groups=groups,
        reference_group=config['fairness']['reference_groups']['race']
    )

    # Compute metrics
    evaluator.compute_all_group_metrics()
    evaluator.compute_fairness_metrics()

    # Save tables
    summary_df = evaluator.get_summary_table()
    summary_df.to_csv(tables_path / "fairness_group_metrics.csv", index=False)
    logger.info(f"Group metrics:\n{summary_df}")

    disparity_df = evaluator.get_disparity_table()
    disparity_df.to_csv(tables_path / "fairness_disparities.csv", index=False)
    logger.info(f"Disparity metrics:\n{disparity_df}")

    # Check criteria
    criteria = evaluator.check_fairness_criteria()
    logger.info(f"Fairness criteria: {criteria}")

    # === 2025 State-of-the-Art Additions ===

    # 1. Bootstrap Confidence Intervals
    logger.info("Computing bootstrap confidence intervals for fairness metrics...")
    ci_analyzer = FairnessConfidenceIntervals(
        y_test.values, y_prob, groups, n_bootstrap=200
    )
    metrics_with_ci = ci_analyzer.bootstrap_group_metrics()
    metrics_with_ci.to_csv(tables_path / "fairness_metrics_with_ci.csv", index=False)
    logger.info(f"Metrics with CI:\n{metrics_with_ci}")

    # 2. Calibration Fairness (Sufficiency)
    logger.info("Analyzing calibration fairness...")
    calib_analyzer = CalibrationFairnessAnalyzer(y_test.values, y_prob, groups)
    calibration_results = calib_analyzer.check_sufficiency()
    calibration_df = calibration_results['calibration_by_group']
    calibration_df.to_csv(tables_path / "calibration_fairness.csv", index=False)
    logger.info(f"Calibration by group:\n{calibration_df}")
    logger.info(f"Sufficiency satisfied: {calibration_results['satisfies_sufficiency']}")

    # 3. Intersectional Fairness (if SES available)
    intersectional_df = None
    try:
        # Create protected attributes DataFrame
        protected_df = pd.DataFrame({'race': groups})

        # Try to add SES if available in X_test
        ses_cols = [c for c in X_test.columns if 'ses' in c.lower() or 'X1SESQ5' in c]
        if ses_cols:
            protected_df['ses'] = X_test[ses_cols[0]].values

        if len(protected_df.columns) > 1:
            logger.info("Computing intersectional fairness...")
            intersect_analyzer = IntersectionalFairnessAnalyzer(
                y_test.values, y_pred, y_prob, protected_df, min_group_size=20
            )
            intersect_analyzer.compute_intersectional_metrics()
            intersectional_df = intersect_analyzer.get_summary_table()
            intersectional_df.to_csv(tables_path / "intersectional_fairness.csv", index=False)
            logger.info(f"Intersectional analysis:\n{intersectional_df}")

            # Worst subgroups
            worst = intersect_analyzer.get_worst_subgroups(metric='tpr', n=5)
            logger.info(f"Worst TPR subgroups: {[w.subgroup for w in worst]}")
    except Exception as e:
        logger.warning(f"Intersectional analysis failed: {e}")

    # Bias mitigation
    logger.info("Applying bias mitigation (threshold adjustment)...")
    optimizer = ThresholdOptimizer(constraint='equal_opportunity')
    y_pred_adjusted = optimizer.fit_transform(y_test.values, y_prob, groups)

    # Compare before/after
    comparison_df = compare_before_after_mitigation(
        y_test.values, y_pred, y_pred_adjusted, y_prob, groups
    )
    comparison_df.to_csv(tables_path / "mitigation_comparison.csv", index=False)
    logger.info(f"Mitigation comparison:\n{comparison_df}")

    # Generate comprehensive report
    report = generate_fairness_report(
        evaluator,
        str(reports_path / "fairness_report.txt")
    )

    return {
        'evaluator': evaluator,
        'summary_df': summary_df,
        'disparity_df': disparity_df,
        'comparison_df': comparison_df,
        'y_pred_adjusted': y_pred_adjusted,
        'thresholds': optimizer.thresholds,
        # 2025 additions
        'metrics_with_ci': metrics_with_ci,
        'calibration_df': calibration_df,
        'calibration_results': calibration_results,
        'intersectional_df': intersectional_df
    }


def step_generate_figures(
    config: dict,
    model_results: dict,
    fairness_results: dict,
    explainability_results: dict = None
):
    """
    Step 4: Generate Figures (2025 State-of-the-Art)

    Create publication-quality figures including new 2025 visualizations.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING FIGURES (2025)")
    logger.info("=" * 60)

    figures_path = Path(config['paths']['figures'])
    figures_path.mkdir(parents=True, exist_ok=True)

    # Create all figures including 2025 state-of-the-art
    saved_files = create_all_figures_2025(
        y_true=model_results['y_test'].values,
        y_pred=model_results['y_pred'],
        y_prob=model_results['y_prob'],
        groups=model_results['groups_test'],
        output_dir=str(figures_path),
        feature_importance=model_results.get('feature_importance'),
        metrics_with_ci=fairness_results.get('metrics_with_ci'),
        intersectional_df=fairness_results.get('intersectional_df'),
        calibration_df=fairness_results.get('calibration_df'),
        explanation_comparison=explainability_results.get('explanation_comparison') if explainability_results else None,
        fairness_shap=explainability_results.get('fairness_shap') if explainability_results else None,
        model_results=model_results.get('results_df')
    )

    logger.info(f"Generated {len(saved_files)} figures")


def run_full_pipeline(config_path: str):
    """Run the complete analysis pipeline (2025 State-of-the-Art)."""
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE (2025 STATE-OF-THE-ART)")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 60)

    # Load config
    config = load_config(config_path)

    # Create output directories
    for path_key in ['results', 'figures', 'tables', 'models']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Run pipeline
    df = step_data_prep(config)
    model_results = step_train_models(config, df)

    # 2025: Explainability Analysis (SHAP, Permutation Importance)
    explainability_results = step_explainability_analysis(config, model_results)

    # 2025: Enhanced Fairness Analysis (CI, Intersectional, Calibration)
    fairness_results = step_fairness_analysis(config, model_results)

    # 2025: Enhanced Figures
    step_generate_figures(config, model_results, fairness_results, explainability_results)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE (2025 STATE-OF-THE-ART)")
    logger.info("=" * 60)

    return {
        'data': df,
        'model_results': model_results,
        'explainability_results': explainability_results,
        'fairness_results': fairness_results
    }


def main():
    parser = argparse.ArgumentParser(description="ECLS Fairness Study Pipeline")
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--step', '-s',
        choices=['data_prep', 'train_models', 'fairness_analysis', 'figures', 'all'],
        default='all',
        help='Pipeline step to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.step == 'all':
            run_full_pipeline(args.config)
        else:
            config = load_config(args.config)
            
            if args.step == 'data_prep':
                step_data_prep(config)
            else:
                logger.info("For individual steps after data_prep, use notebooks or run 'all'")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
