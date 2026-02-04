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
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import joblib

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
    generate_fairness_report
)
from src.visualization import create_all_figures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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


def step_fairness_analysis(config: dict, model_results: dict) -> dict:
    """
    Step 3: Fairness Analysis
    
    Evaluate algorithmic fairness and apply mitigation.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: FAIRNESS ANALYSIS")
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
    
    # Generate report
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
        'thresholds': optimizer.thresholds
    }


def step_generate_figures(config: dict, model_results: dict, fairness_results: dict):
    """
    Step 4: Generate Figures
    
    Create publication-quality figures.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING FIGURES")
    logger.info("=" * 60)
    
    figures_path = Path(config['paths']['figures'])
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Create figures
    saved_files = create_all_figures(
        y_true=model_results['y_test'].values,
        y_pred=model_results['y_pred'],
        y_prob=model_results['y_prob'],
        groups=model_results['groups_test'],
        output_dir=str(figures_path),
        feature_importance=model_results.get('feature_importance')
    )
    
    logger.info(f"Generated {len(saved_files)} figures")


def run_full_pipeline(config_path: str):
    """Run the complete analysis pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE")
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
    fairness_results = step_fairness_analysis(config, model_results)
    step_generate_figures(config, model_results, fairness_results)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return {
        'data': df,
        'model_results': model_results,
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
