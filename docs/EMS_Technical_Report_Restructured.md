# Advanced Energy Management System (EMS) Technical Report

## Table of Contents

0. [Executive Summary](#executive-summary)
1. [Management Introduction](#management-introduction) *(to be generated last)*
2. [Project Context](#project-context)
   - [Project Background](#project-background)
   - [Problem Statement](#problem-statement)
   - [Goal Specification and Added Value](#goal-specification-and-added-value)
   - [Literature Review](#literature-review)
3. [System Design](#system-design)
   - [Functional Requirements](#functional-requirements)
   - [Architecture Design](#architecture-design)
     - [Component Overview](#component-overview)
     - [Data Flow](#data-flow)
     - [Integration Patterns](#integration-patterns)
     - [Agent-Based Structure](#agent-based-structure)
   - [Methodology](#methodology)
     - [Data Preprocessing and Analysis](#data-preprocessing-and-analysis)
     - [Machine Learning for Device Usage Prediction](#machine-learning-for-device-usage-prediction)
     - [Mixed-Integer Linear Programming (MILP) Optimization](#mixed-integer-linear-programming-milp-optimization)
     - [Uncertainty Handling and Robust Optimization](#uncertainty-handling-and-robust-optimization)
     - [Continuous Learning Pipeline](#continuous-learning-pipeline)
     - [MLflow Integration](#mlflow-integration)
   - [Mathematical Formulation](#mathematical-formulation)
     - [Device Optimization Model](#device-optimization-model)
     - [Battery Operation Model](#battery-operation-model)
     - [Global Building Constraints](#global-building-constraints)
     - [Probabilistic Constraint Formulation](#probabilistic-constraint-formulation)
   - [Implementation Details](#implementation-details)
     - [Data Pipeline](#data-pipeline)
     - [Machine Learning Pipeline](#machine-learning-pipeline)
     - [Optimization Service](#optimization-service)
     - [Device Agents](#device-agents)
     - [Battery Agent](#battery-agent)
     - [Deployment Architecture](#deployment-architecture)
4. [Evaluation](#evaluation)
   - [Results and Analysis](#results-and-analysis)
     - [Cost Savings](#cost-savings)
     - [Device Usage Prediction Performance](#device-usage-prediction-performance)
     - [PV Self-Consumption](#pv-self-consumption)
     - [Battery Value](#battery-value)
     - [Method Comparisons](#method-comparisons)
     - [Market Adaptability](#market-adaptability)
   - [Discussion and Insights](#discussion-and-insights)
     - [Key Findings](#key-findings)
     - [Engineering Challenges](#engineering-challenges)
     - [Practical Implementation Considerations](#practical-implementation-considerations)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
   - [Future Directions](#future-directions)
7. [Recommendations](#recommendations)
8. [Bibliography](#bibliography)
   - [References](#references)
9. [Appendices](#appendices)
   - [Appendix A – Code Listings](#appendix-a-code-listings)
   - [Setup and Usage Guide](#setup-and-usage-guide)

## Table A – Abbreviations

| Abbreviation | Full Term | First Location Used |
|--------------|-----------|-------------------|
| EMS | Energy Management System | Executive Summary |
| MILP | Mixed-Integer Linear Programming | Executive Summary |
| PMF | Probability Mass Function | Executive Summary |
| DER | Distributed Energy Resource | Problem Statement |
| PV | Photovoltaic | Executive Summary |
| EV | Electric Vehicle | Executive Summary |
| SOC | State of Charge | Mathematical Formulation |
| API | Application Programming Interface | Implementation Details |
| BMS | Building Management System | System Architecture |
| AUC | Area Under Curve | Results and Analysis |
| JS | Jensen-Shannon | Results and Analysis |
| V2G | Vehicle-to-Grid | Implementation Details |

## Table B – Key Concepts

| Concept | One-line Definition | Section |
|---------|-------------------|---------|
| Agent-Based Architecture | System design using autonomous agents for device-specific optimization | System Architecture |
| Phases Optimization | Production-standard optimization method using device operation phases | Mathematical Formulation |
| Probabilistic Constraints | Soft constraints based on learned probability distributions | Mathematical Formulation |
| DuckDB Integration | Zero-copy analytical database for efficient data processing | Implementation Details |
| MLflow Tracking | Experiment tracking and model lifecycle management | MLflow Integration |
| GlobalOptimizer | Primary optimization agent implementing MILP scheduling | System Architecture |

## Executive Summary

This technical report presents a comprehensive Energy Management System (EMS) designed to optimize the operation of flexible energy devices, batteries, and PV systems in residential and commercial buildings. Building upon established MILP-based energy management frameworks (such as those by Antunes et al., 2022; Bradac et al., 2014; Gerards et al., 2015), our system advances the field by integrating probabilistic modeling of device usage patterns with robust-to-uncertainty mixed-integer linear programming (MILP) optimization techniques.

The system addresses the pressing need for automated energy management in both advanced markets with dynamic pricing (like the Netherlands) and emerging markets (like Curaçao) by integrating machine learning with traditional optimization methods. By learning user behavior patterns and device operational characteristics, the EMS creates personalized schedules that minimize energy costs while maintaining user comfort and respecting system constraints.

Building on the foundation of existing research, our system contributes the following advancements:

1. **Enhanced probabilistic approach to device usage modeling** using gradient-boosted tree models (LightGBM for daily usage prediction, CatBoost for hourly patterns)
2. **Tighter integration of learned probability mass functions (PMFs)** as soft constraints in MILP formulations
3. **Improved handling of PV generation and Electric Vehicle operation uncertainties**
4. **Implementation of a closed feedback loop** that continuously improves device models through Bayesian updates
5. **Comprehensive MLflow integration** for model versioning, tracking, and deployment for both prediction models and optimization models

The results demonstrate consistent cost savings between 10-30% compared to unoptimized operation, increased self-consumption of on-site PV generation, and effective battery value realization through price arbitrage.

## Project Context

### Project Background

### Problem Statement

### Goal Specification and Added Value

### Literature Review