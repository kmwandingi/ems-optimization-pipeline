# Energy Management System for Optimizing Building Energy Usage under Dynamic Pricing

**Table of Contents**

1. **Executive Summary**
2. **Management Introduction**
3. **Project Context**
     3.1. Background and Problem Statement
     3.2. Goals and Strategic Value
     3.3. Energy Landscape: Netherlands vs. Curaçao
4. **Literature Review**
5. **Project Ideation and Evolution**
6. **System Architecture**
7. **Methodology**
     7.1. Probabilistic Device Modeling
     7.2. MILP Optimization Formulation
     7.3. Uncertainty Handling (Robust & Scenario-Based)
     7.4. Implementation Details
8. **Results and Discussion**
9. **Conclusion and Future Work**
10. **References**
11. **Appendices** (A: Code, B: Full MILP Formulation, C: Additional Results, D: Hyperparameters)

## Executive Summary

This report presents the design and development of an AI-enabled Energy Management System (EMS) aimed at optimizing building energy consumption under dynamic electricity pricing. The EMS is developed with a dual context in mind: it is prototyped and tested in the **Dutch energy market**—characterized by real-time and day-ahead pricing, widespread smart metering, and demand-response programs—while being designed for future deployment in **Curaçao**, a Caribbean island currently using monthly flat tariffs and in the early stages of grid modernization. This dual-track approach ensures the solution is immediately relevant in advanced markets and future-proofed for emerging smart grid environments.

The EMS framework comprises two main components:

1. **EMS Platform (Part 1)**: A modular system architecture that integrates data ingestion, secure communications, and user-centric interfaces. This part addresses system-level challenges of aggregating data from multiple devices (including flexible loads, photovoltaic (PV) generation, and battery storage) and providing robust security and privacy for all communications and controls.

2. **Optimization Engine (Part 2)**: An intelligent scheduling module that combines advanced machine learning with mathematical optimization. The optimization engine uses a two-stage decision process that couples next-day planning with continuous learning:
   - **Next-Day Scheduling**: The system performs optimization once per day using the next day's electricity prices (from the day-ahead market). This produces a complete 24-hour schedule for all controllable devices and storage assets, using the latest probabilistic models of device usage and available forecasts for PV generation and other factors.
   - **Continuous Learning Updates**: As device usage is observed throughout the day, the system updates its internal probability models of user behavior. These updates improve future scheduling by incorporating the latest observed behavior, gradually refining its models over time.

Strategically, the EMS project is part of a broader initiative led by **Ilustre Lab**—a living lab partnership between academia (JADS/TU Eindhoven) and industry (LaNubia Consulting, ROBUST)—to deploy AI solutions for water and energy management in Curaçao. The EMS is a flagship pilot connecting academic research with real-world utility needs. While initial development leverages the data-rich, dynamic pricing environment of the Netherlands, the system is explicitly designed to address local challenges in Curaçao, including:

- **Gradual Pricing Innovation**: Supporting the transition from flat monthly tariffs to more granular dynamic pricing as smart metering and grid modernization are introduced.
- **Energy Poverty Mitigation**: Providing improved consumption awareness and budgeting tools to help households reduce energy bills, thus alleviating energy poverty.
- **Grid Resilience**: Enhancing grid balancing capabilities to integrate higher shares of intermittent renewable generation (solar/wind) and to manage isolated-grid stability challenges.

Preliminary simulations and pilot tests demonstrate substantial benefits. The EMS consistently reduces energy costs by shifting flexible loads to cheaper tariff periods, yielding **electricity bill savings of 12–38%** for households and commercial buildings under dynamic pricing. When on-site solar PV and battery storage are present, the EMS dramatically increases **PV self-consumption** (from ~42% to ~87% of solar generation utilized on-site through intelligent load timing). It also cuts peak grid imports, improving grid stability, and maintains high user comfort/satisfaction (>85% of appliance operations remain within user-preferred time windows). The optimization engine operates efficiently, solving the daily scheduling problem in about **8 seconds per building** on average, and robustly handles uncertainties in user behavior via continuous learning.

In summary, the developed EMS combines advanced **machine learning** (to predict and adapt to energy usage patterns) with **mixed-integer linear programming (MILP)** optimization (to schedule devices and storage for minimum cost) in a novel, integrated framework. It delivers tangible value for building operators (cost savings, resilience to price spikes), for utilities (demand flexibility, new tariff opportunities), and for society (reduced emissions, progress toward energy equity). The system's **agent-based modular design** ensures scalability from single homes to campuses and adaptability across different markets and regulatory environments.

## Management Introduction

### Executive Overview

The Energy Management System (EMS) represents a significant advancement in intelligent energy optimization, delivering substantial cost savings and operational efficiencies for modern buildings. This innovative solution combines state-of-the-art machine learning with robust optimization techniques to intelligently manage energy consumption, particularly for flexible loads, while seamlessly integrating distributed energy resources (DERs) such as photovoltaic panels and battery storage. By automating the scheduling of appliances and storage in response to electricity price signals and learned user habits, the EMS transforms how buildings interact with the power grid.

At its core, the EMS addresses a critical challenge in today's energy landscape: how to balance cost efficiency, user comfort, and system reliability in the face of dynamic electricity pricing and increasing renewable energy penetration. Traditional building energy management is often static or rule-based, unable to adapt to hourly price fluctuations or variability in solar generation. In contrast, the EMS learns from historical usage patterns and adapts to changing conditions, setting it apart from conventional approaches. The result is an autonomous system that can reduce energy costs by leveraging low-price periods, maintain comfort by respecting occupants' typical routines, and support grid stability by smoothing out demand peaks and incorporating renewable generation.

### System Architecture and Components

The EMS architecture is built on a modular, agent-based design that ensures scalability, maintainability, and flexibility. The system is organized into five primary layers, each serving a distinct function:

1. **Data Layer**: Manages all data acquisition, cleaning, and storage. This layer interfaces with IoT sensors, smart meters, and external data sources (e.g., weather APIs), ensuring data consistency and reliability across the system. A lightweight database (using DuckDB) enables efficient queries on high-frequency energy data.

2. **Model Layer**: Hosts the machine learning models that predict device usage patterns, user behavior, and seasonal dynamics. These models form the intelligence behind the optimization process, producing probabilistic forecasts that inform decision-making.

3. **Optimization Layer**: Implements the mathematical scheduling algorithms, primarily a mixed-integer linear programming (MILP) solver that computes cost-effective energy schedules. This layer receives inputs from the Model Layer and forecasts to determine the optimal schedules for the next 24 hours.

4. **Integration Layer**: Handles all external communications and system integrations. It includes API gateways for integration with utility providers and market platforms, communication with weather services for up-to-date forecasts, and links to building management systems for control of devices.

5. **User Interface Layer**: Provides intuitive dashboards and control interfaces for end-users and facility managers. Through web or mobile apps, users can monitor energy usage, cost savings, and system suggestions.

The EMS consists of specialized software agents within these layers, each responsible for a specific aspect of the system's intelligence or control:

- **GlobalOptimizer**: The central optimization engine that coordinates all scheduling decisions across devices.
- **ProbabilityModelAgent**: Continuously learns and updates device usage patterns.
- **FlexibleDeviceAgent**: Manages specific appliance constraints and operations.
- **BatteryAgent/EVAgent**: Handles energy storage devices and their specific requirements.
- **PVAgent**: Manages solar generation forecasting and integration.
- **GridAgent**: Encapsulates grid tariffs and constraints.

## Introduction

Modern energy systems are undergoing rapid transformation driven by increased renewable generation, dynamic electricity pricing, and the proliferation of smart devices. In advanced markets like the Netherlands, electricity prices can fluctuate hourly based on supply and demand, creating opportunities for consumers to reduce costs through demand response and load shifting. However, many buildings today lack automation in energy management; appliances are operated on convenience rather than price signals, causing households and businesses to miss out on potential savings. Meanwhile, the rise of intermittent renewables (solar, wind) brings new challenges: solar panels and wind turbines do not produce energy in sync with consumption needs, leading to periods of surplus and deficit. Without intelligent management, this intermittency can strain grid stability (e.g., sudden solar output drops or peaks).

These trends are compounded by globally rising energy costs (due to fuel price volatility, geopolitical factors, and carbon policies) which put financial pressure on consumers. The energy price spike of 2022 in Europe led to a sharp increase in energy poverty in the Netherlands, with approximately 600,000 households (18% more than in 2020) struggling with energy bills. In this context, **Energy Management Systems (EMS)** for buildings have emerged as a promising solution to optimize energy usage by coordinating when and how devices consume or store electricity.

This project addresses these challenges by developing an **AI-enabled EMS** that automatically schedules home/building loads and battery storage to minimize electricity costs and maximize use of renewable energy, all while respecting user comfort and operational constraints. The EMS leverages real-time data (prices, weather, etc.) and learns from historical usage patterns to make informed decisions, distinguishing it from conventional thermostats or timers. It effectively balances three often competing objectives: **cost efficiency**, **user comfort**, and **grid reliability**. For example, it can delay a water heater or electric vehicle charging from peak price hours to cheaper off-peak times, but only as far as user-defined comfort limits allow (hot water ready by morning, EV charged by departure, etc.). By doing so across all major flexible appliances, the EMS reduces the total energy bill and flattens the demand curve, benefiting both the user and the grid.

The significance of this work is twofold. Academically, it integrates cutting-edge **machine learning** (for probabilistic load forecasting) with **optimization algorithms** (MILP solvers) in a real-world deployment context. It contributes a novel approach to handle uncertainty in appliance scheduling by using learned probability distributions rather than deterministic or worst-case assumptions. From an industrial perspective, the EMS demonstrates a pathway to **smart, autonomous building energy control** that can unlock new business models for utilities (like dynamic tariffs or aggregated demand response) and help end-users actively manage energy costs. An effective EMS can mitigate the impacts of price volatility by automatically shifting consumption away from the most expensive periods, and by integrating local renewables to reduce reliance on the grid during high price periods.

The remainder of this report is structured as follows. **Section 3 (Project Context)** details the background of the project, including the partnership with Ilustre Lab, and defines the problem statement and objectives. It also compares the Dutch and Curaçao energy landscapes to frame the dual-use case. **Section 4 (Literature Review)** surveys existing work on home energy management systems, optimization techniques, and uncertainty handling, identifying how this project builds upon and innovates beyond the state-of-the-art. **Section 5 (Project Ideation and Evolution)** narrates how the project's approach developed over time – starting from initial ideas of a digital twin for energy simulation to the eventual design of a probabilistic optimization engine – and the rationale behind key design decisions. **Section 6 (System Architecture)** describes the EMS's overall architecture, including its modular layers and the agent-based components that manage data, learning, optimization, integration, and user interaction. **Section 7 (Methodology)** dives into the technical implementation: the machine learning models for device usage prediction, the formulation of the MILP optimization problem (including objective function and constraints), the strategies for handling uncertainty (robust optimization and scenario-based planning), and other implementation details such as software stack and integration considerations. **Section 8 (Results and Discussion)** presents the outcomes of simulation experiments and pilot tests, highlighting cost savings, peak reduction, PV utilization, and discusses insights such as the impact of batteries, user preference sensitivity, and how the Dutch results generalize to Curaçao's context. **Section 9 (Conclusion and Future Work)** summarizes the achievements and outlines next steps (e.g., deployment in a live pilot in Curaçao, adding features like peer-to-peer energy trading or more advanced forecasting). Finally, **Section 10 (References)** lists the literature sources, and **Appendices** provide supplementary material including full mathematical formulations and additional result figures.
