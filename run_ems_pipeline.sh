#!/bin/bash
# Script to run the comprehensive integrated EMS pipeline for all building devices

# Default values
BUILDING_ID="DE_KN_residential4"
ENABLE_EV=true
HISTORICAL_DAYS=10
LIVE_DAYS=5
TEST_MODE=false

# Display usage information
function show_usage {
    echo "Usage: $0 [options]"
    echo "Run the comprehensive integrated EMS pipeline for all building devices."
    echo ""
    echo "Options:"
    echo "  -b, --building ID    Building ID to analyze (default: $BUILDING_ID)"
    echo "  --no-ev              Disable EV optimization"
    echo "  -h, --historical N   Number of historical days to use for training (default: $HISTORICAL_DAYS)"
    echo "  -l, --live N         Number of live days to optimize (default: $LIVE_DAYS)"
    echo "  --test               Run in test mode (smaller dataset, more verbose output)"
    echo "  --help               Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                   # Run with default settings"
    echo "  $0 -b DE_KN_residential1             # Run for a different building"
    echo "  $0 --no-ev                           # Run without EV optimization"
    echo "  $0 -h 5 -l 3                         # Use 5 historical days and 3 live days"
    echo "  $0 --test                            # Run in test mode for debugging"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--building)
            BUILDING_ID="$2"
            shift 2
            ;;
        --no-ev)
            ENABLE_EV=false
            shift
            ;;
        -h|--historical)
            HISTORICAL_DAYS="$2"
            shift 2
            ;;
        -l|--live)
            LIVE_DAYS="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build the command
CMD="python scripts/03_ integrated_pipeline.py $BUILDING_ID"

if [ "$ENABLE_EV" = false ]; then
    CMD="$CMD --no-ev"
fi

CMD="$CMD --historical-days $HISTORICAL_DAYS --live-days $LIVE_DAYS"

if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --test"

    # For test mode, use smaller dataset unless explicitly specified
    if [ "$HISTORICAL_DAYS" = "10" ]; then
        HISTORICAL_DAYS=3
        CMD="$CMD --historical-days $HISTORICAL_DAYS"
    fi

    if [ "$LIVE_DAYS" = "5" ]; then
        LIVE_DAYS=2
        CMD="$CMD --live-days $LIVE_DAYS"
    fi
fi

# Display the parameters
echo "=============================================="
echo "Running EMS Pipeline with the following parameters:"
echo "Building ID:       $BUILDING_ID"
echo "EV Optimization:   $([ "$ENABLE_EV" = true ] && echo "Enabled" || echo "Disabled")"
echo "Historical Days:   $HISTORICAL_DAYS"
echo "Live Days:         $LIVE_DAYS"
echo "Test Mode:         $([ "$TEST_MODE" = true ] && echo "Enabled" || echo "Disabled")"
echo "=============================================="

# Run the command
echo "Starting pipeline... (This may take a few minutes)"
# Add the current directory to PYTHONPATH to help find the modules
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Setting PYTHONPATH to include current directory: $PYTHONPATH"
eval $CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "Pipeline completed successfully!"
    echo "Results are available in the following locations:"
    echo "- Schedules:      results/schedules/${BUILDING_ID}_all_schedules.json"
    echo "- Visualizations: results/visualizations/${BUILDING_ID}_*.png"
    echo "- Battery plots:  results/visualizations/battery/${BUILDING_ID}_*.png"
    if [ "$ENABLE_EV" = true ]; then
        echo "- EV plots:       results/visualizations/ev/${BUILDING_ID}_*.png"
    fi
    echo "=============================================="
else
    echo "Pipeline execution failed. Please check the error messages above."
fi