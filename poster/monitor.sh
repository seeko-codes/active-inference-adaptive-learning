#!/bin/bash
# Visual monitor for AI precompute — makes it obvious the laptop is working

OUTPUT_FILE="/private/tmp/claude-501/-Users-aatutor/39b57a23-7e5e-4eec-84a2-5c0e227ec289/tasks/b4g2skxcw.output"
RESULT_FILE="$HOME/adaptive-learning/poster/results/trajectories_ai.json"
START_TIME=$(date +%s)

# Colors
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
RESET='\033[0m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'

clear

while true; do
    COLS=$(tput cols 2>/dev/null || echo 80)
    ROWS=$(tput lines 2>/dev/null || echo 24)
    clear

    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    SECS=$(( ELAPSED % 60 ))

    # Center helper
    center() {
        local text="$1"
        local len=${#text}
        local pad=$(( (COLS - len) / 2 ))
        [ $pad -lt 0 ] && pad=0
        printf "%${pad}s%s\n" "" "$text"
    }

    # Vertical padding to center content
    CONTENT_HEIGHT=30
    VPAD=$(( (ROWS - CONTENT_HEIGHT) / 2 ))
    [ $VPAD -lt 0 ] && VPAD=0
    for ((i=0; i<VPAD; i++)); do echo; done

    # Check if done
    if [ -f "$RESULT_FILE" ]; then
        echo -e "${GREEN}"
        center ""
        center "============================================================"
        center "||                                                        ||"
        center "||                                                        ||"
        center "||        ####   ###  #   # #####                         ||"
        center "||        #   # #   # ##  # #                             ||"
        center "||        #   # #   # # # # ###                           ||"
        center "||        #   # #   # #  ## #                             ||"
        center "||        ####   ###  #   # #####                         ||"
        center "||                                                        ||"
        center "||          PRECOMPUTE COMPLETE!                           ||"
        center "||                                                        ||"
        center "||      Safe to close the laptop now.                     ||"
        center "||                                                        ||"
        center "============================================================"
        echo -e "${RESET}"
        echo ""
        center "Output: $RESULT_FILE"
        exit 0
    fi

    # Count progress from log output
    PCT=""
    CURRENT=""
    TOTAL=""
    LAST_PROGRESS=$(grep -o 'AI \[[0-9]*/[0-9]*\]' "$OUTPUT_FILE" 2>/dev/null | tail -1)
    if [ -n "$LAST_PROGRESS" ]; then
        CURRENT=$(echo "$LAST_PROGRESS" | grep -o '\[.*/' | tr -d '[/')
        TOTAL=$(echo "$LAST_PROGRESS" | grep -o '/.*\]' | tr -d '/]')
        if [ -n "$CURRENT" ] && [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
            PCT=$(( CURRENT * 100 / TOTAL ))
        fi
    fi

    # Pulsing animation
    FRAME=$(( NOW % 4 ))
    case $FRAME in
        0) PULSE="       *       " ;;
        1) PULSE="     * * *     " ;;
        2) PULSE="   * * * * *   " ;;
        3) PULSE="     * * *     " ;;
    esac

    # Big warning
    echo -e "${BG_RED}${WHITE}"
    center ""
    center "=================================================================="
    center "||                                                              ||"
    center "||                                                              ||"
    center "||    #   #  ###        ###  #    ###   ###  #####  |           ||"
    center "||    ##  # #   #      #   # #   #   # #     #      |          ||"
    center "||    # # # #   #      #     #   #   #  ###  ###     |         ||"
    center "||    #  ## #   #      #   # #   #   #     # #       |         ||"
    center "||    #   #  ###        ###  ###  ###  ###  #####    |          ||"
    center "||                                                              ||"
    center "||             DO  NOT  CLOSE  THIS  LAPTOP                     ||"
    center "||                                                              ||"
    center "||                                                              ||"
    center "=================================================================="
    echo -e "${RESET}"

    echo ""
    echo -e "${CYAN}"
    center "AI SIMULATION RUNNING"
    center "10,000 students  x  7 cores  x  active inference"
    center ""
    center "$PULSE"
    echo -e "${RESET}"

    echo ""

    # Time
    TIME_STR="Elapsed:  ${HOURS}h ${MINS}m ${SECS}s"
    echo -e "${WHITE}"
    center "$TIME_STR"
    echo -e "${RESET}"

    # Progress bar
    if [ -n "$PCT" ]; then
        BAR_WIDTH=$(( COLS - 30 ))
        [ $BAR_WIDTH -gt 80 ] && BAR_WIDTH=80
        [ $BAR_WIDTH -lt 20 ] && BAR_WIDTH=20
        FILLED=$(( PCT * BAR_WIDTH / 100 ))
        EMPTY=$(( BAR_WIDTH - FILLED ))
        BAR=$(printf "%${FILLED}s" | tr ' ' '=')
        SPACE=$(printf "%${EMPTY}s" | tr ' ' '-')

        echo ""
        echo -e "${GREEN}"
        center "[${BAR}${SPACE}]  ${PCT}%"
        center "${CURRENT} / ${TOTAL} students"
        echo -e "${RESET}"

        if [ "$CURRENT" -gt 0 ]; then
            SECS_PER=$(( ELAPSED / CURRENT ))
            REMAINING=$(( SECS_PER * (TOTAL - CURRENT) ))
            REM_H=$(( REMAINING / 3600 ))
            REM_M=$(( (REMAINING % 3600) / 60 ))
            echo -e "${YELLOW}"
            center "ETA:  ~${REM_H}h ${REM_M}m remaining"
            echo -e "${RESET}"
        fi
    else
        echo ""
        echo -e "${DIM}"
        center "Waiting for first progress update..."
        echo -e "${RESET}"
    fi

    echo ""
    echo -e "${DIM}"
    center "Ctrl+C stops this monitor (simulation keeps running)"
    echo -e "${RESET}"

    sleep 2
done
