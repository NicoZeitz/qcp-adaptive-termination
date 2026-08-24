#!/usr/bin/env bash
############################################################ GENERAL SETUP ############################################################
set -euo pipefail

# set current working directory to the script's parent directory (project dir)
cd "$(dirname "$0")/.."

# sync project dependencies with uv
uv sync

# activate the virtual environment
source .venv/bin/activate

USER=$(whoami)
ASSET_USER="nzeitz"

PROJECT_DIR="/mnt/labstore/${USER}/qcp-adaptive-termination"
CACHE_DIR="/mnt/labstore/${ASSET_USER}/.cache"

BASE_DATASET_DIR="/mnt/labstore/${ASSET_USER}/dataset"
BASE_LOG_DIR="/mnt/labstore/${USER}/.logs"

############################################################ DATASET SPECIFIC SETUP ############################################################
DATASET_NAME="wikidata-federated-overlap-dataset"

DATASET_DIR="${BASE_DATASET_DIR}/${DATASET_NAME}"
DATA_DIR="${DATASET_DIR}"
INPUT_DIR="${DATASET_DIR}/input"
############################################################ RUN SETUP ############################################################
RUN_NAME="evaluate-selection-strategies"

# TODO: Set to true to run files in parallel, false for sequential execution
PARALLEL=true
# TODO: set to true, then only print out the commands that would be executed, without actually running them
TEST=false

# write stdout and stderr to log files (in addition to printing to console)
mkdir -p "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}"
# exec > >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stdout.log") 2> >(tee -i "${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/stderr.log" >&2)

# Load all input files from directory
shopt -s nullglob
FILES=("${INPUT_DIR}"/*.json)
FILES=("${FILES[@]#${INPUT_DIR}/}")
shopt -u nullglob

# Or use a specific list of files
FILES=(
    "How_many_countries_have_a_human_development_index_above_08.json"
    "How_many_NATO_members_have_a_GDP_less_than_fifty_billion.json"
    "How_many_people_live_in_the_middle_east.json"
    "Since_how_many_years_do_bundesliga_clubs_exist_on_average_Today_is_the_20022032.json"
    "What_is_the_average_area_in_square_kilometers_of_Chinese_provinces.json"
    "What_is_the_average_attendance_of_NBA_matches.json"
    "What_is_the_average_capacity_of_UEFA_Champions_League_venues.json"
    "What_is_the_average_life_expectancy_in_countries_with_a_population_greater_than_.json"
    "What_is_the_average_maximum_passenger_capacity_of_all_aircraft.json"
    "What_is_the_average_number_of_participants_in_Mens_FIFA_World_Cup.json"
    "What_is_the_average_passenger_capacity_of_all_aircraft_manufactured_by_Saab_AB.json"
    "What_is_the_average_population_of_German_states.json"
    "What_is_the_combined_area_of_all_continents_in_square_kilometers.json"
    "What_is_the_largest_province_in_China_by_area.json"
    "What_is_the_median_population_of_German_states_Which_state_is_the_closest_to_it.json"
    "What_is_the_minimum_and_maximum_population_of_German_states.json"
    "What_is_the_most_populated_state_in_east_germany.json"
    "What_is_the_total_area_in_square_kilometers_of_all_west_german_states.json"
    "What_is_the_total_population_of_all_NATO_member_countries.json"
    "What_is_the_total_population_of_countries_with_a_GDP_above_1_trillion_USD.json"
    "What_was_the_combined_area_of_the_German_states_that_were_part_of_the_DDR_measur.json"
    "Which_aircraft_has_the_highest_maximum_passenger_capacity.json"
    "Which_airline_has_the_largest_fleet.json"
    "Which_Champions_League_venue_has_the_highest_capacity.json"
    "Which_continent_has_the_second_largest_total_population.json"
    "Which_country_has_the_largest_population_among_countries_with_a_life_expectancy_.json"
    "Which_country_has_the_lowest_elevation_What_is_the_name_and_the_depth_of_this_po.json"
    "Which_football_club_has_the_most_social_media_followers_across_the_top_five_euro.json"
    "Which_German_state_has_the_largest_area.json"
    "Which_NBA_team_has_the_largest_number_of_followers.json"
    "Which_province_is_the_northernmost_province_in_China.json"
    "Which_Womens_World_cup_had_the_most_goals_scored.json"
)

############################################################ RUN EVERYTHING ############################################################

echo "================== Running '${RUN_NAME}' with dataset '${DATASET_NAME}'"
for FILE in "${FILES[@]}"; do

    INPUT_FILE="${INPUT_DIR}/${FILE}"
    QUERY=$(jq -r '.value.query' "${INPUT_FILE}")
    QUERY_SLUG=$(basename "${INPUT_FILE}" .json)

    # take first 2 letter from $USER and uppercase them
    USR=$(echo "${USER:0:2}" | tr '[:lower:]' '[:upper:]')
    LOG_DIR="${BASE_LOG_DIR}/${DATASET_NAME}/${RUN_NAME}/${QUERY_SLUG}"
    NAME="[$USR] ${RUN_NAME} (${DATASET_NAME}): ${QUERY}"

    echo "  ================== Running file '${FILE}'"
    mkdir -p "${LOG_DIR}"

    if [[ "${TEST}" == "true" ]]; then
        # echo "      [TEST] Would run: python3 -m scripts.evaluate_selection_strategies --name \"${NAME}\" --input \"${INPUT_FILE}\" --output_dir \"${LOG_DIR}\" --data_dir \"${DATA_DIR}\" --start-stage 3 --end-stage 3"
        continue
    fi

    python3 -m scripts.evaluate_selection_strategies \
      --name "${NAME}" \
      --input "${INPUT_FILE}" \
      --output_dir "${LOG_DIR}" \
      --data_dir "${DATA_DIR}" \
      --start-stage 3 --end-stage 3 &
    #   > >(tee "${LOG_DIR}/stdout.log") \
    #   2> >(tee "${LOG_DIR}/stderr.log" >&2) &

    if [[ "${PARALLEL}" != "true" ]]; then
        wait
    fi
done

if [[ "${PARALLEL}" == "true" ]]; then
    wait
fi
