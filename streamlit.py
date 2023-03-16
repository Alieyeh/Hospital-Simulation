import streamlit as st
import acutestrokeunit
from acutestrokeunit import AcuteStrokeUnit, single_run, multiple_replications, Scenario


# FILE_NAME = xxmarkdown


def read_in_file(file_name):
    with open(file_name, "r") as f:
        contents = f.read()
    return contents


st.title("Acute stroke unit simulation model")
# st.markdown(read_in_file(FILE_NAME))

with st.sidebar:
    acutestrokeunit.N_BEDS = st.slider("No. Beds", 5, 15)
    acutestrokeunit.DEFAULT_N_REPS = st.slider("No. Replications", 5, 15)
    acutestrokeunit.DEFAULT_RESULTS_COLLECTION_PERIOD = st.number_input("Run length(days)", value=365)
    acutestrokeunit.DEFAULT_WARMUP = st.number_input("Warmup period(days)", value=120)

if st.button("Run the simulation"):
    with st.spinner("Simulation model is running..."):
        default_args = Scenario()
        # create the model
        model = AcuteStrokeUnit(default_args)
        model.run()
        results = model.run_summary_frame()
    st.success("Simulation complete.")
    st.table(results)
