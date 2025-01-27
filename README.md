# TAMSAT-ALERT API for soil moisture (Version 2)

## Summary

The TAMSAT-ALERT API is software (Python code) designed to allow users to easily generate tailored agricultural drought information for their crop growing season using TAMSAT soil moisture estimates and forecasts. The API provides both soil moisture conditions from the start of the season up to a current date in the season as well as forecasts out to the end of the season. The software is designed so that users do not need to edit the Python code and only need provide minimal inputs to allow the API to run, namely:

* Period of interest
* Region of interest
* The current date
* The climatological period from which soil moisture anomalies are derived
* Meteorological tercile forecast weights

With this information, the API will download the required data, compute the drought metrics, apply the forecast weightings and output the drought metrics in several formats, depending on the data type.

## Running the API

As the API is a single Python script which takes arguments, it can be run on the command line or within a Jupyter Notebook (provided in this Git repository).
Full details on how to run the TAMSAT-ALERT API can be found in the document 'TAMSAT-ALERT_API_guide.pdf'. This will provide details on:
* Python installation (via Anaconda)
* Required Python libraries
* API inputs and outputs
* Running the API (command line or Jupyter Notebook)

