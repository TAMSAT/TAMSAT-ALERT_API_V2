# TAMSAT-ALERT API for soil moisture (Version 2)
Authors: Vicky Boult, Ross Maidment, Emily Black<br>
Institution: TAMSAT Group (https://www.tamsat.org.uk), University of Reading<br>
Contact: Ross Maidment (TAMSAT Operations Lead): r.i.maidment@reading.ac.uk

## Summary

The TAMSAT-ALERT API is software (Python code) designed to allow users to easily generate tailored agricultural drought information for their crop growing season using TAMSAT soil moisture estimates and forecasts. 

The API provides soil moisture conditions (expressed as the water requirement satisfaction index (WRSI)) from:
1)	the start of the season up to a current date in the season (‘WRSI current’)
2)	the start of the season out to the end of the season, with soil moisture forecasts used to extend WRSI out to the end of the season (‘WRSI forecast’)

The software is designed so that users do not need to edit the Python code and only need provide minimal inputs to allow the API to run, namely:

* Period of interest
* Region of interest
* The current date
* The climatological period from which soil moisture anomalies are derived
* Meteorological tercile forecast weights

With this information, the API will download the required data, compute the drought metrics, apply the forecast weightings and output the drought metrics in several formats, depending on the data type.

## Running the API

As the API is a single Python script which takes arguments, it can be run on the command line or within a Jupyter Notebook (provided in this Git repository).
Full details on how to run the TAMSAT-ALERT API can be found in the document 'TAMSAT-ALERT_API_guide.pdf'. This will provide details on:
* API overview and changes since the TAMSAT-ALERT API Version 1 (Sections 1 and 2)
* Prerequisites (Section 3)
* Python installation via Anaconda and required Python libraries (Section 4)
* Running the API (command line or Jupyter Notebook) (Section 5)
* API outputs (Section 6)
* Test cases (Section 7)

