# shell-ai-hackaton-2023
Data and code related to [the Shell AI Hackaton 2023](https://www.hackerearth.com/challenges/competitive/shellai-hackathon-2023/)

## Installation
First clone the repository:
```bash
git clone https://github.com/akalikadien/shell-ai-hackaton-2023.git
```

Next, install or load conda and create a new environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate shell-ai-hackaton
```

## Data

* **Biomass_History.csv**:  A time-series of biomass availability in the state of Gujarat from year 2010 to 2017. We have considered arable land as a map of 2418 equisized grid blocks (harvesting sites). For ease of use, we have flattened the map and provided location index, latitude, longitude, and year wise biomass availability for each harvesting site
* **Distance_Matrix.csv**:  The travel distance from source grid block to destination grid block, provided as a 2418 x 2418 matrix. Note that this is not a symmetric matrix due to U-turns, one-ways etc. that may result into different distances for ‘to’ and ‘fro’ journey between source and destination.
* **sample_submission.csv**: Contains sample format for submission

## Code


## Citations for data
Rajeevan, M., Jyoti Bhate, A.K.Jaswal, 2008 : Analysis of variability and trends of extreme rainfall events over India using 104 years of gridded daily rainfall data, Geophysical Res. Lttrs, Vol.35, L18707, doi:10.1029/2008GL035143.