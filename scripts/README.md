# Generate IR with filters

`generate_IR_with_filters` is a script to generate room impulse response (RIRs) files, ready to be used to convolve audio files.
Also provided are filters and features which add further functionalities to RIR generation. Goal of those features are achieving the most realistic reverb quality possible, add sonic flexibility, yet striking a balance with performance to increase gpuRIR's attractiveness to machine learning applications such as de-reverberation.

# Features
## Frequency dependent absorption coefficients
Models the impact different wall materials have on the reflective properties of a room. Depending on the material, different frequency bands are absorbed more or less. More materials can be defined manually in `wall-materials/materials.py`.

Frequency bands are allocated dynamically in a logarithmic manner, and band distribution is parameterized depending to the user's need for performance or quality.

### Usage
Set `freq_dep_abs_coeff` to `True` in order to use this feature.
Add six materials (`mat.name_of_material`) to `wall_materials` array corresponding to left wall, right wall, front wall, back wall, floor, ceiling.

### Parameters
* **params** gpuRIR parameter collection.
* **LR** If true, uses Linkwitz-Riley (LR) filtering. If false, use Butterworth filtering. 
* **order** Order of LR or Butterworth filter.
* **band_width** Starting frequency width of band. The smaller the value the less performant but more realistic the algorithm is. Must be greater than 1.
* **factor*** Multiplication factor of band width. The smaller the value, the less performant but more realistic the algorithm is. Must be greater than 1.

# Filters
### Usage of filters
Instantiate new filters inside the `filters` array on the `generate_IR_with_filters.py` script. Some commented-out code are placed there as a guide. All filters have pre-defined parameters as standard values, but can be overridden with user values.

The filters are applied in the order the user provided, topmost filters are applied first.

## Air absorption
Applying air absorption using bandpass or STFT (Short Time Fourier Transformation) methods.

### Parameters
* **f** Frequency of pure tone.
* **T** Temperature(degrees Celsius)
* **hr** Relative humidity
* **ps** Atmospheric pressure ratio. 

### Bandpass specific parameters
* **order** Butterworth bandpass order (bandpass only)

### STFT specific parameters
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** Determines how big the Hann window is.

## CharacteristicFilter: Source and receiver characteristics
Allows the simulation of microphone and speaker frequency responses, for example using a small bluetooth speaker as a source and a phone microphone as the receiver. 
Uses 1D interpolation to create models, and STFT (Short Time Fourier Transformation) to apply the modelled and interpolated characteristics (defined in `filters/characteristic_models.py`). More microphone and speaker models can be defined in `filters/characteristic_models.py` by translating a standard frequency response graph into an array.

### Parameters
* **model** Model of speaker/microphone (defined in *filters/characteristic_models.py*)
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** Determines how big the Hann window is.
* **plot** Plots the interpolated frequency response to aid comparison to source data.

## Linear Filter
Allows the simulation of microphone and speaker frequency responses, but using discrete linear filters.

### Parameters
* **numtabs** Length of the filter.
* **bands** Number of frequency bands.
* **desired** Desired behavior of filter.
* **fs** Sampling rate (must match source file)
* **plot** Plots the filter to visualize filter.

