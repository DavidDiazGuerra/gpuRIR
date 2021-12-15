# gpuRIR extensions
Expands the functionality of gpuRIR by multiple optional extensions, listed as follows:

* Frequency dependent absorption coefficients (virtual room materials)
* Air absorption (via Bandpassing or STFT)
* Source and receiver characteristics (virtual mic / speaker models
* Binaural receiver (two receivers in room)
* Head-related transfer function (HRTF, simulated human hearing)
* Room parameter class for gpuRIR (consolidating and automatically calculating parameters)
* Easy impulse response (IR) file generation with stereo support

We have example usages of these features in our example scripts:
`examples/mono_filters.py` and `examples/stereo_filters.py`.

## Frequency dependent absorption coefficients

Models the impact different wall materials have on the reflective properties of a room. Depending on the material, different frequency bands are absorbed more or less. More materials can be defined manually in `wall_absorption/materials.py`.

We divide up the RIR into frequency bands using bandpass filters with high- and
lowpass filters on the edges. The frequency bands in the lower frequencies are
narrower in order to apply wall absorption data accurately in those lower
registers. This is because the source data is defined up to 4kHz in our case.

### Usage
Set `freq_dep_abs_coeff` to `True` in order to use this feature.
Add six materials (`mat.name_of_material`) to `wall_materials` array corresponding to left wall, right wall, front wall, back wall, floor, ceiling.

### Relevant scripts 

* `gpuRIR/extensions/wall_absorption/freq_dep_abs_coeff.py`
* `gpuRIR/extensions/wall_absorption/materials.py`

### Parameters
* **params** gpuRIR parameter collection.
* **band_width** Starting frequency width of band. The smaller the value the less performant but more realistic the algorithm is. Must be greater than 1.
* **factor** By what factor the starting `band_width` is multiplied each round. The smaller the value, the less performant but more realistic the algorithm is. Must be greater than 1.
* **order** Order of LR or Butterworth filter. If LR is used, the order is internally halved and 2 bandpass filters are used.
* **LR** If true, uses Linkwitz-Riley (LR) filtering. If false, use Butterworth filtering. 
* **use_bandpass_on_borders** Enables bandpass filters on the lowest and highest frequency band instead of using high- and lowpass filters.
* **visualize** Plot the band borders and spectrogram.
* **verbose** Print debugging information.

# Filters
### Usage of filters
Instantiate new filters inside the `filters` array on the `mono_filters.py` example script. Some commented-out code are placed there as a guide. All filters have pre-defined parameters as standard values, but can be overridden with user values.

The filters are applied in the order the user provided, topmost filters are applied first.

The array can be left empty if no filters are to be applied.

**Example**
Following is a simple example with a tiny simulated speaker as source, air absorption and a simulated microphone as receiver.
```
filters = [
            CharacteristicFilter(model.tiny_speaker)
            AirAbsBandpass(),
            CharacteristicFilter(model.sm57_freq_response, params.fs)
        ]
```

## Air absorption
Applying air absorption using bandpass or STFT (Short Time Fourier Transformation) methods.

### Relevant scripts 

* `gpuRIR/extensions/filters/air_absorption_bandpass.py`
* `gpuRIR/extensions/filters/air_absorption_calculation.py`
* `develop/gpuRIR/extensions/filters/air_absorption_stft.py`

### Parameters
* **f** Frequency of pure tone.
* **T** Temperature(degrees Celsius)
* **hr** Relative humidity
* **ps** Atmospheric pressure ratio. 

### Bandpass specific parameters
* **max_frequency** Where the last band stops filtering (band- or highpass).
* **min_frequency** Where the first band starts filtering (band- or lowpass).
* **divisions** Into how many bands to separate the spectrum between min- and max frequency.
* **fs** Sampling rate (must match source file).
* **order** Butterworth order
* **LR** Enables Linkwitz-Riley filter.
* **use_bandpass_on_borders** Uses bandpass instead of high/lowpass filters on lowest and highest band.
* **verbose** Terminal output for debugging or further information.
* **visualize** Plots band divisions in a graph.

### STFT specific parameters
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** Determines how big the Hann window is.

## CharacteristicFilter: Source and receiver characteristics
Allows the simulation of microphone and speaker frequency responses, for example using a small bluetooth speaker as a source and a phone microphone as the receiver. 
Uses 1D interpolation to create models, and STFT (Short Time Fourier Transformation) to apply the modelled and interpolated characteristics (defined in `filters/characteristic_models.py`). More microphone and speaker models can be defined in `filters/characteristic_models.py` by translating a standard frequency response graph into an array.

### Relevant scripts

* `gpuRIR/extensions/filters/characteristic_filter.py`
* `gpuRIR/extensions/filters/characteristic_models.py`

### Parameters
* **model** Model of speaker/microphone (examples defined in *filters/characteristic_models.py*)
* **fs** Sampling rate (must match source file)
* **nFFT** Length of fourier transformations used in number of samples
* **noverlap** How much overlap in samples the segments have
* **window** Determines how big the Hann window is.
* **visualize** Plots the interpolated frequency response to aid comparison to source data.

## Linear Filter

Allows the simulation of microphone and speaker frequency responses, but using discrete linear filters.

### Relevant scripts

* `gpuRIR/extensions/filters/linear_filter.py`

### Parameters
* **numtabs** Length of the filter.
* **bands** Number of frequency bands.
* **desired** Desired behavior of filter.
* **fs** Sampling rate (must match source file).
* **visualize** Plots the filter.

More info on those parameters can be found in the SciPy documentation on [firls](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html).

## Head-related transfer function (HRTF)

We use the [CIPIC
database](https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/) to apply
the effect of the HRTF to RIRs generated by gpuRIR. This is accomplished by
applying the HRIR corresponding to the azimuth and elevation of the direct path
to the source. The `HRTF_Filter` is best used with our `BinauralReceiver`,
which simulates human ears as two receivers placed in the room.

Please see `examples/stereo_filters.py` for an example on how these two classes
are used.

### Relevant scripts

* `gpuRIR/extensions/filters/hrtf_filter.py`
* `gpuRIR/extensions/hrtf/hrtf_binaural_receiver.py`
* `gpuRIR/extensions/hrtf/hrtf_rir.py`

### Parameters
* **channel** Which channel (side) to fetch from the database. Options are 'l' for left or 'r' for right.
* **params** An instance RoomParameters used for simulateRIR. Mostly used for head and source positions.
* **verbose** Enable verbose mode to print debugging info.
